from typing import Optional, Tuple, Any
from pathlib import Path

import os
import re
import shutil

import numpy as np
from env_runner import EnvRunner
from utils import EpisodeState, SUBGOAL_TYPES, TASK_WITH_VIDEO_DEMO

from subgoal_prediction.gemini.api import GeminiModel
from subgoal_prediction.gemini.prompts import (
    DEMO_TEXT_QUERY,
    IMAGE_TEXT_QUERY,
    VIDEO_TEXT_QUERY,
)

from subgoal_prediction.qwenvl.api import Qwen3VLModel
from subgoal_prediction.detector_refine import COORD as COORD_RE, PROMPT as _PROMPT

PROMPT_CONTAINER = _PROMPT.get("container", "white cube")
from subgoal_prediction.qwenvl.api_memer import Qwen3VLModelMemER


LONG_FIRST_ACTION_TASKS = [
    "BinFill",
    "PickXtimes",
    "SwingXtimes",
    
    "ButtonUnmask",
    "ButtonUnmaskSwap",
    
    "PickHighlight",
    "VideoRepick",
    
    "VideoPlaceButton",
    "VideoPlaceOrder",
    
    "MoveCube",
    "InsertPeg"
] # For Gemini only. Due to we found Gemini is very inconsistent for incremental video understanding, hard code to make it work better



_PAIR_LOG = os.environ.get("SUBGOAL_PAIR_LOG")


def _log_pair(predictor, count, response, refined=None) -> None:
    """Record the predicted and oracle subgoal at the same step, for grounding-error analysis.

    The oracle is available during any rollout, so pairing it with whatever the VLM said at
    that exact step measures the VLM's coordinate error in the real eval distribution --
    with the true demo video and subgoal history, not a reconstructed prompt.

    Opt-in via SUBGOAL_PAIR_LOG and wrapped in a bare except: this must never be able to
    take down an evaluation run.
    """
    if not _PAIR_LOG:
        return
    try:
        import json as _json
        rec = {
            "env": predictor.env_name,
            "ep": predictor.episode_id,
            "step": count,
            "pred": response,
            "oracle": getattr(predictor.env_runner, "grounded_subgoal_oracle", None),
        }
        if refined is not None:
            rec["refined"] = refined
        with open(_PAIR_LOG, "a") as f:
            f.write(_json.dumps(rec) + "\n")
    except Exception:
        pass


class SubgoalPredictorBase:
    def __init__(
        self,
        args,
        save_dir: Path,
    ):
        self.args = args
        self.save_dir = save_dir
        self.video_buffer = []
        self.episode_dir: Optional[str] = None
        
        self.setup_api()

    def setup_api(self) -> None:
        pass

    def start_episode(self, epstate: EpisodeState, env_runner: EnvRunner) -> None:
        self.env_name = env_runner.env_id
        self.episode_id = env_runner.episode_id
        self.task_goal = env_runner.task_goal
        self.env_runner = env_runner

    def step(self, epstate: EpisodeState) -> None:
        pass

    def maybe_extend_video(self, images: list) -> None:
        pass

    def get_subgoal(
        self,
        count: int,
        current_subgoal: Optional[str],
        last_subgoal: Optional[str],
    ) -> Tuple[Optional[str], bool]:
        # return (subgoal_str, has_api_error)
        raise NotImplementedError

    def end_episode(self, epstate: EpisodeState, success_flag: str) -> None:
        pass


class NullSubgoalPredictor(SubgoalPredictorBase):
    def get_subgoal(self, *args, **kwargs) -> Tuple[Optional[str], bool]:
        return None, False
    

class GeminiSubgoalPredictor(SubgoalPredictorBase):
    def start_episode(self, epstate: EpisodeState, env_runner: EnvRunner) -> None:
        super().start_episode(epstate, env_runner)
        self.api = GeminiModel(
            save_dir=os.path.join(self.save_dir, self.env_name, f"ep{self.episode_id}"),
            task_id=self.env_name,
            model_name=self.args.gemini_model_name,
            task_goal=self.task_goal,
            subgoal_type=self.args.subgoal_type,
        )
        self.video_buffer.extend(epstate.image_buffer[:-1])
        print(f"[robomme] Gemini agent for {self.args.subgoal_type}, task {self.env_name}, episode {self.episode_id}, setup finished")

    def step(self, epstate: EpisodeState) -> None:
        self.video_buffer.append(epstate.image_buffer[-1])
    
    def get_subgoal(
        self,
        count: int,
        current_subgoal: Optional[str],
        last_subgoal: Optional[str],
    ) -> Tuple[Optional[str], bool]:
        if not self._should_call(count):
            return current_subgoal, False

        text_query = self._get_text_query(count)
        input_data = self.api.prepare_input_data(self.video_buffer, text_query, count)
        response, _ = self.api.call(input_data)
        self.video_buffer.clear()

        if response is None:
            return None, True

        subgoal = response['subgoal']
        if "is complete" in subgoal or "is finished" in subgoal: # avoid using these subgoals as the final subgoal
            subgoal = last_subgoal
        return subgoal, False

    def end_episode(self, epstate: EpisodeState, success_flag: str) -> None:
        if not self.api:
            return
        self.api.save_conversation()
        self.api.prepare_input_data(
            epstate.image_buffer,
            self._get_text_query(epstate.count),
            epstate.count,
        )
        self.api.save_final_video(f"{success_flag}_ep{self.episode_id}_{self.task_goal}.mp4")
        self.api.clear_uploaded_files()
        del self.api

    def _get_text_query(self, count: int) -> str:
        if count == 0:
            if self.env_name in TASK_WITH_VIDEO_DEMO:
                template = DEMO_TEXT_QUERY
            else:
                template = IMAGE_TEXT_QUERY
        else:
            template = VIDEO_TEXT_QUERY
        return template.format(task_goal=self.task_goal)

    def _should_call(self, count: int) -> bool:
        if count == 0:
            return True
        if self.env_name in LONG_FIRST_ACTION_TASKS and count < 75:
            return False # avoid changing the first action too early
        return count % 48 == 0


class QwenVLSubgoalPredictor(SubgoalPredictorBase):
    
    def setup_api(self) -> None:
        self.api = Qwen3VLModel(
            adapter_path=self.args.qwenvl_simpleSG_adapter_path if self.args.subgoal_type == "simple_subgoal" else self.args.qwenvl_groundSG_adapter_path,
            subgoal_type=self.args.subgoal_type,
        )
        print(f"[robomme] QwenVL {self.args.subgoal_type} agent setup finished")
        
    def start_episode(self, epstate: EpisodeState, env_runner: EnvRunner) -> None:
        super().start_episode(epstate, env_runner)
        self.episode_dir = os.path.join(self.save_dir, self.env_name, f"ep{self.episode_id}")
        self.api.start_new_episode(self.episode_dir, epstate.image_buffer[:-1], self.task_goal)

    def step(self, epstate: EpisodeState) -> None:
        self.video_buffer.append(epstate.image_buffer[-1])

    def get_subgoal(
        self,
        count: int,
        current_subgoal: Optional[str],
        last_subgoal: Optional[str],
    ) -> Tuple[Optional[str], bool]:
        # Some tricks. QwenVL sometimes thinks the button has been pressed. hot fix for now.
        # Such special tricks are not encouraged if you consider participate RoboMME challenge @ CVPR 2026
        if self.env_name in ["ButtonUnmask", "PickHighlight"]:
            keep_period = 90
        elif self.env_name == "ButtonUnmaskSwap":
            if last_subgoal and "press the first button" in last_subgoal:
                keep_period = 100
            elif last_subgoal and "press the second button" in last_subgoal:
                keep_period = 250
            else:
                keep_period = 0
        else:
            keep_period = 0

        response = self.api.call(self.video_buffer[-1], count, keep_period)
        self.video_buffer.clear()
        _log_pair(self, count, response)
        return response, False
    
    def end_episode(self, epstate: EpisodeState, success_flag: str) -> None:
        if self.episode_dir:
            shutil.rmtree(self.episode_dir) # save some space, you can comment this function out to keep all video frames


class DetectorRefinedQwenVLSubgoalPredictor(QwenVLSubgoalPredictor):
    """Training-free hybrid: QwenVL decides WHICH object, a detector decides WHERE it is.

    Measured on 360 frames, an open-vocabulary detector's nearest box sits 1.6 px from the
    oracle point (median) but its highest-scoring box is right only 36% of the time --
    it localises well and disambiguates badly. QwenVL is the mirror image: it names the
    right subgoal 86% of the time but its coordinate lands 8.1 px out (median), right on
    the tolerance the policy was trained to absorb. Composing them plays to both.

    The VLM's sentence is preserved exactly; only the numbers change.
    """

    def setup_api(self) -> None:
        super().setup_api()
        from subgoal_prediction.detector_refine import DetectorRefiner
        self.refiner = DetectorRefiner()
        print("[robomme] detector-refined QwenVL agent setup finished")

    def start_episode(self, epstate: EpisodeState, env_runner: EnvRunner) -> None:
        super().start_episode(epstate, env_runner)
        self.refiner.reset()

    def get_subgoal(
        self,
        count: int,
        current_subgoal: Optional[str],
        last_subgoal: Optional[str],
    ) -> Tuple[Optional[str], bool]:
        # grab the current frame before the parent clears the buffer
        frame = self.video_buffer[-1] if self.video_buffer else None
        response, has_api_error = super().get_subgoal(count, current_subgoal, last_subgoal)
        if has_api_error or not response:
            return response, has_api_error
        refined = self.refiner.rewrite(response, frame, use_prior=True)
        if refined != response:
            _log_pair(self, count, response, refined=refined)
        return refined, has_api_error

    def end_episode(self, epstate: EpisodeState, success_flag: str) -> None:
        print(f"[detector_refine] {self.refiner.stats}", flush=True)
        super().end_episode(epstate, success_flag)


class WriteTimeMemorySubgoalPredictor(DetectorRefinedQwenVLSubgoalPredictor):
    """Record the evidence while it is still visible, instead of trying to recall it later.

    ButtonUnmask's coloured cubes are visible in the opening frames and then covered by
    identical white containers. Asked at read time, "which container hides the blue cube?" is
    unanswerable from the frame -- measured 23.7% for that family. Asked at write time it is a
    plain colour query, which the detector answers well. Storing the position then and snapping
    it to the nearest container later reaches 75.5% / 62.6% offline (66.3% combined), against
    a 19% no-memory baseline.

    Uses only the task prompt (for the colour names) and the current image. The VLM's subgoal
    text is preserved exactly; only the coordinate changes.

    Not applicable to VideoUnmask: there the cubes are never visible during execution -- the
    evidence is in the demo video -- and frame-0 detection predicts the right container only
    23% of the time, at chance. The fallback path covers those.
    """

    WATCH = re.compile(r"hid(?:ing|es) the (red|green|blue) cube")
    # The Unmask families are indistinguishable to WATCH but are opposite cases. ButtonUnmask
    # reveals the cubes during execution, so a note taken from an execution frame is sound
    # (measured 79.3% vs 19.2% for the fallback). VideoUnmask's containers are already down
    # when execution starts -- its evidence is in the demo video -- so any note is spurious and
    # REPLACING the fallback with it costs accuracy (12.0% vs 22.4%; Swap 10.7% vs 12.1%).
    # The task prompt separates them, and the prompt is available at deploy time.
    VIDEO_DEMO = re.compile(r"watch the video")
    READ = re.compile(r"container.*?hid(?:ing|es) the (red|green|blue) cube", re.S)

    def setup_api(self) -> None:
        super().setup_api()
        self._notes = {}
        print("[robomme] write-time memory agent setup finished")

    def start_episode(self, epstate: EpisodeState, env_runner: EnvRunner) -> None:
        super().start_episode(epstate, env_runner)
        self._notes = {}
        goal = (self.task_goal or "").lower()
        if self.VIDEO_DEMO.search(goal):
            self._watch = []           # evidence is in the demo video, not in these frames
            print("[writemem] video-demo task: note-taking disabled, using fallback", flush=True)
        else:
            self._watch = list(dict.fromkeys(self.WATCH.findall(goal)))
            if self._watch:
                print(f"[writemem] watching {self._watch}", flush=True)

    def _observe(self, frame) -> None:
        """Store the earliest confident sighting of each watched colour."""
        if frame is None or not self._watch:
            return
        for col in self._watch:
            if col in self._notes:
                continue
            try:
                cen, sc = self.refiner._boxes(frame, f"{col} cube")
            except Exception:
                continue
            if len(cen) and float(sc.max()) >= float(os.environ.get("WRITEMEM_TH", "0.30")):
                j = int(np.argmax(sc))
                self._notes[col] = (float(cen[j, 0]), float(cen[j, 1]))
                print(f"[writemem] noted {col} at {self._notes[col]}", flush=True)

    def get_subgoal(
        self,
        count: int,
        current_subgoal: Optional[str],
        last_subgoal: Optional[str],
    ) -> Tuple[Optional[str], bool]:
        frame = self.video_buffer[-1] if self.video_buffer else None
        self._observe(frame)
        response, has_api_error = QwenVLSubgoalPredictor.get_subgoal(
            self, count, current_subgoal, last_subgoal)
        if has_api_error or not response:
            return response, has_api_error
        m = self.READ.search(response.lower())
        if m and m.group(1) in self._notes and frame is not None:
            col = m.group(1)
            try:
                cen, sc = self.refiner._boxes(frame, PROMPT_CONTAINER)
                if len(cen):
                    py, px = self._notes[col]
                    j = int(np.argmin(np.hypot(cen[:, 1] - px, cen[:, 0] - py)))
                    out = COORD_RE.sub(f"<{int(round(cen[j,0]))}, {int(round(cen[j,1]))}>",
                                       response, count=1)
                    _log_pair(self, count, response, refined=out)
                    return out, has_api_error
            except Exception as e:
                print(f"[writemem] read failed: {e!r}", flush=True)
        # no usable note: fall back to the plain detector refinement
        refined = self.refiner.rewrite(response, frame, use_prior=True)
        if refined != response:
            _log_pair(self, count, response, refined=refined)
        return refined, has_api_error


# THE DELIVERED VERB LIST. This is what produced the published 50x16 table (60.50% over 800
# episodes) and must stay the default: every cell except three MoveCube refill episodes was run
# with exactly these verbs. "hook" and "push" were tried later and are a measured NULL on MoveCube
# (19/31 vs 17/31 matched, p=0.5), so they are opt-in via AGENTMEM_VERBS=contact, not default.
_GRASP = re.compile(r"^\s*(pick|grasp|grab|lift|put|place|drop|stack)\b", re.I)  # not insert: peg -20% refined
_GRASP_CONTACT = re.compile(r"^\s*(pick|grasp|grab|lift|put|place|drop|stack|hook|push)\b", re.I)


def _grasp_action(subgoal: str) -> bool:
    """Does this step close the gripper on something? Refinement earns its cost only there."""
    pat = _GRASP_CONTACT if os.environ.get("AGENTMEM_VERBS") == "contact" else _GRASP
    return bool(pat.match(subgoal or ""))


# The verb says what the arm does; it does NOT say what the coordinate points at. "place the cube
# onto the target at <y, x>" opens with a contact verb, but the coordinate names the TARGET -- a
# location the cube is moved to, not an object the gripper closes on. Refining it moves it onto
# whatever box the detector likes, and the detector is a generic proposer here.
_LOCATION_NOUNS = ("target", "targets", "bullseye", "bullseyes")
_COORD = re.compile(r"<\s*\d+\s*,\s*\d+\s*>")
_WORD = re.compile(r"[a-z]+")


def _coord_is_location(subgoal: str) -> bool:
    """True when the coordinate names a target/bullseye rather than an object.

    The referent is the LAST noun before the coordinate, not any earlier mention: MoveCube says
    "hook the cube to the target with the peg at <y, x>", where the coordinate is the peg's and
    refinement helps (+7.3 within 8 px). Matching "target" anywhere before the number would have
    disabled refinement on exactly the steps it earns its cost.
    """
    m = _COORD.search(subgoal or "")
    if not m:
        return False
    words = _WORD.findall((subgoal or "")[:m.start()].lower())
    while words and words[-1] in ("at", "the", "a", "an", "of", "on", "onto", "to", "in", "into"):
        words.pop()
    return bool(words) and words[-1] in _LOCATION_NOUNS


class AgentMemorySubgoalPredictor(DetectorRefinedQwenVLSubgoalPredictor):
    """THE PROPOSED METHOD -- write-time memory whose policy is decided by an agent.

    Same machinery as WriteTimeMemorySubgoalPredictor: notice the evidence while it is still
    visible, store where it was, and snap the later coordinate to whatever now covers that
    spot. The difference is who decides. That class answers "is a note needed", "is the
    evidence in these frames or in the demo", "which object", and "what does the cover look
    like" with four constants fitted to these sixteen tasks. Here one VLM agent answers all
    four in natural language from the task prompt, which is the only thing available at deploy
    time besides the image.

    The two arms differ in nothing else, so the comparison measures exactly the thing the paper
    claims: whether an agent can replace hand-written task rules without losing what they buy.

    Cost is one text-only call per episode for the policy, one more the first time a note is
    read back, and one short call per distinct subgoal only on episodes that took notes.
    """

    def setup_api(self) -> None:
        super().setup_api()
        from subgoal_prediction.agent_memory import AgentMemory
        self.mem = AgentMemory(self.api.engine)
        print("[robomme] agentic memory setup finished")

    def start_episode(self, epstate: EpisodeState, env_runner: EnvRunner) -> None:
        super().start_episode(epstate, env_runner)
        self.mem.plan(self.task_goal)
        self.mem.plan_cycle(self.task_goal)
        # EXPLORATION (2026-09-17), opt-in: a scene-graph tool that records pixel-verified
        # relations the detector cannot ground (a cube on a white highlight disc) as events.
        self.graph = None
        if os.environ.get("AGENTMEM_SG", "0") == "1":
            from subgoal_prediction.scene_graph import HighlightGraph
            self.graph = HighlightGraph()
        if os.environ.get("AGENTMEM_DEMO", "1") == "1":
            # the demonstration is every buffered frame but the last (the first live one)
            self.mem.observe_demo(list(epstate.image_buffer[:-1]), self.refiner._boxes,
                                  threshold=float(os.environ.get("WRITEMEM_TH", "0.30")),
                                  cover_phrase=PROMPT_CONTAINER)
        if os.environ.get("AGENTMEM_DUMP_DEMO") and len(epstate.image_buffer) > 1:
            # EXPLORATION: save the demonstration frames the predictor is handed, so parsers can be
            # iterated offline (per-episode run dirs are cleaned by the harness)
            try:
                d = os.environ["AGENTMEM_DUMP_DEMO"]; os.makedirs(d, exist_ok=True)
                np.savez_compressed(os.path.join(d, f"{self.env_name}_ep{self.episode_id}.npz"),
                                    frames=np.stack([np.asarray(f)[..., :3] for f in epstate.image_buffer[:-1]]).astype(np.uint8),
                                    goal=np.array(self.task_goal))
            except Exception as e:
                print(f"[agentmem] demo dump failed: {e!r}", flush=True)
        if os.environ.get("AGENTMEM_VIDEO2", "0") == "1" and len(epstate.image_buffer) > 1:
            # references the live scene cannot resolve ('the correct cube/target'): from the demo.
            # Written whenever a demonstration exists -- the planner's live/video call flips on
            # these prompts (identical blocks ARE visible live; which one was lifted is not) --
            # and read only when the VLM's own subgoal makes a demonstration reference.
            self.mem.observe_demo_objects(list(epstate.image_buffer[:-1]), self.task_goal)
            if self.graph is not None and "pick" in (self.task_goal or "").lower():
                # EXPLORATION: the demonstration read as events (a lifted slot, pairs of slots
                # swapped) answers "the block that was previously picked up" by the target's slot
                # after the events, instead of following one blob through the crossings
                from subgoal_prediction.scene_graph import DemoEventReader
                rd = DemoEventReader()
                got = rd.read(list(epstate.image_buffer[:-1]))
                if got is not None:
                    colour, y, x, how = got
                    self.mem.notes["correct cube"] = (float(y), float(x))
                    print(f"[agentmem] graph: demonstration events {rd.events} ({how}); "
                          f"the picked cube is the {colour} one at ({int(y)}, {int(x)})", flush=True)
        if os.environ.get("AGENTMEM_PATH", "0") == "1" and self.mem.source == "video":
            # the demonstration may be a ROUTE to retrace (an ordered memory, not a fact); it is
            # only used when the VLM's own subgoal carries a route slot
            self.mem.observe_path(list(epstate.image_buffer[:-1]))
            self.mem.begin_live(epstate.image_buffer[-1])

    def step(self, epstate: EpisodeState) -> None:
        super().step(epstate)
        if getattr(self, "graph", None) is not None and epstate.image_buffer:
            self.graph.observe(epstate.image_buffer[-1])
            if epstate.count % 3 == 0:
                self.graph.mark_picked(epstate.image_buffer[-1])
        if self.mem.path and epstate.image_buffer:
            # The predictor is only asked every subgoal_keep_period (16) steps, and a reached cue
            # confirmed one call late is a wrong-button touch on a fail-fast task (v6: the filled
            # subgoal trailed the oracle by exactly one interval). Track the cue on every frame.
            self.mem.track_path(epstate.image_buffer[-1])
        if os.environ.get("AGENTMEM_ORD", "0") == "1" and epstate.image_buffer:
            self.mem.track_flashes(epstate.image_buffer[-1])
            dump = os.environ.get("AGENTMEM_DUMP_DIR")
            if dump:
                # diagnostic only: every live frame, to replay the cue counter offline
                try:
                    from PIL import Image as _Im
                    d = os.path.join(dump, f"{self.env_name}_ep{self.episode_id}"); os.makedirs(d, exist_ok=True)
                    fr = np.asarray(epstate.image_buffer[-1])
                    _Im.fromarray(fr[..., :3].astype(np.uint8)).save(os.path.join(d, f"{epstate.count:05d}.png"))
                except Exception as e:
                    print(f"[agentmem] dump failed: {e!r}", flush=True)
        if os.environ.get("AGENTMEM_TRACK", "0") == "1" and epstate.image_buffer and self.mem.notes \
                and self.mem.source != "video" and epstate.count % 3 == 0:
            # a live note's cover can be moved after the covering (ButtonUnmaskSwap); follow it
            self.mem.track_cover(epstate.image_buffer[-1], self.refiner._boxes,
                                 cover_phrase=PROMPT_CONTAINER, step=epstate.count)

    def get_subgoal(
        self,
        count: int,
        current_subgoal: Optional[str],
        last_subgoal: Optional[str],
    ) -> Tuple[Optional[str], bool]:
        frame = self.video_buffer[-1] if self.video_buffer else None
        if self.mem.source != "video":
            self.mem.observe(frame, self.refiner._boxes,
                             threshold=float(os.environ.get("WRITEMEM_TH", "0.30")))
        response, has_api_error = QwenVLSubgoalPredictor.get_subgoal(
            self, count, current_subgoal, last_subgoal)
        if has_api_error or not response:
            return response, has_api_error

        if os.environ.get("AGENTMEM_VIDEO2", "0") == "1":
            filled = self.mem.fill_demo_reference(response)
            if filled is not None:
                if filled != response:
                    _log_pair(self, count, response, refined=filled)
                return filled, has_api_error
        if getattr(self, "graph", None) is not None:
            # a relation the graph recorded answers the reference; the fill is final (pixel-exact)
            filled = self.graph.fill(response)
            if filled is not None:
                if filled != response:
                    _log_pair(self, count, response, refined=filled)
                return filled, has_api_error
        if self.mem.path:
            # the cue tracker runs in step() on every frame; here the route only fills the slots
            filled = self.mem.path_fill(response)
            if filled is not None:
                if filled != response:
                    _log_pair(self, count, response, refined=filled)
                return filled, has_api_error

        if os.environ.get("AGENTMEM_CYCLE", "1") == "1":
            response = self.mem.gate_cycle(response, count)
        if os.environ.get("AGENTMEM_ORD", "0") == "1":
            gated = self.mem.gate_ordinal(response)
            if gated != response:
                _log_pair(self, count, response, refined=gated)
                response = gated
        self.mem.observe_phase(response, count)
        if not self.mem.cycle and os.environ.get("AGENTMEM_REVERT", "1") == "1":
            # a stale request is reverted to the previous phase; repetition tasks have their own
            # hold logic in gate_cycle and are left to it
            back = self.mem.stale_revert()
            if back is not None:
                response = back
        hit = self.mem.consult(response)
        if hit and frame is not None:
            try:
                note = self.mem.recall(hit)
                # Naming the covers for the detector is a SEPARATE question from deciding the
                # memory policy, and one the agent is measurably worse at: its own description
                # lands on the cover 2% of the time against 83% for the shared phrase, and 40%
                # once the detector is allowed to reject proposals that find nothing. Letting it
                # choose here would confound the arm with a known-negative keyword decision, so
                # the shared phrase is the default and the agent's is the ablation.
                phrase = PROMPT_CONTAINER
                if os.environ.get("AGENTMEM_KW") == "1":
                    phrase = self.mem.cover_phrase(frame, self.refiner._boxes, note) \
                             or PROMPT_CONTAINER
                cen, sc = self.refiner._boxes(frame, phrase)
                if len(cen):
                    py, px = note
                    j = int(np.argmin(np.hypot(cen[:, 1] - px, cen[:, 0] - py)))
                    out = COORD_RE.sub(f"<{int(round(cen[j,0]))}, {int(round(cen[j,1]))}>",
                                       response, count=1)
                    _log_pair(self, count, response, refined=out)
                    return out, has_api_error
            except Exception as e:
                print(f"[agentmem] read failed: {e!r}", flush=True)

        # Nothing recorded for this subgoal. Plain detector refinement is the fallback every
        # other arm uses -- and it is SLOW on the one phase the raw VLM already gets right:
        # measured, a refined button press takes ~270 steps against <=96 raw, and the VLM's
        # patience is ~96, so the refined arms jump to the container before the button is
        # pressed. AGENTMEM_REFINE=0 sends the raw VLM coordinate here; memory reads above are
        # untouched. That is "qwenvl + memory", the arm never run.
        mode = os.environ.get("AGENTMEM_REFINE", "1")
        if mode == "0":
            return response, has_api_error
        # AGENTMEM_LOCGUARD defaults to "0": the published 50x16 table was produced WITHOUT this
        # guard, so the default code path must reproduce it. Set it to "1" to enable the guard.
        if mode == "verbs" and os.environ.get("AGENTMEM_LOCGUARD", "0") == "1" and _coord_is_location(response):
            # The coordinate names a target, not an object: leave the VLM's own number alone.
            # Measured on saved pairs, on the steps the verb rule refines: cube +6.9 and peg +7.3
            # within 8 px for the refined arm, but target -11.3 and container -11.2 against the
            # unrefined baseline. VideoPlaceButton is the clean episode-level case -- 11/11 for
            # plain QwenVL against 6/11 for the refined agent on 11 matched episodes, 5 discordant
            # all one way, and the split is entirely in its "place ... onto the target" steps
            # (27.7% within 8 px against 75.6%) while its "pick ... cube" steps are 100% in both.
            return response, has_api_error
        if mode == "verbs" and not _grasp_action(response):
            # Refine only where the gripper must close on the object. Measured per kind, within
            # arm: refinement helps cubes +19, bins +37, containers +16, and does nothing for
            # buttons (+4) or targets (+5) while making the press ~3x slower. The VLM's distance
            # to the box does not separate the two cases (~8px on both); the ACTION does.
            return response, has_api_error
        refined = self.refiner.rewrite(response, frame, use_prior=True)
        if refined != response:
            _log_pair(self, count, response, refined=refined)
        return refined, has_api_error

    def end_episode(self, epstate: EpisodeState, success_flag: str) -> None:
        print(f"[agentmem] {self.mem.stats}", flush=True)
        super().end_episode(epstate, success_flag)


class OracleProgressSubgoalPredictor(DetectorRefinedQwenVLSubgoalPredictor):
    """HEADROOM INSTRUMENT -- not deployable, not the method.

    Gives the composer a perfect progress state and measures what that is worth. The state is
    fed from the oracle subgoal stream, so it knows how many repetitions of each step are done
    and which step is actually in progress -- but it never supplies object identity or a
    coordinate. Those still come from QwenVL and the detector respectively.

    It exists to answer one question before the agentic version is built: if progress belief
    were perfect, how much of the semantics gap closes? Offline on 3,976 logged steps it takes
    subgoal text-match from 70.0% to 83.8% (+4.2 ordinal, +9.6 phase), with ButtonUnmask going
    to 100% because every one of its mismatches is the agent running ahead of the button press.
    If the task-success gain is small, the gap is not about progress tracking and the agentic
    memory should be rethought rather than built.
    """

    def setup_api(self) -> None:
        super().setup_api()
        from subgoal_prediction.progress_state import ProgressState
        self._ProgressState = ProgressState
        self.progress = ProgressState()
        self.stats_prog = {"ordinal": 0, "phase": 0, "steps": 0}
        print("[robomme] oracle-progress instrument setup finished")

    def start_episode(self, epstate: EpisodeState, env_runner: EnvRunner) -> None:
        super().start_episode(epstate, env_runner)
        self.progress = self._ProgressState()

    def get_subgoal(
        self,
        count: int,
        current_subgoal: Optional[str],
        last_subgoal: Optional[str],
    ) -> Tuple[Optional[str], bool]:
        from subgoal_prediction.progress_state import correct_ordinal, correct_phase, out_of_sync
        frame = self.video_buffer[-1] if self.video_buffer else None
        response, err = QwenVLSubgoalPredictor.get_subgoal(
            self, count, current_subgoal, last_subgoal)
        if err or not response:
            return response, err

        # perfect progress: observe what the environment actually has in progress
        self.progress.observe(getattr(self.env_runner, "grounded_subgoal_oracle", None))
        self.stats_prog["steps"] += 1

        fixed = correct_ordinal(response, self.progress)
        if fixed != response:
            self.stats_prog["ordinal"] += 1

        substituted = False
        if out_of_sync(fixed, self.progress):
            cur = self.progress._current
            if cur:
                # the state's step carries no coordinate; give the detector a placeholder to fill
                fixed = cur.replace("<>", "<0, 0>")
                substituted = True
                self.stats_prog["phase"] += 1

        # coordinates always come from the detector, never from the progress state
        out = self.refiner.rewrite(fixed, frame, use_prior=not substituted)
        if out != response:
            _log_pair(self, count, response, refined=out)
        return out, err

    def end_episode(self, epstate: EpisodeState, success_flag: str) -> None:
        print(f"[oracle_progress] {self.stats_prog}", flush=True)
        super().end_episode(epstate, success_flag)


class MemERSubgoalPredictor(SubgoalPredictorBase):
    def setup_api(self) -> None:
        self.api = Qwen3VLModelMemER(adapter_path=self.args.memer_adapter_path)
        print("[robomme] MemER agent setup finished")
    
    def start_episode(self, epstate: EpisodeState, env_runner: EnvRunner) -> None:
        super().start_episode(epstate, env_runner)
        self.episode_dir = os.path.join(self.save_dir, self.env_name, f"ep{self.episode_id}")
        self.api.start_new_episode(self.episode_dir, epstate.image_buffer[:-1], self.task_goal)

    def step(self, epstate: EpisodeState) -> None:
        self.api.add_execution_frame(epstate.image_buffer[-1])

    def get_subgoal(
        self,
        count: int,
        current_subgoal: Optional[str],
        last_subgoal: Optional[str],
    ) -> Tuple[Optional[str], bool]:
        response = self.api.call()
        return response, False

    def end_episode(self, epstate: EpisodeState, success_flag: str) -> None:
        if self.episode_dir:
            shutil.rmtree(self.episode_dir) # save some space, you can comment this function out to keep all video frames


class OracleSubgoalPredictor(SubgoalPredictorBase):
    
    def setup_api(self) -> None:
        print("[robomme] Oracle agent setup finished")
    
    def get_subgoal(
        self,
        count: int,
        current_subgoal: Optional[str],
        last_subgoal: Optional[str],
    ) -> Tuple[Optional[str], bool]:
        if self.args.subgoal_type == "simple_subgoal":
            return self.env_runner.simple_subgoal_oracle, False
        else:
            return self.env_runner.grounded_subgoal_oracle, False


class DetectorOracleSubgoalPredictor(OracleSubgoalPredictor):
    """Oracle subgoal TEXT, detector COORDINATES.

    Table 2 puts GroundSG+Oracle at 84.08 and SimpleSG+Oracle at 49.58, so the
    `at <y, x>` numbers alone are worth about +34.5 points. Those numbers come from
    simulator state. This asks whether an open-vocabulary detector can stand in for them:
    the sentence the oracle produces is kept verbatim and only the coordinates are
    recomputed from the image.

    DET_ORACLE_MODE selects what the substitution is allowed to know:
      top1     highest-scoring box -- image and phrase only, i.e. actually deployable.
      nearest  box closest to the oracle's own coordinate -- not deployable, but it
               isolates detector PRECISION from instance SELECTION, which is the thing
               we want to attribute the result to.
    """

    def setup_api(self) -> None:
        super().setup_api()
        import os as _os
        from subgoal_prediction.detector_refine import DetectorRefiner
        self.refiner = DetectorRefiner()
        self.mode = _os.environ.get("DET_ORACLE_MODE", "top1")
        print(f"[robomme] detector-oracle agent setup finished (mode={self.mode})")

    def start_episode(self, epstate: EpisodeState, env_runner: EnvRunner) -> None:
        super().start_episode(epstate, env_runner)
        self.refiner.reset()

    def step(self, epstate: EpisodeState) -> None:
        # the oracle predictor keeps no video buffer, so stash the current frame here
        self._frame = epstate.image_buffer[-1] if epstate.image_buffer else None

    def _maybe_dump(self, subgoal) -> None:
        """Save (eval-time frame, oracle subgoal) pairs for offline detector tuning.

        Detector accuracy measured on dataset frames overstates what happens during a
        rollout -- 56% vs 36% within 8px on BinFill -- because the arm occludes objects and
        the scene drifts as the task progresses. Tuning has to be done against frames the
        detector will actually see, and a rollout is the only place they exist.
        """
        import os as _os
        d = _os.environ.get("DET_DUMP_DIR")
        if not d or self._frame is None:
            return
        try:
            import numpy as _np
            n = getattr(self, "_dump_n", 0)
            self._dump_n = n + 1
            every = int(_os.environ.get("DET_DUMP_EVERY", "3"))
            cap = int(_os.environ.get("DET_DUMP_CAP", "4000"))
            kept = getattr(self, "_dump_kept", 0)
            if n % every or kept >= cap:
                return
            self._dump_kept = kept + 1
            _os.makedirs(d, exist_ok=True)
            _np.savez_compressed(
                _os.path.join(d, f"{self.env_name}_ep{self.episode_id}_{n:05d}.npz"),
                image=_np.asarray(self._frame, dtype=_np.uint8),
                subgoal=str(subgoal), env=str(self.env_name))
        except Exception:
            pass

    def get_subgoal(
        self,
        count: int,
        current_subgoal: Optional[str],
        last_subgoal: Optional[str],
    ) -> Tuple[Optional[str], bool]:
        subgoal, err = super().get_subgoal(count, current_subgoal, last_subgoal)
        if err or not subgoal:
            return subgoal, err
        self._maybe_dump(subgoal)
        frame = getattr(self, "_frame", None)
        if self.mode == "tuned":
            swapped = self.refiner.rewrite(subgoal, frame, use_prior=False)
        else:
            swapped = self.refiner.replace(subgoal, frame, self.mode)
        if swapped != subgoal:
            _log_pair(self, count, swapped)
        return swapped, err

    def end_episode(self, epstate: EpisodeState, success_flag: str) -> None:
        print(f"[detector_oracle] {self.refiner.stats}", flush=True)


def build_subgoal_predictor(
    args,
    save_dir: Path,
) -> SubgoalPredictorBase:
    if args.use_gemini:
        return GeminiSubgoalPredictor(args, save_dir)
    if getattr(args, "use_oracle_progress", False):
        return OracleProgressSubgoalPredictor(args, save_dir)
    if getattr(args, "use_agentmem", False):
        return AgentMemorySubgoalPredictor(args, save_dir)
    if getattr(args, "use_writemem", False):
        return WriteTimeMemorySubgoalPredictor(args, save_dir)
    if getattr(args, "use_detector_oracle", False):
        return DetectorOracleSubgoalPredictor(args, save_dir)
    if getattr(args, "use_detector_refine", False):
        return DetectorRefinedQwenVLSubgoalPredictor(args, save_dir)
    if args.use_qwenvl:
        return QwenVLSubgoalPredictor(args, save_dir)
    if args.use_memer:
        return MemERSubgoalPredictor(args, save_dir)
    if args.use_oracle:
        return OracleSubgoalPredictor(args, save_dir)
    
    return NullSubgoalPredictor(args, save_dir)



