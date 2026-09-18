"""Agentic write-time memory.

The benchmark hides information on purpose: a cube is covered by an identical white container,
a demonstration ends before execution begins, an object is moved out of frame. Asked at READ
time -- "which container hides the blue cube?" -- the question is unanswerable from the current
image, and the measured accuracy is at chance. Asked at WRITE time, while the cube is still
uncovered, it is a plain colour query the detector answers well.

The hard part is not storing the note. It is knowing, from the task prompt alone and before
anything has happened, THAT a note will be needed, WHICH object it is about, and whether the
robot will ever see that object with its own camera. A previous version answered all three with
per-task regular expressions over the goal string. Those work on these sixteen tasks and
generalise to nothing.

Here one VLM agent answers the same three questions in natural language, from the prompt only:

    NEED    -- will something be needed later that is not visible when it is needed?
    SOURCE  -- will the robot see it in its own camera, or was it shown only in a prior video?
    WATCH   -- which objects, as phrases a detector can act on?

and then, per subgoal, a fourth:

    CONSULT -- does what I am about to do refer to something I recorded?

The agent sees the task prompt and the current image, nothing else -- the same information the
policy has at deploy time. There are no task names, no per-task branches, and no pattern
matching on the goal string anywhere in this file.
"""

from typing import Optional, List, Dict, Tuple
import collections
import contextlib
import tempfile
import os
import re
import textwrap

import numpy as np

from swift.llm import InferRequest, RequestConfig

from subgoal_prediction.detector_refine import phrase_of


PLAN_PROMPT = """A robot must carry out this task, seeing only its camera image at each moment:

"{goal}"

Tasks differ. In some, everything the robot needs stays in the picture the whole time -- it
repeats an action, or reaches a target it can see. In others something is deliberately covered
up, and afterwards several things look alike, so the robot can only tell them apart if it looked
at them before the covering happened.

Decide which kind THIS task is.

1. When the robot must choose what to act on, will the thing that identifies the right choice
   have stopped being visible by then?
   Write one line: NEED: yes    or    NEED: no

2. If yes, when could the robot have seen that identifying thing? Answer live if the robot's own
   camera shows it at any point once the task has begun, including in the opening moments before
   anything is covered. Answer video only if it was shown in a demonstration that played before
   the robot started acting, and never appears again.
   Write one line: SOURCE: live    or    SOURCE: video

3. If yes and live, name the things that must be recorded before they stop being visible. Give
   one short noun phrase for each, as the task names it, separated by semicolons. Name only what
   gets hidden -- not what hides it, not buttons, not targets, and not actions.
   Write one line beginning: WATCH:

Answer with those three lines and nothing else."""


COVER_PROMPT = """This is what the robot sees now. Earlier it recorded where these objects were, and they are no longer in view:
{keys}

Something in this picture is covering them, and the robot must pick the right one. Look at the picture and name what the covers look like, so an ordinary object detector can find them. Give three different phrases, simplest first, each one or two plain words that an everyday object detector would know. Separate them with semicolons.

Answer with the three phrases only."""


# Kept as the ablation rung "ask the agent every step". Measured 72% recall at 8%
# precision on 1478 steps: asked per step the model says yes almost always, including
# on "pick up the first blue cube" when the blue cube IS the note. Not used.
CYCLE_PROMPT = """A robot is given this task:

"{goal}"

Some tasks repeat a short motion a stated number of times and then finish with a different
action. Only the actions INSIDE the repeated motion count as the cycle -- not anything done once
to set it up beforehand (such as first picking something up), and not what comes after. If the
task does not say a motion is repeated a number of times, the cycle is none.

CYCLE: the actions inside the repeated motion, in order, separated by semicolons -- or none
COUNT: how many times that motion repeats, as a number -- or none
FINAL: the action that comes after the repetitions -- or none

Answer with those three lines and nothing else."""


CONSULT_PROMPT = """A robot is about to do this: "{subgoal}"

Earlier in this episode it recorded where these objects were, before they were hidden:
{keys}

Is the robot about to act on a DIFFERENT object that can only be told apart by which recorded
object it relates to -- for example a cover, chosen because of what is underneath it?

Answer none if the robot is simply acting on a recorded object itself, or if the choice needs no
memory at all. Otherwise answer with just the recorded object's phrase."""


# EXPLORATION (2026-09-18), opt-in AGENTMEM_SG=agent: after the plan, a separate question asks, for
# each thing the agent named, how the task description picks it out; each answer selects one
# scene-graph reader (scene_graph.py). Offline on 16 prompts x 5 wordings (campaign/sg/
# route_study_v4.py) -- see the review log for the counts. A separate call, not an appended
# line: appended, the question flipped the plan's own SOURCE decision on two tasks.
ROUTE_PROMPT = """A robot must carry out this task, seeing only its camera image at each moment:

"{goal}"

It will record where these things are before they can no longer be told apart:
{items}

For each one, say how the task description picks it out from things that look the same, choosing
one word: appearance -- by its own colour or shape; mark -- by a temporary mark on or around it,
such as a highlighted patch or a flash; handled -- as the one that was handled (picked up, moved)
in the demonstration; sequence -- by when it was used in the demonstration: first or second, or
just before or just after something else happened, such as a button press. A thing described by
an event or by its order is sequence even if it also has a colour.
   Write one line per thing: HOW: <thing> = appearance    or    = mark    or    = handled    or    = sequence

Answer with those lines and nothing else."""

HOW_LINE = re.compile(r"how\s*:\s*([^=\n]*?)\s*=\s*(appearance|mark|handled|sequence)", re.I)


_COORD_IN_TEXT = re.compile(r"\s*at\s*<\s*\d+\s*,\s*\d+\s*>")


def phase_key(subgoal: str) -> str:
    """The subgoal with its coordinate removed. Timers that key on the raw text never fire,
    because the VLM's numbers jitter from call to call and reset them every time."""
    return " ".join(_COORD_IN_TEXT.sub("", (subgoal or "").lower()).split())


NEED = re.compile(r"need\s*:\s*(yes|no)", re.I)
SOURCE = re.compile(r"source\s*:\s*(live|video)", re.I)
WATCH = re.compile(r"watch\s*:\s*(.+)", re.I)


class AgentMemory:
    """Episode-scoped memory whose policy is decided by the agent, not by rules.

    Holds the notes for one episode. `plan` is one text-only call at episode start; `consult`
    is one short call per subgoal, and only when there is something to consult. Everything is
    cached, so an episode costs one call plus at most one per distinct subgoal.
    """

    def __init__(self, engine, log_prefix: str = "[agentmem]"):
        self.engine = engine
        self.tag = log_prefix
        # keep every byte on /workspace: TMPDIR is not exported by the campaign scripts, so
        # a bare mkdtemp would land on the shared root disk
        root = os.environ.get("TMPDIR") or "/workspace/yunbei/robomem/tmp"
        os.makedirs(root, exist_ok=True)
        self._scratch = tempfile.mkdtemp(prefix="agentmem_", dir=root)
        self.reset()

    def reset(self) -> None:
        self.notes: Dict[str, Tuple[float, float]] = {}
        self.watch: List[str] = []
        self.need = False
        self.source = ""
        self.kinds: List[str] = []          # AGENTMEM_SG=agent: reader kinds the agent chose
        self._consulted: Dict[str, Optional[str]] = {}
        self._cover: Optional[str] = None
        self.cycle: List[str] = []          # repeated actions, from the goal
        self.cycle_n: Optional[int] = None  # how many repetitions the goal asks for
        self.final: str = ""                # what comes after them
        self._cycle_hits: List[int] = []    # completed visits per cycle element
        self._cyc_key: Optional[str] = None
        self._cyc_start: Optional[int] = None
        self._cyc_counted = False
        self._last_cycle_subgoal: Optional[str] = None
        self._wording: Dict[int, str] = {}
        self._phase_key: Optional[str] = None
        self._phase_start: Optional[int] = None
        self._phase_subgoal: Optional[str] = None
        self._prev_subgoal: Optional[str] = None
        self._prev_key: Optional[str] = None
        self._stale_key: Optional[str] = None
        self._revert_start: Optional[int] = None
        self._stale_logged = False
        self._step = 0
        self.calls = 0
        self.path: List[dict] = []          # ordered segments of the demonstrated route
        self._live: Optional[dict] = None   # cue tracker over the live frames
        self._flash: Optional[dict] = None  # verified reach-flash counter over the live frames
        self._ord_order: List[str] = []     # ordinal-slot elements in order of first appearance
        self._ord_wording: Dict[str, str] = {}
        self._ord_hold = False              # a premature transition was seen before any flash
        self._trk: Dict[str, dict] = {}     # live cover tracking per note: gone step, position

    # ---------------------------------------------------------------- agent

    def _ask(self, text: str, max_tokens: int = 80, image=None) -> str:
        """One text-only turn from the base model.

        The composer's LoRA is trained to emit grounded subgoals and would drag these answers
        into that format, so it is switched off for the agent's own reasoning where the runtime
        allows it. The agent and the composer are the same weights otherwise.
        """
        self.calls += 1
        if image is None:
            req = InferRequest(messages=[{"role": "user", "content": text}])
        else:
            import imageio
            path = os.path.join(self._scratch, "ask.png")
            imageio.imwrite(path, image)
            req = InferRequest(messages=[{"role": "user", "content": f"<image>{text}"}],
                               images=[path])
        cfg = RequestConfig(max_tokens=max_tokens, temperature=0)
        with self._base_model():
            out = self.engine.infer([req], request_config=cfg)
        return (out[0].choices[0].message.content or "").strip()

    @contextlib.contextmanager
    def _base_model(self):
        model = getattr(self.engine, "model", None)
        fn = getattr(model, "disable_adapter", None)
        if fn is None:
            yield
            return
        try:
            with fn():
                yield
        except Exception:
            yield

    # ---------------------------------------------------------------- policy

    # A phase held this long without advancing is treated as premature. 150 was the best of
    # {150, 300, 600} offline; the effect is broad, not a knife edge.
    STALE_AFTER = int(os.environ.get("AGENTMEM_STALE", "150"))

    WIDTHS = (0, 78, 70)      # 0 = one line per instruction, the canonical unwrapped form

    @staticmethod
    def _rewrap(template: str, width: int) -> str:
        """Re-flow each instruction to a given width, leaving the words untouched."""
        out = []
        for para in template.split("\n\n"):
            parts = para.split("\n   Write one line")
            body = " ".join(parts[0].split())
            if width:
                body = "\n".join(textwrap.wrap(body, width, subsequent_indent="   "))
            out.append(body + ("\n   Write one line" + parts[1] if len(parts) > 1 else ""))
        return "\n\n".join(out)

    def _ask_plan(self, goal: str, width: int):
        """One reading of the policy question. Returns (take notes?, phrases)."""
        ans = self._ask(self._rewrap(PLAN_PROMPT, width).format(goal=goal))
        m = NEED.search(ans); need = bool(m) and m.group(1).lower() == "yes"
        m = SOURCE.search(ans); src = m.group(1).lower() if m else ""
        m = WATCH.search(ans)
        w = [x.strip().strip(".").lower()
             for x in re.split(r"[;,]", m.group(1)) if x.strip()][:4] if m else []
        return (need and bool(w)), w, need, src

    def plan(self, task_goal: str) -> None:
        self._plan(task_goal)
        if os.environ.get("AGENTMEM_SG", "0") == "agent" and task_goal:
            try:
                self.kinds = self._route(task_goal.strip())
            except Exception as e:
                print(f"{self.tag} route failed: {e!r}", flush=True)

    def _route(self, goal: str) -> List[str]:
        """EXPLORATION (2026-09-18), opt-in AGENTMEM_SG=agent. After the plan, the agent says for each
        thing it named how the task description picks it out -- appearance / mark / handled /
        sequence -- and the predictor turns on one scene-graph reader per kind. A separate call,
        so the plan's own decisions are untouched. Three widths; a kind is chosen when at least
        two readings name it. Three more text-only calls per episode."""
        items = "\n".join(f"- {w}" for w in self.watch) if self.watch else \
            "- whatever the robot must pick out from things that look the same"
        kinds: List[str] = []
        for width in self.WIDTHS:
            text = self._rewrap(ROUTE_PROMPT, width).replace("{goal}", goal).replace("{items}", items)
            ans = self._ask(text, max_tokens=120)
            kinds.extend(sorted({b.lower() for _, b in HOW_LINE.findall(ans)}))
        cnt = collections.Counter(kinds)
        chosen = sorted(k for k, c in cnt.items() if c >= 2 and k != "appearance")
        print(f"{self.tag} readers chosen by the agent: {chosen or 'none'}  (kinds per reading {kinds})", flush=True)
        return chosen

    def _plan(self, task_goal: str) -> None:
        """Decide, from the prompt alone, whether to take notes and about what.

        Asked once, the answer turns out to depend on where the question's lines happen to break:
        the same words under four wrappings scored 62% to 100% F1 end to end, because on one task
        family the model flips between "the evidence is in my own camera" and "the evidence was in
        the demonstration". Recall was 100% under every wrapping -- the disagreement is entirely
        about taking notes that are not needed -- so the question is asked at three widths and the
        answer is the majority, with a tie resolved as no notes. Three text-only calls per episode.
        """
        self.reset()
        if not task_goal:
            return
        goal = task_goal.strip()

        # Ablation switches (all opt-in, default = the reported behaviour):
        #   AGENTMEM_VOTE=1        one canonical wrapping instead of the three-width majority
        #   AGENTMEM_POLICY=rules  keyword rule instead of the agent: watch the colour-cube phrases
        #                          the goal names (all three if none), source=video iff the goal
        #                          mentions a video/demonstration
        #   AGENTMEM_POLICY=all    always take notes on all three colour cubes (write gate off)
        mode = os.environ.get("AGENTMEM_POLICY", "agent")
        if mode in ("rules", "all"):
            g = goal.lower()
            found = [f"{c} cube" for c in ("red", "green", "blue") if re.search(rf"\b{c} cubes?\b", g)]
            self.watch = found if (mode == "rules" and found) else ["red cube", "green cube", "blue cube"]
            self.need = True
            self.source = "video" if re.search(r"\bvideo|demonstrat", g) else "live"
            print(f"{self.tag} policy={mode}: watching {self.watch} in the "
                  f"{'demonstration' if self.source == 'video' else 'live frames'}", flush=True)
            return
        widths = (0,) if os.environ.get("AGENTMEM_VOTE", "3") == "1" else self.WIDTHS
        votes, phrases, seen, srcs = 0, [], 0, []
        for width in widths:
            try:
                take, w, need, src = self._ask_plan(goal, width)
            except Exception as e:
                print(f"{self.tag} plan failed: {e!r}", flush=True)
                continue
            seen += 1
            self.need = self.need or need
            if take:
                votes += 1
                phrases.append(w)
                if src: srcs.append(src)

        if not seen or votes * 2 <= seen:
            print(f"{self.tag} no memory needed ({votes}/{seen} votes for taking notes)", flush=True)
            return

        cnt = collections.Counter(p for w in phrases for p in set(w))
        self.watch = [p for p, c in cnt.items() if c * 2 > len(phrases)] or \
                     sorted({p for w in phrases for p in w})
        # WHERE the evidence is decides which frames are watched. "video" used to mean "take no
        # notes"; the demonstration is handed to the predictor at episode start, and the same
        # earliest-sighting loop over its frames put the note within 1.4px of the oracle's cover
        # on 10/10 VideoUnmask episodes offline.
        self.source = "video" if srcs.count("video") * 2 > len(srcs) else "live"
        print(f"{self.tag} watching {self.watch} in the "
              f"{'demonstration' if self.source == 'video' else 'live frames'}  "
              f"({votes}/{seen} votes)", flush=True)

    # Calibrated reference colours (detector_refine.py). Used to VERIFY a detection: Grounding
    # DINO returns a confident "green cube" box whether or not a green cube is there, so a score
    # cannot say when an object has been covered. The pixels can.
    COLOUR_REF = {"red": (192.0, 6.0, 6.0), "green": (6.0, 181.0, 6.0), "blue": (5.0, 5.0, 196.0)}

    @classmethod
    def _colour_ok(cls, im, cy, cx, colour, r=4, tol=90.0) -> bool:
        ref = cls.COLOUR_REF.get(colour)
        if ref is None or im is None:
            return True                       # no colour handle: cannot verify, do not block
        h, w = im.shape[:2]
        y0, y1 = int(max(cy - r, 0)), int(min(cy + r + 1, h))
        x0, x1 = int(max(cx - r, 0)), int(min(cx + r + 1, w))
        patch = np.asarray(im)[y0:y1, x0:x1].reshape(-1, 3).astype(float)
        if not len(patch):
            return False
        return (np.linalg.norm(patch - np.array(ref), axis=1) < tol).mean() >= 0.25

    @staticmethod
    def _colour_of(phrase: str) -> Optional[str]:
        for c in ("red", "green", "blue"):
            if c in phrase.lower():
                return c
        return None

    def observe(self, frame, boxes_fn, threshold: float = 0.30) -> None:
        """Record the earliest confident sighting of each watched phrase, from a LIVE frame.

        Earliest, not best: the whole point is that these objects stop being identifiable, so a
        later sighting is a sighting of something else. Each phrase is recorded once.
        """
        if frame is None or not self.watch:
            return
        for phrase in self.watch:
            if phrase in self.notes:
                continue
            if self._colour_of(phrase) is None and os.environ.get("AGENTMEM_COLOURRULE", "1") == "1":
                # A note the pixels cannot verify is a note the detector can fake: on PickHighlight
                # ('cubes highlighted with white areas') the earliest confident 'highlighted cube'
                # box is any cube, taken before the highlight exists, and the read then lands
                # 47 px off where the plain baseline's refinement lands within 2 (3/13 vs 5/13).
                if not getattr(self, "_skipped_log", set()) or phrase not in self._skipped_log:
                    self._skipped_log = getattr(self, "_skipped_log", set()) | {phrase}
                    print(f"{self.tag} no colour to verify {phrase!r}: not noted", flush=True)
                continue
            try:
                cen, sc = boxes_fn(frame, phrase)
            except Exception:
                continue
            if len(cen) and float(sc.max()) >= threshold:
                j = int(np.argmax(sc))
                self.notes[phrase] = (float(cen[j, 0]), float(cen[j, 1]))
                print(f"{self.tag} noted {phrase} at {self.notes[phrase]}", flush=True)

    def track_cover(self, frame, boxes_fn, cover_phrase: str = "white cube",
                    max_jump: float = 18.0, step: int = 0) -> None:
        """Follow each live note's cover after the covering, frame by frame -- the live form of
        what observe_demo does over the demonstration. Measured need: on ButtonUnmaskSwap the
        containers are swapped after covering, and the untracked note points 30-170 px from the
        oracle's container on 18 of 29 failures (noref, n=36). Once the noted object's colour is
        no longer at its note, the container nearest the note is the cover, and from then on it
        is followed by nearest-neighbour within max_jump. Where nothing moves this is a no-op."""
        if frame is None or not self.notes:
            return
        for p, note in self.notes.items():
            st = self._trk.setdefault(p, {"gone": None, "pos": None})
            if st["gone"] is None:
                if self._colour_ok(frame, note[0], note[1], self._colour_of(p)):
                    continue
                try:
                    cen, _ = boxes_fn(frame, cover_phrase)
                except Exception:
                    continue
                if not len(cen):
                    continue
                d = np.hypot(cen[:, 0] - note[0], cen[:, 1] - note[1]); j = int(np.argmin(d))
                if d[j] > 30:
                    continue                  # nothing is on the note: occluded, not covered
                st["gone"] = step; st["pos"] = (float(cen[j, 0]), float(cen[j, 1]))
                print(f"{self.tag} {p} covered at step {step}; following its cover", flush=True)
                continue
            try:
                cen, _ = boxes_fn(frame, cover_phrase)
            except Exception:
                continue
            if not len(cen):
                continue
            d = np.hypot(cen[:, 0] - st["pos"][0], cen[:, 1] - st["pos"][1]); j = int(np.argmin(d))
            if d[j] <= max_jump:
                st["pos"] = (float(cen[j, 0]), float(cen[j, 1]))

    def observe_demo(self, frames, boxes_fn, threshold: float = 0.30, stride: int = 3,
                     cover_phrase: str = "white cube", max_jump: float = 18.0) -> None:
        """Take the notes from the demonstration, and FOLLOW whatever covers them.

        Two measured facts shape this. On VideoUnmask the earliest sighting alone lands the note
        within 1.4px of the oracle's cover on 10/10 episodes. On VideoUnmaskSwap the same note is
        41px off on 8/10, because the containers are swapped after covering -- the cube's position
        no longer marks its cover. So once the watched object is seen to disappear (a detected box
        whose pixels are no longer its colour), the container nearest that spot is tracked by
        nearest-neighbour to the end of the demonstration, and the tracked position is the note.
        Offline that is 100% within 8px on both tasks (median 1.1px). Where nothing moves the
        tracker is a no-op.
        """
        if self.source != "video" or not self.watch or frames is None or not len(frames):
            return
        n = len(frames)
        idx = list(range(0, n, max(1, stride)))
        if idx[-1] != n - 1:
            idx.append(n - 1)
        colour_rule = os.environ.get("AGENTMEM_COLOURRULE", "1") == "1"      # ablation: "0" notes colourless phrases
        state = {p: dict(note=None, gone=None) for p in self.watch
                 if self._colour_of(p) is not None or not colour_rule}
        for p in self.watch:
            if self._colour_of(p) is None and colour_rule:
                print(f"{self.tag} demo: no colour to verify {p!r}: not noted", flush=True)
        if not state:
            return
        for i in idx:
            im = frames[i]
            for p, st in state.items():
                if st["gone"] is not None:
                    continue
                colour = self._colour_of(p)
                try:
                    cen, sc = boxes_fn(im, p)
                except Exception:
                    continue
                ok = [j for j in range(len(cen)) if float(sc[j]) >= threshold
                      and self._colour_ok(im, cen[j, 0], cen[j, 1], colour)]
                if ok:
                    if st["note"] is None:
                        j = max(ok, key=lambda j: float(sc[j]))
                        st["note"] = (float(cen[j, 0]), float(cen[j, 1]))
                        print(f"{self.tag} demo: noted {p} at frame {i}", flush=True)
                elif st["note"] is not None:
                    st["gone"] = i
                    print(f"{self.tag} demo: {p} covered at frame {i}", flush=True)
            if all(st["gone"] is not None for st in state.values()):
                break
        # follow each cover from the moment of covering to the end of the demonstration
        for p, st in state.items():
            if st["note"] is None:
                continue
            note = st["note"]
            # ablation: AGENTMEM_DEMOFOLLOW=0 keeps the earliest sighting and never follows the cover
            if st["gone"] is not None and os.environ.get("AGENTMEM_DEMOFOLLOW", "1") == "1":
                try:
                    cen, _ = boxes_fn(frames[st["gone"]], cover_phrase)
                except Exception:
                    cen = np.zeros((0, 2))
                if len(cen):
                    trk = cen[int(np.argmin(np.hypot(cen[:, 0] - note[0], cen[:, 1] - note[1])))]
                    for i in [k for k in idx if k > st["gone"]]:
                        try:
                            cen, _ = boxes_fn(frames[i], cover_phrase)
                        except Exception:
                            continue
                        if not len(cen):
                            continue
                        d = np.hypot(cen[:, 0] - trk[0], cen[:, 1] - trk[1])
                        j = int(np.argmin(d))
                        if d[j] <= max_jump:
                            trk = cen[j]
                    moved = float(np.hypot(trk[0] - note[0], trk[1] - note[1]))
                    note = (float(trk[0]), float(trk[1]))
                    if moved > 8:
                        print(f"{self.tag} demo: cover of {p} moved {moved:.0f}px after covering "
                              f"-- following it", flush=True)
            self.notes[p] = note
        print(f"{self.tag} demonstration read: {len(self.notes)}/{len(self.watch)} notes "
              f"from {n} frames", flush=True)

    # ---------------------------------------------------------------- path memory
    # An ordered memory of WHERE the demonstration went. The cue is a target that changes its
    # appearance when reached -- here it flashes pure red for 40 steps, in the demonstration and
    # in the live phase alike -- so the memory is the ordered list of cue positions, and the live
    # phase advances through it on the same cue. A segment is named by the fixed front camera's
    # convention alone (image-down is the robot's forward, image-right its left); a homography
    # fitted on held-out episodes changes nothing (26/26 either way on PatternLock, offline).
    # The VLM keeps composing the subgoal; the memory only fills the slots it cannot see.
    DIRS = {"forward": (1, 0), "backward": (-1, 0), "left": (0, 1), "right": (0, -1),
            "forward-left": (1, 1), "forward-right": (1, -1),
            "backward-left": (-1, 1), "backward-right": (-1, -1)}
    DIR_SLOT = re.compile(r"^(\s*move\s+)((?:forward|backward)(?:-(?:left|right))?|left|right)(\s*)$", re.I)
    SIDE_SLOT = re.compile(r"\b(nearest\s+)(left|right)(\s+target)\b", re.I)
    SWING_SLOT = re.compile(r"\b(counterclockwise|clockwise)\b", re.I)

    @staticmethod
    def _cue_points(im, min_area: int = 8, merge: float = 11.0):
        """Centres of pure-red cue blobs (rings render at r=148 and r=231 with g,b<=8; an
        orange-red obstacle has g~66 and does not pass)."""
        from scipy import ndimage
        im = np.asarray(im)[..., :3]
        r, g, b = (im[..., i].astype(int) for i in range(3))
        lab, n = ndimage.label((r > 140) & (g < 30) & (b < 30))
        pts = []
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            if len(ys) >= min_area:
                pts.append([ys.mean(), xs.mean(), float(len(ys))])
        out = []                                  # rings of one cue share a centre: merge
        for y, x, a in sorted(pts, key=lambda q: -q[2]):
            for o in out:
                if np.hypot(y - o[0], x - o[1]) < merge:
                    o[2] += a; break
            else:
                out.append([y, x, a])
        return out

    @classmethod
    def _cue_step(cls, st: dict, im, k: int, gap: float = 12.0) -> List[Tuple[float, float]]:
        """Advance a cue tracker by one frame; returns the cues confirmed NEW at this frame.
        A cue must be seen in two consecutive frames (a one-frame glint under the stick tip is
        not a reached target) and is dated to its first frame."""
        cur = cls._cue_points(im)
        conf = [(cy, cx, ck) for cy, cx, ck in st["pend"]
                if any(np.hypot(cy - y, cx - x) < gap for y, x, _ in cur)]
        st["pend"] = [(y, x, k) for y, x, _ in cur
                      if not any(np.hypot(y - py, x - px) < gap for py, px, _ in st["prev"])
                      and not any(np.hypot(y - cy, x - cx) < gap for cy, cx, _ in conf)]
        st["prev"] = cur
        st["events"] += [((cy, cx), ck) for cy, cx, ck in conf]
        return [(cy, cx) for cy, cx, _ in conf]

    @staticmethod
    def _trail_side(im, A, B) -> Tuple[float, int]:
        """Mean signed offset of the white tip trail from chord A->B, in chord units (>0 is
        clockwise on screen). The trail is a thin curve (fill <= 0.4 of its box); the robot's
        white parts are compact blobs and are dropped."""
        from scipy import ndimage
        im = np.asarray(im)[..., :3]
        r, g, b = (im[..., i].astype(int) for i in range(3))
        m = (r > 200) & (g > 200) & (b > 200) & (abs(r - g) < 25) & (abs(g - b) < 25)
        lab, n = ndimage.label(m); pts = []
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
            if 8 <= len(ys) <= 400 and h < 70 and w < 70 and len(ys) / (h * w) < 0.45:
                pts += list(zip(ys, xs))
        if not pts:
            return 0.0, 0
        P = np.array(pts, float); A = np.array(A, float); B = np.array(B, float)
        d = B - A; L = float(np.linalg.norm(d)) + 1e-9; u = d / L
        q = P - A; t = q @ u; s = u[0] * q[:, 1] - u[1] * q[:, 0]
        keep = (t > -0.15 * L) & (t < 1.15 * L) & (abs(s) < 0.9 * L)
        if keep.sum() < 3:
            return 0.0, int(keep.sum())
        return float(np.mean(s[keep])) / L, int(keep.sum())

    @staticmethod
    def _flash_shape(im, cy, cx, radius: float = 14.0):
        """Is the red blob at (cy, cx) a reach-flash RING rather than a cube? Measured on 208
        saved rollouts: a flash has an empty centre (>=0.35 of a 5x5 window not red) and a ring
        fill (<=0.6 of its box) or is a thin sliver of ring peeking from under the cube; a cube
        is filled (hole 0, fill 0.6-0.96). Judged on the UNION of red pixels within `radius` of
        the cue centre: live frames render a re-triggered ring as several fragments, and the
        single fragment nearest the centre failed the test (v7 ep0: 0 flashes counted while
        the environment registered the swing). Returns (ok, area)."""
        im = np.asarray(im)[..., :3]
        r, g, b = (im[..., i].astype(int) for i in range(3))
        m = (r > 140) & (g < 30) & (b < 30)
        ys, xs = np.nonzero(m)
        if not len(ys):
            return False, 0
        near = np.hypot(ys - cy, xs - cx) <= radius
        ys, xs = ys[near], xs[near]
        if len(ys) < 8:
            return False, int(len(ys))
        h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        area = int(len(ys)); fill = area / (h * w)
        yc, xc = int(round(ys.mean())), int(round(xs.mean()))
        win = m[max(0, yc - 2): yc + 3, max(0, xc - 2): xc + 3]; hole = 1.0 - float(win.mean())
        sliver = min(h, w) <= 2 and area >= 8
        return bool(hole >= 0.35 and (fill <= 0.6 or sliver)), area

    def track_flashes(self, frame) -> int:
        """Count verified reach-flashes on the live frames: a red ring that appears at a fixed
        spot (two-frame persistence, ring shape, and no motion over the next six frames).
        Returns the count so far. A flash split by the cube above it counts once."""
        if frame is None:
            return 0
        if self._flash is None:
            self._flash = {"prev": self._cue_points(frame), "pend": [], "events": [], "k": 0,
                           "verify": [], "flashes": []}
            return 0
        st = self._flash; st["k"] += 1; k = st["k"]
        for cy, cx in self._cue_step(st, frame, k):
            ok, area = self._flash_shape(frame, cy, cx)
            if ok:
                st["verify"].append((cy, cx, k))
        keep = []
        for cy, cx, k0 in st["verify"]:
            if k < k0 + 3:                    # a static ring is settled in 3 frames; 6 cost a chunk
                keep.append((cy, cx, k0)); continue
            cur = self._cue_points(frame)
            d = [np.hypot(y - cy, x - cx) for y, x, _ in cur]
            if d and min(d) <= 1.5:
                last = st["flashes"][-1] if st["flashes"] else None
                # A flash at the SAME target as the last counted one is a re-trigger (the robot
                # left the ring, it faded, the robot came back), not the next step of the cycle:
                # the environment's task list advances only on the other target. v7b ep0: flash 3
                # at the left target again put the count one ahead of the sequence.
                if last is not None and np.hypot(cy - last[0], cx - last[1]) < 20:
                    print(f"{self.tag} re-lit at ({int(cy)}, {int(cx)}) step {k0}: same target, not counted", flush=True)
                elif last is None or k0 - last[2] > 3:
                    st["flashes"].append((cy, cx, k0))
                    print(f"{self.tag} flash {len(st['flashes'])} at ({int(cy)}, {int(cx)}) step {k0}",
                          flush=True)
        st["verify"] = keep
        return len(st["flashes"])

    @property
    def flashes(self) -> int:
        return len(self._flash["flashes"]) if self._flash else 0

    def gate_ordinal(self, subgoal: str) -> str:
        """Repetition counted by the environment's own reach-flash instead of by dwell time.
        The VLM's ordinal subgoals ("... for the second time") name the cycle's elements in
        order; the verified flash count says which element and which repetition is next. Until
        the first flash nothing is touched (tasks whose targets never flash stay the VLM's);
        after it, an ordinal subgoal is rewritten to the expected element and ordinal, and a
        non-ordinal subgoal is held to it while the goal's count (the agent's COUNT) is unmet."""
        if not subgoal:
            return subgoal
        m = self.ORDINAL.search(subgoal)
        if m:
            key = phase_key(self.ORDINAL.sub("", subgoal)).strip()
            if key not in self._ord_order:
                self._ord_order.append(key)
            self._ord_wording[key] = subgoal
        c = self.flashes
        if not self._ord_order:
            return subgoal
        if c == 0:
            # Before any flash the counter has no evidence -- unless the VLM proposes a SECOND
            # distinct ordinal element, which cannot be right before the first has completed
            # (measured: it leads within 16 steps on some episodes, then abandons the cycle).
            # From then on hold the first element until the first flash. A task whose steps
            # advance without flashes (PickXtimes: pick -> place, one ordinal element) never
            # trips this and is left alone.
            if m and len(self._ord_order) >= 2 and key != self._ord_order[0]:
                self._ord_hold = True
            if self._ord_hold:
                out = self.ORDINAL.sub("for the first time", self._ord_wording[self._ord_order[0]])
                if out != subgoal:
                    print(f"{self.tag} no flash yet: {subgoal!r} -> {out!r}", flush=True)
                return out
            return subgoal
        L = len(self._ord_order); n = self.cycle_n
        if n and c >= n * L:
            return subgoal                       # every repetition seen: let the VLM finish
        if not m and not n:
            return subgoal                       # no count to hold against
        key = self._ord_order[c % L]; k = c // L + 1
        if k > len(self.ORD_WORDS):
            return subgoal
        out = self.ORDINAL.sub(f"for the {self.ORD_WORDS[k-1]} time", self._ord_wording[key])
        if out != subgoal:
            print(f"{self.tag} flashes={c}: {subgoal!r} -> {out!r}", flush=True)
        return out

    # ------------------------------------------------------- demonstrated objects / places
    # The demonstration can name an OBJECT ("the same block that was previously picked up") or a
    # PLACE ("the first target it was previously placed on") that the live scene cannot identify.
    # The cue is the cube blob itself: the cube the demonstration lifted is the first resting
    # blob of its colour to move; the places are where the blob dwells, in order. Offline on 13
    # episodes each: picked cube within 8 px on 9/13, k-th place on 12/13.
    ORDW = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4}

    @staticmethod
    def _colour_blobs(im, colour: str, min_area: int = 12):
        from scipy import ndimage
        im = np.asarray(im)[..., :3].astype(int); r, g, b = im[..., 0], im[..., 1], im[..., 2]
        m = {"red": (r > 140) & (g < 60) & (b < 60), "green": (g > 120) & (r < 80) & (b < 80),
             "blue": (b > 120) & (r < 80) & (g < 80)}.get(colour)
        if m is None:
            return []
        lab, n = ndimage.label(m); out = []
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            if len(ys) >= min_area and ys.mean() > 30:      # not the wall / robot band
                out.append((float(ys.mean()), float(xs.mean()), int(len(ys))))
        return out

    def observe_demo_objects(self, frames, task_goal: str, stride: int = 2) -> None:
        """Notes from the demonstration for references the live scene cannot resolve:
        'correct cube' = the cube the demonstration lifted (its position at the demo's end),
        'correct target' = the k-th place the cube dwelled at (k named in the goal)."""
        if frames is None or not len(frames):
            return
        fr = [frames[i] for i in range(0, len(frames), max(1, stride))]
        goal = (task_goal or "").lower()
        colour = next((c for c in ("red", "green", "blue") if c in goal), None)
        if colour is None:                     # unnamed: the colour with most resting cubes at start
            colour = max(("red", "green", "blue"), key=lambda c: len(self._colour_blobs(fr[0], c)))
        F = [[(y, x) for y, x, a in self._colour_blobs(f, colour)] for f in fr]
        if not F[0]:
            return
        # (a) the lifted cube: greedy per-cube nearest-neighbour tracks; first to move > 6 px or
        #     vanish for two frames; followed with a wider jump to the end
        start = list(F[0]); pos = list(start); lost = [0] * len(start); picked = None
        for t in range(1, len(F)):
            claimed = set()
            for i in range(len(start)):
                if picked is not None and i != picked and lost[i] > 0:
                    continue
                d = [(np.hypot(y - pos[i][0], x - pos[i][1]), j) for j, (y, x) in enumerate(F[t]) if j not in claimed]
                lim = 25 if i == picked else 10
                if d and min(d)[0] <= lim:
                    j = min(d)[1]; pos[i] = F[t][j]; claimed.add(j); lost[i] = 0
                else:
                    lost[i] += 1
                if picked is None and (np.hypot(pos[i][0] - start[i][0], pos[i][1] - start[i][1]) > 6 or lost[i] >= 2):
                    picked = i
        if picked is not None and "pick" in goal:
            self.notes["correct cube"] = (float(pos[picked][0]), float(pos[picked][1]))
            print(f"{self.tag} demo: the lifted {colour} cube ends at {tuple(int(v) for v in pos[picked])}", flush=True)
        # (b) the places: the largest blob's continuous track (no jump > 30 px), dwells of >= 10
        #     frames within 3 px, distinct places > 10 px apart, the start dropped
        if "target" in goal or "place" in goal:
            track = []; cur = None
            for t in range(len(F)):
                if not F[t]:
                    track.append(None); continue
                if cur is None:
                    big = max(self._colour_blobs(fr[t], colour), key=lambda b: b[2]); cur = (big[0], big[1])
                else:
                    d = [np.hypot(y - cur[0], x - cur[1]) for y, x in F[t]]; j = int(np.argmin(d))
                    if d[j] > 30:
                        track.append(None); continue
                    cur = F[t][j]
                track.append(cur)
            dwells, run = [], []
            for q in track + [None]:
                if q is not None and run and np.hypot(q[0] - run[0][0], q[1] - run[0][1]) < 3:
                    run.append(q); continue
                if len(run) >= 10:
                    c = (float(np.mean([r[0] for r in run])), float(np.mean([r[1] for r in run])))
                    if not dwells or np.hypot(c[0] - dwells[-1][0], c[1] - dwells[-1][1]) > 10:
                        dwells.append(c)
                run = [q] if q is not None else []
            # Only an ORDINAL reference ("the second target it was placed on") is answerable
            # from the ordered places; a temporal one ("the target right after the button was
            # pressed") needs a press cue this camera does not give (VideoPlaceButton: filling
            # the first place there took the task from 7/13 to 0/7 -- the memory must stay out).
            k = next((self.ORDW[w] for w in self.ORDW if f"the {w} target" in goal), None)
            places = dwells[1:]
            if k is not None and k < len(places):
                self.notes["correct target"] = places[k]
                print(f"{self.tag} demo: {len(places)} places visited; the {k+1}. is {tuple(int(v) for v in places[k])}", flush=True)

    def fill_demo_reference(self, subgoal: str) -> Optional[str]:
        """'the correct cube / target' in the VLM's subgoal refers to the demonstration: put the
        remembered position into its coordinate slot."""
        if not subgoal:
            return None
        for ref in ("correct cube", "correct target"):
            if ref in subgoal.lower() and ref in self.notes and re.search(r"<\s*\d+\s*,\s*\d+\s*>", subgoal):
                y, x = self.notes[ref]
                return re.sub(r"<\s*\d+\s*,\s*\d+\s*>", f"<{int(round(y))}, {int(round(x))}>", subgoal, count=1)
        return None

    def observe_path(self, frames) -> None:
        """Read the demonstrated route: ordered cue events, each segment named by direction
        (8-compass), side (left/right of the previous cue) and turn sense (from the trail)."""
        self.path = []
        st = {"prev": [], "pend": [], "events": []}
        for k, im in enumerate(frames):
            self._cue_step(st, im, k)
        ev = st["events"]
        for i in range(len(ev) - 1):
            (ay, ax), ka = ev[i]; (by, bx), kb = ev[i + 1]
            d = np.array([by - ay, bx - ax], float); d /= (np.linalg.norm(d) + 1e-9)
            name = max(self.DIRS, key=lambda n: float(np.dot(d, np.array(self.DIRS[n], float)
                                                              / np.linalg.norm(self.DIRS[n]))))
            lo, hi = max(kb - 1, ka + 2), min(kb + 3, len(frames))
            acc = [self._trail_side(frames[f], (ay, ax), (by, bx)) for f in range(lo, hi)]
            npx = sum(n for _, n in acc)
            sgn = sum(v * n for v, n in acc) / npx if npx else 0.0
            self.path.append({"dir": name, "side": "left" if bx > ax else "right",
                              "swing": "clockwise" if sgn > 0 else "counterclockwise",
                              "from": (ay, ax), "to": (by, bx)})
        print(f"{self.tag} route read: {len(ev)} cues, {len(self.path)} segments "
              f"{[p['dir'] for p in self.path]} from {len(frames)} frames", flush=True)

    def begin_live(self, frame) -> None:
        """Start the live cue tracker; cues already lit in the first live frame are not events."""
        self._live = {"prev": self._cue_points(frame) if frame is not None else [],
                      "pend": [], "events": [], "k": 0}

    def track_path(self, frame) -> int:
        """Advance the live tracker; returns how many route cues have been reached so far."""
        if not self.path or frame is None:
            return 0
        if self._live is None:
            self.begin_live(frame); return 0
        self._live["k"] += 1
        new = self._cue_step(self._live, frame, self._live["k"])
        if new:
            print(f"{self.tag} route: cue {len(self._live['events'])}/{len(self.path)} reached "
                  f"at {tuple(int(v) for v in new[0])}", flush=True)
        return len(self._live["events"])

    def path_fill(self, subgoal: str) -> Optional[str]:
        """Fill the VLM's own subgoal with the current segment of the remembered route. Returns
        None when the subgoal has no route slot (this is not a route task)."""
        if not self.path or not subgoal:
            return None
        done = len(self._live["events"]) if self._live else 0
        seg = self.path[min(done, len(self.path) - 1)]
        m = self.DIR_SLOT.match(subgoal)
        if m:
            return f"{m.group(1)}{seg['dir']}{m.group(3)}"
        if self.SIDE_SLOT.search(subgoal) and self.SWING_SLOT.search(subgoal):
            out = self.SIDE_SLOT.sub(lambda mm: f"{mm.group(1)}{seg['side']}{mm.group(3)}", subgoal, count=1)
            return self.SWING_SLOT.sub(seg["swing"], out, count=1)
        return None

    NUM = {"one": 1, "once": 1, "two": 2, "twice": 2, "three": 3, "four": 4, "five": 5,
           "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}

    def plan_cycle(self, task_goal: str) -> None:
        """Read the repetition structure out of the goal: what repeats, how often, what follows.

        SwingXtimes is the case that motivated this. The agent's right/left alternation is 100%
        correct; it loses only the count, and then terminates early -- "press the button" while
        the environment still wants swings -- on 33-35% of steps, and the robot never finishes.
        Counting its own confident visits and refusing the final action until the goal's count is
        reached removed premature termination entirely offline (33% -> 0%) and raised subgoal match
        ~18 points. Holding the NEXT element of the cycle mattered (63% vs 49% for repeating the
        last one), and the cycle is in the prompt, so the agent reads it rather than a rule.
        """
        self.cycle, self.cycle_n, self.final = [], None, ""
        self._cycle_hits, self._cyc_key, self._cyc_start = [], None, None
        self._cyc_counted, self._last_cycle_subgoal = False, None
        self._wording = {}
        # The agent's COUNT is needed by the flash-counted gate too (to know when the
        # repetitions are complete); the dwell-time gate itself stays behind AGENTMEM_CYCLE.
        if not task_goal or (os.environ.get("AGENTMEM_CYCLE", "1") != "1"
                             and os.environ.get("AGENTMEM_ORD", "0") != "1"):
            return
        try:
            ans = self._ask(CYCLE_PROMPT.format(goal=task_goal.strip()), max_tokens=80).lower()
        except Exception as e:
            print(f"{self.tag} cycle query failed: {e!r}", flush=True)
            return
        m = re.search(r"cycle\s*:\s*(.+)", ans)
        if m and "none" not in m.group(1)[:8]:
            self.cycle = [x.strip().strip(".") for x in re.split(r"[;,]", m.group(1)) if x.strip()]
        m = re.search(r"count\s*:\s*(\w+)", ans)
        if m:
            w = m.group(1)
            self.cycle_n = self.NUM.get(w, int(w) if w.isdigit() else None)
        m = re.search(r"final\s*:\s*(.+)", ans)
        if m and "none" not in m.group(1)[:8]:
            self.final = m.group(1).strip().strip(".")
        # Sanity checks on the agent's own answer, none of them task-specific:
        #  * a "cycle" that repeats once is a sequence, not a repetition (count >= 2)
        #  * a cycle with a duplicated element is an ENUMERATION, not a repeated motion --
        #    on BinFill the agent lists "put a blue cube into the bin" four times, and gating
        #    on that would hold the final action forever
        #  * the goal itself must speak of repeating; otherwise there is nothing to count
        dedup = [c for i, c in enumerate(self.cycle) if c not in self.cycle[:i]]
        enumerating = len(dedup) != len(self.cycle)
        says_repeat = bool(re.search(r"\b(\w+)\s+times\b|\brepeat", (task_goal or "").lower()))
        if enumerating:
            print(f"{self.tag} cycle rejected: enumeration, not repetition {self.cycle}", flush=True)
        if self.cycle and self.cycle_n and self.cycle_n >= 2 and not enumerating and says_repeat:
            self._cycle_hits = [0] * len(self.cycle)
            print(f"{self.tag} cycle {self.cycle} x{self.cycle_n}, then {self.final!r}", flush=True)
        else:
            self.cycle = []

    def _cycle_index(self, subgoal: str) -> Optional[int]:
        """Which element of the cycle does this subgoal belong to?

        The VERB decides first. Token overlap alone ties on PickXtimes: "place the green cube
        onto the target" shares {green, cube} with "pick up the green cube" and {place, target}
        with "place it on the target", and a tie resolved to the first element counted every
        place-step as a pick -- the counter stayed at 0/3 and the gate held the button press on
        a task that already scored 12/13. Live, in v3, for one full episode. A subgoal's first
        word is its action; only if that does not single out an element does overlap decide.
        """
        if not self.cycle:
            return None
        words = re.findall(r"[a-z\-]+", subgoal.lower())
        if not words:
            return None
        verb = words[0]
        by_verb = [i for i, c in enumerate(self.cycle)
                   if (re.findall(r"[a-z\-]+", c.lower()) or [""])[0] == verb]
        if len(by_verb) == 1:
            return by_verb[0]
        toks = set(words)
        best, score = None, 0
        for i, c in enumerate(self.cycle):
            ct = set(re.findall(r"[a-z\-]+", c.lower())) - {"the", "a", "to", "it", "of", "on"}
            ov = len(toks & ct)
            if ov > score:
                best, score = i, ov
        return best

    def _is_final(self, subgoal: str) -> bool:
        if not self.final:
            return False
        ft = set(re.findall(r"[a-z\-]+", self.final.lower())) - {"the", "a", "to", "it", "of", "on"}
        return bool(ft) and len(ft & set(re.findall(r"[a-z\-]+", subgoal.lower()))) >= max(1, len(ft) // 2)

    ORDINAL = re.compile(r"for the (first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth) time")
    ORD_WORDS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
    # A visit counts once a cycle element has been held this many env steps. Swept on the real
    # streams of both repetition tasks: 16 lets flicker inflate the count (SwingXtimes premature
    # termination only falls to 15%), 48 under-counts PickXtimes' short place-phase (15 of 59
    # button presses wrongly held). 32 releases all 59 and drives SwingXtimes to 0%.
    HOLD_STEPS = int(os.environ.get("AGENTMEM_CYCLE_HOLD", "32"))
    TOLERANCE = int(os.environ.get("AGENTMEM_CYCLE_TOL", "1"))

    def gate_cycle(self, subgoal: str, step: int) -> str:
        """Count the agent's own visits to cycle elements; refuse leaving the cycle until the
        goal's count is met, emitting the next expected element instead; rewrite ordinals from
        the count. Reads nothing but the goal (once) and the agent's own history."""
        if not self.cycle or not self.cycle_n:
            return subgoal
        key = phase_key(subgoal)            # coordinate-free: the VLM's numbers jitter every call
        if key != self._cyc_key:
            self._cyc_key, self._cyc_start, self._cyc_counted = key, step, False
        idx = self._cycle_index(subgoal)
        in_cycle = idx is not None and not self._is_final(subgoal)
        if in_cycle:
            if not self._cyc_counted and (step - self._cyc_start) >= self.HOLD_STEPS:
                self._cycle_hits[idx] += 1
                self._cyc_counted = True
            self._last_cycle_subgoal = subgoal
            self._wording[idx] = subgoal
            k = max(self._cycle_hits[idx] + (0 if self._cyc_counted else 1), 1)
            if self.ORDINAL.search(subgoal) and k <= len(self.ORD_WORDS):
                return self.ORDINAL.sub(f"for the {self.ORD_WORDS[k-1]} time", subgoal)
            return subgoal
        # once the cycle has begun, ANYTHING outside it is leaving early; the goal names only the
        # last action, but the oracle inserts others (put the cube down) before it
        if not in_cycle and self._last_cycle_subgoal is not None:
            done = self._cycle_hits[-1] if self._cycle_hits else 0   # a repetition ends at its last element
            # Tolerate an under-count of one. Any fixed dwell misses some visit somewhere -- live,
            # one short place-phase left the counter at 3/4 and the gate refused a button press the
            # oracle agreed with, which is a certain timeout. A false release is only the baseline's
            # behaviour (12/13 on PickXtimes); a false hold is 0/29 at the cap. Premature
            # termination on SwingXtimes happens at 0 or 1 of 3, which this still catches.
            if done < self.cycle_n - self.TOLERANCE:
                last = self._cycle_index(self._last_cycle_subgoal)
                nxt = (last + 1) % len(self.cycle) if last is not None else 0
                # hold the agent's OWN last wording of the next element; it has usually said it
                # before, and its phrasing need not contain the extractor's phrasing verbatim
                held = self._wording.get(nxt)
                if held is None:
                    held = self._last_cycle_subgoal
                    if last is not None and self.cycle[last].lower() in held.lower():
                        held = held.replace(self.cycle[last], self.cycle[nxt])
                k = self._cycle_hits[nxt] + 1
                if self.ORDINAL.search(held) and k <= len(self.ORD_WORDS):
                    held = self.ORDINAL.sub(f"for the {self.ORD_WORDS[k-1]} time", held)
                print(f"{self.tag} final action requested after {done}/{self.cycle_n} "
                      f"repetitions -- holding {held!r}", flush=True)
                return held
        return subgoal

    def resolve(self, subgoal: str) -> Optional[str]:
        """Which recorded object does this subgoal name in order to identify something ELSE?

        Memory is the right source for a coordinate only when the remembered thing is what tells
        the step's target apart from its lookalikes -- "the container THAT HIDES the blue cube"
        wants the container and names the cube only to say which one. Two other shapes look
        similar and must not trigger a read:

          the step wants the remembered object itself  "pick up the first blue cube"
          the remembered object is a separate argument  "place the blue cube onto the target"

        English separates all three without any task knowledge. The first is ruled out by
        comparing head noun to head noun; the second by position, because a restrictive modifier
        follows the noun it restricts while a separate argument precedes it. Measured over 1478
        steps against the rules this replaces: 100% recall, and precision 36% -> 44% -> 60% as
        each test is added.
        """
        if not self.notes or not subgoal:
            return None
        text = subgoal.lower()
        cut = text.find("at <")
        pre = text[:cut] if cut > 0 else text
        head = (phrase_of(pre) or "").lower()
        hpos = pre.rfind(head) if head else -1
        for phrase in self.notes:
            toks = phrase.lower().split()[-2:]
            if not all(t in text for t in toks):
                continue
            if head and head == (phrase_of(phrase.lower()) or phrase.lower()):
                continue                       # the step is asking for the note itself
            if hpos >= 0 and max(text.rfind(t) for t in toks) < hpos:
                continue                       # named before the head noun: a separate argument
            return phrase
        return None

    def observe_phase(self, subgoal: str, step: int) -> None:
        """Track how long the agent has been asking for the same thing.

        Retrieval answers *what* to recall and has no notion of *when* recall is legal. Measured
        consequence: on ButtonUnmask the agent names the container at step ~96, before the buttons
        are pressed, and memory answers confidently -- 84-91% within 8px of the true container --
        so the policy commits to the wrong object and the episode never advances. The no-memory
        baseline's container guess is poor, so it flounders and drifts back. A better answer to
        the wrong question is worse than a bad one.

        The agent does work the prerequisite first (its subgoal matches the oracle for the first
        ~96 steps); it leaves too early. The only signal for "too early" available from prompt and
        image alone is duration, so that is what is used.
        """
        key = phase_key(subgoal)
        if key != self._phase_key:
            if self._phase_key is not None and key != self._stale_key:
                self._prev_subgoal, self._prev_key = self._phase_subgoal, self._phase_key
            if key != self._stale_key:
                self._stale_key = None            # the agent moved on to something new
            self._phase_key, self._phase_start = key, step
        self._phase_subgoal = subgoal
        self._step = step

    def stale_revert(self) -> Optional[str]:
        """The subgoal to send instead, once the current request has gone stale.

        Withholding the memory note is not enough: measured on the gated arm, after the note is
        withheld the agent still asks for the container on 100% of steps and the fallback detector
        refinement hands the policy a good container coordinate on 92% of them -- the premature
        subgoal stays actionable and all 12 failures still ran to the cap. The baseline recovers
        because it drifts BACK to the prerequisite. So do that explicitly: while the request is
        stale, hand back the previous phase's subgoal -- text and coordinate -- and keep doing so
        until the agent itself moves on to something other than the stale request. Offline,
        revert-on-long-phase took subgoal match from 17.8% to 59.1% on the memory arms.
        """
        if not self.phase_is_stale() or self._prev_subgoal is None:
            return None
        if self._stale_key is None:
            self._stale_key, self._revert_start = self._phase_key, self._step
            print(f"{self.tag} {self._phase_key[:40]!r} unachieved for {self.STALE_AFTER}+ steps "
                  f"-- reverting to {self._prev_key[:40]!r}", flush=True)
        elif self._step - self._revert_start >= self.STALE_AFTER:
            # BOUNDED: the reverted phase has had as long as the stale one. Measured live, a
            # sticky revert reaches the prerequisite far more often (10/14 vs 21/50) and then
            # keeps overriding the agent's now-LEGITIMATE next request, because it carries the
            # same key that went stale. Release, and let the agent's request stand; if it stalls
            # again the timer restarts and the revert fires again.
            print(f"{self.tag} revert released after {self.STALE_AFTER} steps", flush=True)
            self._stale_key, self._phase_start = None, self._step
            return None
        return self._prev_subgoal

    def phase_is_stale(self) -> bool:
        """Has the current request gone unanswered long enough to doubt it is the right one?"""
        if self._phase_start is None:
            return False
        if self._stale_key is not None and self._phase_key == self._stale_key:
            return True                       # sticky: still repeating the request that went stale
        return (self._step - self._phase_start) >= self.STALE_AFTER

    def consult(self, subgoal: str, frame=None, boxes_fn=None) -> Optional[str]:
        """Retrieval. The frame arguments are accepted and ignored.

        An occlusion check belongs here in principle -- a note means something only once what it
        describes is gone -- and it was tried: ask the detector whether the remembered object is
        still in the picture and refuse the read if it is. It does not work. Grounding DINO is a
        generic object proposer whose boxes barely move with the prompt, so it returns a
        confident box for almost any phrase and "still visible" carries little signal. Swept over
        eight thresholds it beat plain reference resolution by three F1 points at best while
        costing 14% of the true reads. The same prompt-independence closed the event layer.
        """
        if self.phase_is_stale():
            # The subgoal has not been achieved in a long time, so it is probably premature.
            # Refuse the READ rather than rewrite the subgoal: the text is left exactly as the
            # composer wrote it and only the confident coordinate is withheld, which restores the
            # flounder-and-recover behaviour the no-memory baseline recovers by.
            if not self._stale_logged:
                print(f"{self.tag} subgoal unachieved for {self.STALE_AFTER}+ steps -- "
                      f"withholding the note until it advances", flush=True)
                self._stale_logged = True
            return None
        rmode = os.environ.get("AGENTMEM_RETRIEVAL", "structural")   # ablations: loose | agent
        if rmode == "loose":
            hit = self._resolve_loose(subgoal)
        elif rmode == "agent":
            hit = self._resolve_agent(subgoal)
        else:
            hit = self.resolve(subgoal)
        if hit:
            print(f"{self.tag} step is identified by {hit!r} -- reading the note", flush=True)
        return hit

    def _resolve_loose(self, subgoal: str) -> Optional[str]:
        """Ablation rung: the first note whose words appear in the subgoal, with neither the
        head-noun nor the modifier-position test."""
        if not self.notes or not subgoal:
            return None
        text = subgoal.lower()
        for phrase in self.notes:
            toks = phrase.lower().split()[-2:]
            if all(t in text for t in toks):
                return phrase
        return None

    def _resolve_agent(self, subgoal: str) -> Optional[str]:
        """Ablation rung: ask the agent per subgoal whether the step relates to a note. Cached
        on the coordinate-free subgoal, so one call per distinct request."""
        if not self.notes or not subgoal:
            return None
        key = phase_key(subgoal)
        if key in self._consulted:
            return self._consulted[key]
        keys = "\n".join(f"- {p}" for p in self.notes)
        try:
            ans = self._ask(CONSULT_PROMPT.format(subgoal=subgoal, keys=keys), max_tokens=24).lower()
        except Exception as e:
            print(f"{self.tag} consult query failed: {e!r}", flush=True)
            ans = ""
        hit = None
        if "none" not in ans[:12]:
            for p in self.notes:
                if p.lower() in ans or all(t in ans for t in p.lower().split()[-2:]):
                    hit = p
                    break
        self._consulted[key] = hit
        return hit

    def cover_phrase(self, frame=None, boxes_fn=None, near=None,
                     radius: float = 24.0) -> Optional[str]:
        """What do the covers look like? The agent proposes, the detector decides.

        The note says where the cube WAS, so the detector needs a phrase for the covers now
        standing there. Asked without the picture the agent answered "dark, flat cloth" for white
        containers; shown the picture it answered "gray rectangular cover", which sounds right and
        is worse -- measured over 160 cover-selection frames it returns no box at all on 82% of
        them and lands on the cover 2% of the time, against 83% for the fitted phrase. Free-form
        description falls outside the detector's vocabulary, and a phrase the detector cannot act
        on is not a description, it is a failure.

        So the agent's answer is checked before it is used, against the thing the note is for: a
        usable phrase must put a box near the recorded position, because that is where the cover
        has to be. Candidates are tried in the agent's own order and the first that passes is
        kept. If none pass, the caller falls back to the path every arm already uses, so this
        decision can help and cannot hurt.
        """
        if self._cover is not None:
            return self._cover or None
        keys = "\n".join(f"- {p}" for p in self.notes) or "- an object"
        try:
            ans = self._ask(COVER_PROMPT.format(keys=keys), max_tokens=32, image=frame)
        except Exception as e:
            print(f"{self.tag} cover query failed: {e!r}", flush=True)
            ans = ""
        cands = []
        for c in re.split(r"[;\n]", re.sub(r'["\'.]', "", ans)):
            c = " ".join(c.strip().lower().split()[:3])
            if c and c not in cands:
                cands.append(c)

        chosen = ""
        for c in cands[:3]:
            if boxes_fn is None or near is None:
                chosen = c
                break
            try:
                cen, _ = boxes_fn(frame, c)
            except Exception:
                continue
            if not len(cen):
                print(f"{self.tag} {c!r} finds nothing", flush=True)
                continue
            d = float(np.hypot(cen[:, 0] - near[0], cen[:, 1] - near[1]).min())
            if d <= radius:
                chosen = c
                break
            print(f"{self.tag} {c!r} finds nothing within {radius:.0f}px of the note "
                  f"(nearest {d:.0f}px)", flush=True)

        self._cover = chosen
        print(f"{self.tag} covers look like {chosen!r}" if chosen else
              f"{self.tag} no proposal survived checking, using the shared phrase", flush=True)
        return chosen or None

    def recall(self, phrase: str) -> Optional[Tuple[float, float]]:
        st = self._trk.get(phrase)
        if st and st["gone"] is not None and st["pos"] is not None:
            note = self.notes.get(phrase)
            if note and np.hypot(st["pos"][0] - note[0], st["pos"][1] - note[1]) > 8:
                print(f"{self.tag} cover of {phrase} moved "
                      f"{np.hypot(st['pos'][0] - note[0], st['pos'][1] - note[1]):.0f}px -- reading the tracked position", flush=True)
            return st["pos"]
        return self.notes.get(phrase)

    @property
    def stats(self) -> dict:
        return {"need": self.need, "source": self.source, "watch": self.watch,
                "notes": len(self.notes), "cover": self._cover, "calls": self.calls,
                "route": len(self.path), "reached": len(self._live["events"]) if self._live else 0,
                "flashes": self.flashes,
                "tracked": {p: st["gone"] for p, st in self._trk.items() if st["gone"] is not None}}
