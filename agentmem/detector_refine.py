"""Refine a VLM's grounded subgoal coordinates with an open-vocabulary detector.

The division of labour follows what the validation measured on 360 frames:

  the VLM decides WHICH object       -- ordinals, counts, occlusion, task history.
                                        A detector cannot do this; it is memory.
  the detector decides WHERE it is   -- best-of-k median error 1.6 px, versus the
                                        VLM's median 8.1 px measured in-distribution.

So we keep the VLM's sentence verbatim and rewrite only the numbers inside `at <y, x>`,
snapping each to the nearest detected box centre.

Two guards, both taken from the measured error distribution rather than picked by feel:

  SNAP_RADIUS  The VLM's error is bimodal -- a tight cluster below 12 px (right object,
               imprecise) and a tail above 32 px (wrong object), with nothing in between.
               Snapping helps the first group and actively hurts the second, where it
               would land precisely on the wrong cube. A gate in the empty region keeps
               the good case and declines the bad one.
  MAX_BOX      The highest-scoring box for "open box" is often the robot arm (median
               99 px). Object boxes here are 17-38 px a side, so oversized boxes are
               rejected outright.

Coordinates are `<y, x>` in 256-space throughout, matching the benchmark's convention.
"""
import json, os, re, threading
from typing import List, Optional

import numpy as np

COORD = re.compile(r"<(\d+),\s*(\d+)>")

# Role words in the subgoals vs appearance words the detector responds to. Measured:
# "container" -> "white cube" moved best-of-k from 62.5% to 80.0%, "target" -> "bullseye"
# from 87.5% to 100%.
VOCAB = ["red cube", "green cube", "blue cube", "container", "button", "target",
         "peg", "bin", "stick", "cube"]
PROMPT = {"container": "white cube", "bin": "open box", "target": "bullseye",
          "cube": "colored cube", "button": "button", "peg": "peg", "stick": "stick",
          "red cube": "red cube", "green cube": "green cube", "blue cube": "blue cube"}

# Reference colours, calibrated offline from oracle patches on a held-out-split tune half.
# At deploy these are constants; only the image and the phrase are read.
COLOUR_REF = {"red": (192.0, 6.0, 6.0), "green": (6.0, 181.0, 6.0), "blue": (5.0, 5.0, 196.0)}
COLOUR_W = float(os.environ.get("DET_COLOUR_W", "0.1"))
# "left-side" resolves to LARGE x: the camera faces the robot, so the robot's left is the
# image's right. Verified against oracle coordinates, not assumed.
LEFT_IS_LARGE_X = os.environ.get("DET_LEFT_BIG", "1") == "1"
SPATIAL_TOPN = int(os.environ.get("DET_SPATIAL_TOPN", "3"))
_LEFT = re.compile(r"\bleft-side\b|\bleftmost\b|\bleft\b")
_RIGHT = re.compile(r"\bright-side\b|\brightmost\b|\bright\b")

MODEL_ID = os.environ.get("DET_MODEL", "IDEA-Research/grounding-dino-base")
BOX_TH = float(os.environ.get("DET_BOX_TH", "0.15"))
TEXT_TH = float(os.environ.get("DET_TEXT_TH", "0.15"))
SNAP_RADIUS = float(os.environ.get("DET_SNAP_RADIUS", "24"))
MAX_BOX = float(os.environ.get("DET_MAX_BOX", "48"))


def phrase_of(pre: str) -> Optional[str]:
    """Rightmost-ENDING vocabulary match, longest on a tie.

    Ranking on the match start would let "cube" beat "green cube", discarding the colour
    that is often the only thing separating three identical shapes.
    """
    pre = pre.lower()
    best, bestkey = None, None
    for v in VOCAB:
        pos = pre.rfind(v)
        if pos < 0:
            continue
        key = (pos + len(v), len(v))
        if bestkey is None or key > bestkey:
            best, bestkey = v, key
    return best


class DetectorRefiner:
    """Lazily-loaded Grounding DINO wrapper. Never raises into the eval loop."""

    def __init__(self):
        self._model = None
        self._proc = None
        self._lock = threading.Lock()
        self.stats = {"seen": 0, "snapped": 0, "no_box": 0, "out_of_radius": 0,
                      "error": 0, "cached": 0}
        # The benchmark fills `at <y, x>` only when the subgoal TEXT changes and caches the
        # result for the rest of the segment (segmentation_utils.py: `if
        # current_subgoal_segment != previous_subgoal_segment`). So the oracle coordinate is
        # the object's position at subgoal ONSET and does not follow the object as the robot
        # moves it. Re-detecting every chunk therefore disagrees with the oracle by a growing
        # margin -- measured at 4.3px median at onset versus 32px later in the segment.
        # Caching by subgoal text reproduces the benchmark's own behaviour, and uses only our
        # own previous output, nothing privileged.
        self._last_key = None
        self._last_out = None

    def _ensure(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            self._torch = torch
            self._proc = AutoProcessor.from_pretrained(MODEL_ID)
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                MODEL_ID).to(dev).eval()
            self._dev = dev
            print(f"[detector_refine] {MODEL_ID} on {dev} "
                  f"(box_th={BOX_TH} radius={SNAP_RADIUS} max_box={MAX_BOX})", flush=True)

    def _boxes(self, image: np.ndarray, prompt: str):
        from PIL import Image
        self._ensure()
        pil = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
        inp = self._proc(images=[pil], text=[prompt.lower().rstrip(".") + "."],
                         return_tensors="pt").to(self._dev)
        with self._torch.no_grad():
            out = self._model(**inp)
        res = self._proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=BOX_TH, text_threshold=TEXT_TH,
            target_sizes=[pil.size[::-1]])[0]
        b = res["boxes"].cpu().numpy()
        if not len(b):
            return np.empty((0, 2))
        sc = res["scores"].cpu().numpy()
        keep = np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]) <= MAX_BOX
        b, sc = b[keep], sc[keep]
        if not len(b):
            return np.empty((0, 2)), np.empty((0,))
        cen = np.stack([(b[:, 1] + b[:, 3]) / 2, (b[:, 0] + b[:, 2]) / 2], axis=1)  # (y, x)
        return cen, sc

    def refine(self, subgoal: str, image: Optional[np.ndarray]) -> str:
        """Rewrite each `at <y, x>` to the nearest plausible detection. Text is untouched."""
        if not subgoal or image is None or "<" not in subgoal:
            return subgoal
        try:
            out, last = [], 0
            for m in COORD.finditer(subgoal):
                self.stats["seen"] += 1
                py, px = int(m.group(1)), int(m.group(2))
                ph = phrase_of(subgoal[:m.start()])
                new = None
                if ph:
                    cen, _ = self._boxes(image, PROMPT.get(ph, ph))
                    if not len(cen):
                        self.stats["no_box"] += 1
                    else:
                        d = np.hypot(cen[:, 1] - px, cen[:, 0] - py)
                        j = int(np.argmin(d))
                        if d[j] <= SNAP_RADIUS:
                            new = (int(round(cen[j, 0])), int(round(cen[j, 1])))
                            self.stats["snapped"] += 1
                        else:
                            self.stats["out_of_radius"] += 1
                out.append(subgoal[last:m.start()])
                out.append(f"<{new[0]}, {new[1]}>" if new else m.group(0))
                last = m.end()
            out.append(subgoal[last:])
            return "".join(out)
        except Exception as e:  # never take down an evaluation
            self.stats["error"] += 1
            if self.stats["error"] <= 3:
                print(f"[detector_refine] refine failed: {e!r}", flush=True)
            return subgoal


    def replace(self, subgoal: str, image: Optional[np.ndarray], mode: str = "top1") -> str:
        """Substitute detector coordinates for the ones already in `subgoal`.

        mode="top1"    the highest-scoring box. Uses nothing but the image and the phrase,
                       so this is what a deployed system could actually produce.
        mode="nearest" the box closest to the coordinate already present. That coordinate is
                       the oracle's, so this is NOT deployable -- it is the ceiling, telling
                       us whether detector precision would suffice if instance selection
                       were solved by some other means.
        """
        if not subgoal or image is None or "<" not in subgoal:
            return subgoal
        try:
            out, last = [], 0
            for m in COORD.finditer(subgoal):
                self.stats["seen"] += 1
                oy, ox = int(m.group(1)), int(m.group(2))
                ph = phrase_of(subgoal[:m.start()])
                new = None
                if ph:
                    cen, sc = self._boxes(image, PROMPT.get(ph, ph))
                    if not len(cen):
                        self.stats["no_box"] += 1
                    else:
                        if mode == "nearest":
                            j = int(np.argmin(np.hypot(cen[:, 1] - ox, cen[:, 0] - oy)))
                        else:
                            j = int(np.argmax(sc))
                        new = (int(round(cen[j, 0])), int(round(cen[j, 1])))
                        self.stats["snapped"] += 1
                out.append(subgoal[last:m.start()])
                out.append(f"<{new[0]}, {new[1]}>" if new else m.group(0))
                last = m.end()
            out.append(subgoal[last:])
            return "".join(out)
        except Exception as e:
            self.stats["error"] += 1
            if self.stats["error"] <= 3:
                print(f"[detector_refine] replace failed: {e!r}", flush=True)
            return subgoal


    @staticmethod
    def _side(pre: str):
        """Which side the phrase names, or None if it names neither or both."""
        l, r = bool(_LEFT.search(pre)), bool(_RIGHT.search(pre))
        return ("L" if l else "R") if l != r else None

    _WHICH_IS = re.compile(r",\s*which is (red|green|blue)")

    @classmethod
    def _colour_word(cls, pre: str, full: str):
        """The colour that describes the object we must localise -- and only that one.

        Two suffix patterns look alike but mean opposite things:
          "...cube at <113, 86>, which is red"        -> red describes THE CUBE  (use it)
          "...container at <121,163> that hides the blue cube"
                                                      -> blue describes a DIFFERENT object
                                                         hidden underneath      (ignore it)
        Rewarding blue-looking containers would be nonsense; containers are white. Measured
        as a wash on this data only because no container box matches a cube colour, so the
        distinction is kept explicit rather than relied upon to stay harmless.
        """
        m = re.search(r"\b(red|green|blue)\b", pre)
        if m:
            return m.group(1)
        if full:
            m = cls._WHICH_IS.search(full)
            if m:
                return m.group(1)
        return None

    def _colour_bonus(self, image, cen, pre: str, full: str = None):
        """Reward boxes whose centre patch matches the colour describing the target."""
        want = self._colour_word(pre, full)
        if want is None:
            return np.zeros(len(cen))
        ref = np.array(COLOUR_REF[want])
        im = np.asarray(image)
        out = []
        for cy, cx in cen:
            y0, y1 = int(max(0, cy - 2)), int(min(im.shape[0], cy + 3))
            x0, x1 = int(max(0, cx - 2)), int(min(im.shape[1], cx + 3))
            patch = im[y0:y1, x0:x1].reshape(-1, 3).astype(float).mean(0)
            out.append(np.exp(-(np.linalg.norm(patch - ref) / 60.0) ** 2))
        return np.array(out)

    def select(self, image, pre: str, phrase: str, prior=None, full: str = None):
        """Choose a coordinate using only the image and the phrase.

        Ranking is the detector score plus a colour bonus. When the phrase names a side,
        the extreme box along x among the top-SPATIAL_TOPN is taken instead -- restricted
        to the top few because a spurious edge detection otherwise wins the extreme.
        `prior` is an optional coordinate the caller already believes (a VLM's own guess,
        never the oracle's); when given, the nearest candidate to it is preferred.
        """
        cen, sc = self._boxes(image, PROMPT.get(phrase, phrase))
        if not len(cen):
            return None
        # Colour is often stated AFTER the coordinate -- "pick up the second highlighted cube
        # at <113, 86>, which is red" -- so the colour word is read from the WHOLE subgoal,
        # not just the text preceding the placeholder. 601 of ~3000 frames state it that way,
        # concentrated in exactly the memory-heavy tasks.
        s = sc + COLOUR_W * self._colour_bonus(image, cen, pre, full)
        sd = self._side(pre)
        if sd and SPATIAL_TOPN:
            o = np.argsort(-s)[:SPATIAL_TOPN]
            c2 = cen[o]
            big = (sd == "L") == LEFT_IS_LARGE_X
            j = int(np.argmax(c2[:, 1])) if big else int(np.argmin(c2[:, 1]))
            return c2[j]
        if prior is not None:
            d = np.hypot(cen[:, 1] - prior[1], cen[:, 0] - prior[0])
            j = int(np.argmin(d))
            return cen[j] if d[j] <= SNAP_RADIUS else None
        return cen[int(np.argmax(s))]

    def reset(self) -> None:
        """Forget the cached segment. Call at the start of every episode."""
        self._last_key = None
        self._last_out = None

    def rewrite(self, subgoal: str, image, use_prior: bool) -> str:
        """Replace every `at <y, x>` using select(). Text is preserved exactly.

        The answer is cached per subgoal text, matching how the benchmark fills coordinates.
        """
        if not subgoal or image is None or "<" not in subgoal:
            return subgoal
        key = COORD.sub("<>", subgoal)
        if key == self._last_key and self._last_out is not None:
            self.stats["cached"] += 1
            return self._last_out
        try:
            out, last = [], 0
            for m in COORD.finditer(subgoal):
                self.stats["seen"] += 1
                pre = subgoal[:m.start()]
                ph = phrase_of(pre)
                new = None
                if ph:
                    prior = (int(m.group(1)), int(m.group(2))) if use_prior else None
                    c = self.select(image, pre.lower(), ph, prior, full=subgoal.lower())
                    if c is None:
                        self.stats["no_box"] += 1
                    else:
                        new = (int(round(c[0])), int(round(c[1])))
                        self.stats["snapped"] += 1
                out.append(subgoal[last:m.start()])
                out.append(f"<{new[0]}, {new[1]}>" if new else m.group(0))
                last = m.end()
            out.append(subgoal[last:])
            res = "".join(out)
            self._last_key, self._last_out = key, res
            return res
        except Exception as e:
            self.stats["error"] += 1
            if self.stats["error"] <= 3:
                print(f"[detector_refine] rewrite failed: {e!r}", flush=True)
            return subgoal
