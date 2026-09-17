"""A small scene-graph tool for the agentic memory: objects as nodes, pixel-verified relations as
edges, and the events that relations produce over time. Opt-in (AGENTMEM_SG=1); the released
method never imports it unless asked.

Why a graph: the detector grounds NOUNS ("blue cube") but not RELATIONS ("the cube that sits on a
white area", "the cube inside the bin"), and several RoboMME references are relations that hold
only briefly. PickHighlight is the clean case: after the button press a white disc appears under
each cube to be picked and vanishes ~20 steps later, long before the composer is asked; the oracle
asks for "the highlighted cube ... which is blue". The relation on(cube, white disc), read while
the disc is visible, identifies those cubes (offline on the 50 published episodes: 52/56 oracle
targets within 8 px, median 2.1 px). The graph records the relation as an event and answers the
later reference from it.

Everything here is pixel arithmetic on the 256x256 front camera; no model call.
"""
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

COORD = re.compile(r"<\s*\d+\s*,\s*\d+\s*>")
COLOURS = ("red", "green", "blue")


def colour_blobs(im, colour: str, min_area: int = 12) -> List[Tuple[float, float, int]]:
    """Centres and areas of saturated blobs of one cube colour (same thresholds as agent_memory)."""
    from scipy import ndimage
    a = np.asarray(im)[..., :3].astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    m = {"red": (r > 140) & (g < 60) & (b < 60), "green": (g > 120) & (r < 80) & (b < 80),
         "blue": (b > 120) & (r < 80) & (g < 80)}[colour]
    lab, n = ndimage.label(m)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(ys) >= min_area and ys.mean() > 30:              # not the wall / robot band
            out.append((float(ys.mean()), float(xs.mean()), int(len(ys))))
    return out


def white_blobs(im, min_area: int = 20, max_area: int = 700, top: int = 28):
    """Bright, unsaturated blobs on the table: highlight discs, the button, parts of the arm."""
    from scipy import ndimage
    a = np.asarray(im)[..., :3].astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = np.maximum(np.maximum(r, g), b); mn = np.minimum(np.minimum(r, g), b)
    m = (mn > 175) & ((mx - mn) < 30)
    lab, n = ndimage.label(m)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        h = ys.max() - ys.min() + 1; w = xs.max() - xs.min() + 1; area = int(len(ys))
        if area < min_area or area > max_area or ys.min() < top or h > 40 or w > 40:
            continue
        out.append((float(ys.mean()), float(xs.mean()), max(h, w) / 2.0, area))
    return out


class HighlightGraph:
    """Nodes: cubes (colour, position) and white discs. Edge on(cube, disc) while the cube's centre
    lies within the disc. An edge that holds on two consecutive frames is recorded as the event
    'cube highlighted' with the cube's colour and position; the record is the memory's answer to
    "the highlighted cube" for the rest of the episode.

    The disc is read as white pixels in an annulus around each cube's centre (robust to two discs
    merging into one blob and to the cube covering the disc's centre). Static white blobs seen in
    the first live frame (the button) are never discs; edges are accepted only within `window`
    frames of the first one so that the arm's white parts, which arrive later, are not."""

    def __init__(self, window: int = 40, tag: str = "[agentmem]"):
        self.tag = tag
        self.window = window
        self.static: List[Tuple[float, float]] = []
        self.first_disc: Optional[int] = None
        self.k = -1
        self.pending: Dict[Tuple[str, int, int], Tuple[float, float, int]] = {}
        self.highlighted: List[dict] = []          # {colour, y, x, frame, picked}
        self._seen_first = False

    RING = (4.0, 16.0)   # annulus around a cube centre in which a highlight disc shows
    RING_MIN = 25        # white pixels in the annulus to call the cube highlighted

    def _white_mask(self, frame):
        """Bright unsaturated pixels, with the static blobs (the button), the arm (large or touching
        the top) and anything far too big to be a disc removed."""
        from scipy import ndimage
        a = np.asarray(frame)[..., :3].astype(int)
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        mx = np.maximum(np.maximum(r, g), b); mn = np.minimum(np.minimum(r, g), b)
        m = (mn > 175) & ((mx - mn) < 30)
        lab, n = ndimage.label(m)
        keep = np.zeros_like(m)
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            area = len(ys); h = ys.max() - ys.min() + 1; w = xs.max() - xs.min() + 1
            if area < 15 or area > 1200 or ys.min() < 28 or h > 70 or w > 70:
                continue
            cy, cx = ys.mean(), xs.mean()
            if any(np.hypot(cy - sy, cx - sx) < 14 for sy, sx in self.static):
                continue
            keep[ys, xs] = True
        return keep

    def observe(self, frame) -> None:
        self.k += 1
        if frame is None:
            return
        if not self._seen_first:
            self._seen_first = True
            self.static = [(y, x) for y, x, _, _ in white_blobs(frame)]
            return
        if self.first_disc is not None and self.k - self.first_disc > self.window:
            return
        white = self._white_mask(frame)
        if not white.any():
            self.pending = {}
            return
        H, W = white.shape
        yy, xx = np.mgrid[0:H, 0:W]
        now = {}
        for colour in COLOURS:
            for cy, cx, _ in colour_blobs(frame, colour):
                d = np.hypot(yy - cy, xx - cx)
                ring = (d >= self.RING[0]) & (d <= self.RING[1])
                if int((white & ring).sum()) >= self.RING_MIN:
                    now[(colour, int(round(cy / 6)), int(round(cx / 6)))] = (cy, cx, self.k)
        if now and self.first_disc is None:
            self.first_disc = self.k
        for key, (cy, cx, k) in now.items():
            if key in self.pending and not self._known(key[0], cy, cx):
                self.highlighted.append({"colour": key[0], "y": cy, "x": cx, "frame": k, "picked": False})
                print(f"{self.tag} graph: {key[0]} cube on a white disc at ({int(cy)}, {int(cx)}), frame {k}",
                      flush=True)
        self.pending = now

    def _known(self, colour, cy, cx) -> bool:
        return any(h["colour"] == colour and np.hypot(h["y"] - cy, h["x"] - cx) < 10 for h in self.highlighted)

    GONE_CHECKS = 5      # consecutive checks (every 3rd step) without the colour at the note: a real lift,
                         # not the gripper passing over it

    def mark_picked(self, frame) -> None:
        """A highlighted cube whose colour has left its position for several consecutive checks has
        been picked up; a cube that reappears is un-marked (the gripper only passed over it)."""
        if frame is None:
            return
        for h in self.highlighted:
            blobs = colour_blobs(frame, h["colour"])
            present = any(np.hypot(y - h["y"], x - h["x"]) < 8 for y, x, _ in blobs)
            if present:
                if h["picked"] and h.get("gone", 0) < 20:
                    h["picked"] = False
                h["gone"] = 0
                continue
            h["gone"] = h.get("gone", 0) + 1
            if not h["picked"] and h["gone"] >= self.GONE_CHECKS:
                h["picked"] = True
                print(f"{self.tag} graph: highlighted {h['colour']} cube gone from ({int(h['y'])}, {int(h['x'])})",
                      flush=True)

    def fill(self, subgoal: str) -> Optional[str]:
        """Answer a 'highlighted cube' reference from the recorded events: the not-yet-picked
        highlighted cube of the colour the composer names, else any not-yet-picked one. The
        coordinate is replaced; if the composer named a colour that was never highlighted, the
        colour word is corrected too, because the pixels verified the recorded cube's colour."""
        if not subgoal or "highlight" not in subgoal.lower() or not self.highlighted:
            return None
        left = [h for h in self.highlighted if not h["picked"]]
        if not left:
            return None
        named = next((c for c in COLOURS if re.search(rf"\b{c}\b", subgoal.lower())), None)
        cands = [h for h in left if h["colour"] == named] or left
        h = cands[0]
        out = COORD.sub(f"<{int(round(h['y']))}, {int(round(h['x']))}>", subgoal, count=1)
        if named and named != h["colour"]:
            out = re.sub(rf"\b{named}\b", h["colour"], out)
        return out

    @property
    def stats(self) -> dict:
        return {"highlighted": [(h["colour"], int(h["y"]), int(h["x"]), h["picked"]) for h in self.highlighted]}
