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


class DemoEventReader:
    """The demonstration of a 'pick up the same block that was previously picked up' task read as
    EVENTS rather than followed as one blob. What the benchmark does (read from its source): the
    arm picks the target and puts it back; then, with the arm still, the simulator slides pairs of
    cubes past each other on two lanes (the first swap always moves the target to its nearest
    neighbour's slot; later swaps move other cubes, possibly the target again); the hard scenes
    are a cluster of fifteen cubes and no swap. Nodes are the slots (every cube blob in the first
    frame, all colours); a slot is vacant when no blob of its colour lies within 6 px; a blob
    farther than 12 px from every slot of its colour is free (a cube in the air or sliding).
    Events: lifted(slot) = during the first free interval that is not a swap, the vacant slot of
    that colour nearest the free blob (the arm occludes a nearer slot on its way, which is what
    'first blob to vanish' falls for); swapped(a, b) = a run of >= 6 frames with two slots vacant
    together and the sliding cubes visible on at least half of it; a swap exchanges the two, so no
    tracking through the crossing is needed. If the lift stayed hidden behind the gripper, the
    target is the member of the first swap's pair whose slot was vacant longer before it.
    Offline on 21 dumped demonstrations against the oracle: 21/21 within 8 px (the followed-blob
    rule: 14/21)."""

    NEAR, FAR, MIN_PAIR = 6.0, 12.0, 6

    def __init__(self, stride: int = 2, tag: str = "[agentmem]"):
        self.stride = stride; self.tag = tag; self.events: List[str] = []

    def read(self, frames) -> Optional[Tuple[str, float, float, str]]:
        if frames is None or len(frames) < 4:
            return None
        fr = [frames[i] for i in range(0, len(frames), max(1, self.stride))]; T = len(fr)
        slots = [(c, y, x) for c in COLOURS for y, x, a in colour_blobs(fr[0], c)]
        if not slots:
            return None
        vac = np.zeros((T, len(slots)), bool); free: List[list] = [[] for _ in range(T)]
        for t, f in enumerate(fr):
            B = {c: colour_blobs(f, c) for c in COLOURS}
            for i, (c, sy, sx) in enumerate(slots):
                vac[t, i] = not any(np.hypot(y - sy, x - sx) <= self.NEAR for y, x, a in B[c])
            for c in COLOURS:
                S = [(y, x) for cc, y, x in slots if cc == c]
                for y, x, a in B[c]:
                    if S and min(np.hypot(y - sy, x - sx) for sy, sx in S) > self.FAR:
                        free[t].append((c, y, x))
        hasfree = np.array([bool(v) for v in free])
        two = vac.sum(1) >= 2; swaps = []; s = None
        for t in range(T + 1):
            v = bool(two[t]) if t < T else False
            if v and s is None:
                s = t
            if not v and s is not None:
                if t - s >= self.MIN_PAIR and hasfree[s:t].mean() >= 0.5:
                    cnt = vac[s:t].sum(0); a, b = np.argsort(-cnt)[:2].tolist()
                    if cnt[a] >= 0.6 * (t - s) and cnt[b] >= 0.6 * (t - s):
                        swaps.append((s, t, (a, b)))
                s = None
        inswap = np.zeros(T, bool)
        for s0, e0, _ in swaps:
            inswap[s0:e0] = True
        t0 = next((t for t in range(T - 1) if hasfree[t] and hasfree[t + 1] and not inswap[t] and not inswap[t + 1]), None)
        if t0 is not None:
            t1 = t0
            while t1 + 1 < T and hasfree[t1 + 1] and not inswap[t1 + 1]:
                t1 += 1
            c0, fy, fx = free[t0][0]
            cands = [i for i, (c, y, x) in enumerate(slots) if c == c0 and vac[t0:t1 + 1, i].mean() >= 0.5] or \
                    [i for i, (c, y, x) in enumerate(slots) if c == c0]
            pick = min(cands, key=lambda i: np.hypot(slots[i][1] - fy, slots[i][2] - fx))
            how = f"lifted {slots[pick][0]} cube seen in frames {t0 * self.stride}-{t1 * self.stride}"
        elif swaps:
            s0, e0, (a, b) = swaps[0]
            pick = max((a, b), key=lambda i: int(vac[:s0, i].sum()))
            how = "lift hidden; the first swap's pair, longer vacancy before it"
        else:
            pick = int(np.argmax(vac.sum(0))); how = "lift hidden, no swap; longest vacancy"
        cur = pick
        self.events = [f"lifted slot {pick} {tuple(int(v) for v in slots[pick][1:])}"]
        for s0, e0, (a, b) in swaps:
            self.events.append(f"swapped slots {a}<->{b} at frames {s0 * self.stride}-{e0 * self.stride}")
            cur = b if cur == a else a if cur == b else cur
        c, y, x = slots[cur]
        return c, float(y), float(x), how


class PlaceEventReader:
    """The demonstration of 'place the cube on the target right before / after the button was
    pressed' read as events: placed(target) from the cube's dwells, pressed from the arm dwelling
    on the button. Nodes: the targets and the button (bright unsaturated blobs of the first frame;
    the button is the large one, a target renders as a disc and a ring) and the cube named in the
    goal. placed(k): the cube's track rests >= 10 frames within 3 px on a target (the demonstration
    ends by putting the cube down off every target, which is not a placement). pressed: grey arm
    pixels within 14 px of the button exceed the button's own base by > 60 for >= 6 frames (the
    button itself stays visible under the gripper, so occlusion is no cue). The answer is the last
    placement begun before the press ('before') or the first begun after it ('after'); with no
    press seen, the first / second placement (the benchmark's demonstration is pick, place, press,
    pick, place, put down). Abstains unless the answer lies on a target. Offline on 8 dumped
    demonstrations: the press found once per demonstration, every answer on a target."""

    def __init__(self, stride: int = 2, tag: str = "[agentmem]"):
        self.stride = stride; self.tag = tag; self.events: List[str] = []

    @staticmethod
    def _arm(im):
        a = np.asarray(im)[..., :3].astype(int); r, g, b = a[..., 0], a[..., 1], a[..., 2]
        mx = np.maximum(np.maximum(r, g), b); mn = np.minimum(np.minimum(r, g), b)
        return (mn > 60) & (mx < 200) & ((mx - mn) < 20)

    def read(self, frames, goal: str) -> Optional[Tuple[float, float, str]]:
        goal = (goal or "").lower()
        word = "before" if "before" in goal else "after" if "after" in goal else None
        colour = next((c for c in COLOURS if c in goal), None)
        if frames is None or len(frames) < 4 or word is None or colour is None:
            return None
        fr = [frames[i] for i in range(0, len(frames), max(1, self.stride))]
        W = white_blobs(fr[0])
        if len(W) < 2:
            return None
        by, bx, _, ba = max(W, key=lambda w: w[3])
        targets: List[Tuple[float, float]] = []
        for y, x, rad, a in W:
            if np.hypot(y - by, x - bx) > 6 and not any(np.hypot(y - ty, x - tx) < 4 for ty, tx in targets):
                targets.append((y, x))
        # the cube's track and its dwells
        track = []; cur = None
        for im in fr:
            B = colour_blobs(im, colour)
            if not B:
                track.append(None); continue
            if cur is None:
                big = max(B, key=lambda b: b[2]); cur = (big[0], big[1])
            else:
                dd = [np.hypot(y - cur[0], x - cur[1]) for y, x, a in B]; j = int(np.argmin(dd))
                if dd[j] > 30:
                    track.append(None); continue
                cur = (B[j][0], B[j][1])
            track.append(cur)
        dwells = []; run = []
        for t, q in enumerate(track + [None]):
            if q is not None and run and np.hypot(q[0] - run[0][1][0], q[1] - run[0][1][1]) < 3:
                run.append((t, q)); continue
            if len(run) >= 10:
                c = (float(np.mean([p[1][0] for p in run])), float(np.mean([p[1][1] for p in run])))
                if not dwells or np.hypot(c[0] - dwells[-1][1][0], c[1] - dwells[-1][1][1]) > 10:
                    dwells.append((run[0][0], c))
            run = [(t, q)] if q is not None else []
        places = [(t, c) for t, c in dwells[1:] if any(np.hypot(c[0] - ty, c[1] - tx) <= 8 for ty, tx in targets)]
        # the press: the arm dwelling on the button
        H, Wd = np.asarray(fr[0]).shape[:2]; yy, xx = np.mgrid[0:H, 0:Wd]; near = np.hypot(yy - by, xx - bx) <= 14
        base = int((self._arm(fr[0]) & near).sum())
        press = None; s = None
        for t in range(len(fr) + 1):
            hot = t < len(fr) and int((self._arm(fr[t]) & near).sum()) - base > 60
            if hot and s is None:
                s = t
            if not hot and s is not None:
                if t - s >= 6 and press is None:
                    press = s
                s = None
        self.events = [f"placed on target at {tuple(int(v) for v in c)} from frame {t * self.stride}" for t, c in places]
        if press is not None:
            self.events.append(f"button pressed at frame {press * self.stride}")
            self.events.sort(key=lambda e: int(e.rsplit(" ", 1)[1]))
            before = [c for t, c in places if t < press]; after = [c for t, c in places if t > press]
            ans = (before[-1] if before else None) if word == "before" else (after[0] if after else None)
            how = f"{word} the press at frame {press * self.stride}"
        else:
            ans = (places[0][1] if places else None) if word == "before" else (places[1][1] if len(places) > 1 else None)
            how = f"no press seen; the {'first' if word == 'before' else 'second'} placement"
        if ans is None:
            return None
        return float(ans[0]), float(ans[1]), how
