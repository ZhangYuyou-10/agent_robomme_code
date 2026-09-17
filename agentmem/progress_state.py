"""Progress state: what has happened so far, independent of what the agent believes.

The detector campaign showed grounding is recoverable (78.8% given perfect instance selection)
while the best deployable arm sits at 41.0%. The residual is semantics, and it decomposes as
41% wrong-ordinal, 31% "ahead" (skipping a prerequisite), 21% "behind" (repeating a done step)
across the ~30% of steps where QwenVL names a different subgoal than the oracle.

All three are failures of PROGRESS belief, not of perception. This module holds the progress
state separately so it can be maintained by whatever mechanism we choose -- an oracle stream
(as a headroom instrument), deterministic events, or an agent -- behind one interface.

Nothing here is task-specific: steps are (action, object, ordinal) triples parsed from subgoal
text, and the counters are generic over whatever actions and objects appear.
"""
import re
from typing import Dict, List, Optional, Tuple

ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
            "ninth", "tenth"]
ORD_RE = re.compile(r"\b(" + "|".join(ORDINALS) + r")\b")
COORD_RE = re.compile(r"<(\d+),\s*(\d+)>")
# The action a subgoal performs. Generic verbs, not per-task phrasing.
ACTION_RE = [
    ("press",   re.compile(r"\bpress\b")),
    ("pick",    re.compile(r"\bpick up\b|\bgrasp\b")),
    ("place",   re.compile(r"\bplace\b|\bput\b|\bdrop\b|\binsert\b")),
    ("move",    re.compile(r"\bmove\b|\bpush\b|\bhook\b")),
    ("static",  re.compile(r"\bstatic\b|\bremain\b")),
]


def canon(subgoal: str) -> str:
    """Subgoal text with coordinates and ordinals removed -- the step's identity."""
    s = COORD_RE.sub("<>", (subgoal or "").lower())
    return ORD_RE.sub("<n>", s).strip()


def action_of(subgoal: str) -> Optional[str]:
    s = (subgoal or "").lower()
    for name, rx in ACTION_RE:
        if rx.search(s):
            return name
    return None


def ordinal_of(subgoal: str) -> Optional[int]:
    m = ORD_RE.search((subgoal or "").lower())
    return ORDINALS.index(m.group(1)) + 1 if m else None


class ProgressState:
    """Counts completed steps by their canonical identity.

    `observe` is fed one authoritative subgoal per step -- from the oracle when this is used as
    a headroom instrument, or from an event detector / agent otherwise. A step is counted as
    completed when the authoritative subgoal CHANGES away from it, which is the same rule the
    benchmark itself uses to advance (`segmentation_utils.py:63` refills only on change).
    """

    def __init__(self):
        self.completed: Dict[str, int] = {}    # canonical step -> times completed
        self.actions_done: Dict[str, int] = {} # action -> times completed
        self.history: List[str] = []
        self._current: Optional[str] = None    # coordinate-stripped, ORDINAL PRESERVED

    def observe(self, subgoal: Optional[str]) -> None:
        """Advance on a change in the ordinal-PRESERVING text, count against the stripped key.

        Transitions must be detected on text that keeps the ordinal: "press the first button"
        and "press the second button" are different steps, and collapsing them (as the canonical
        key does) makes the progression invisible -- the counter then rewrites a correct
        "second" back to "first". Counting still uses the stripped key, so repetitions of the
        same step accumulate.
        """
        if not subgoal:
            return
        raw = COORD_RE.sub("<>", subgoal.lower()).strip()
        if raw == self._current:
            return
        if self._current is not None:
            key = canon(self._current)
            self.completed[key] = self.completed.get(key, 0) + 1
            a = action_of(self._current)
            if a:
                self.actions_done[a] = self.actions_done.get(a, 0) + 1
            self.history.append(self._current)
        self._current = raw

    def times_completed(self, subgoal: str) -> int:
        return self.completed.get(canon(subgoal), 0)

    def expected_ordinal(self, subgoal: str) -> int:
        """Which repetition of this step is next, 1-based."""
        return self.times_completed(subgoal) + 1

    def summary(self) -> str:
        """One line the agent can read. Kept short; long histories crowd the VLM prompt."""
        if not self.history:
            return "nothing completed yet"
        parts = []
        for k, v in list(self.completed.items())[-6:]:
            parts.append(f"{k} x{v}" if v > 1 else k)
        return "completed so far: " + "; ".join(parts)


def correct_ordinal(subgoal: str, state: ProgressState) -> str:
    """Rewrite the ordinal in `subgoal` to the repetition the state says comes next.

    Only the ordinal word changes; the action, object and coordinates are untouched. This is the
    narrow, progress-only correction -- it cannot smuggle in object identity.
    """
    if not subgoal or not ORD_RE.search(subgoal.lower()):
        return subgoal
    want = state.expected_ordinal(subgoal)
    if not 1 <= want <= len(ORDINALS):
        return subgoal
    repl = ORDINALS[want - 1]

    def _sub(m):
        w = m.group(1)
        return repl.upper() if w.isupper() else (repl.capitalize() if w[0].isupper() else repl)

    return ORD_RE.sub(_sub, subgoal, count=1)


def current_action(state: "ProgressState") -> Optional[str]:
    """The action of the step the state believes is in progress."""
    return action_of(state._current) if state._current else None


def out_of_sync(pred: str, state: "ProgressState") -> bool:
    """Does the prediction name a different step than the one actually in progress?

    An earlier version ranked actions with a hand-made ACTION_ORDER table to tell "ahead" from
    "behind". That table is wrong in general -- ButtonUnmask presses its button BEFORE the
    container pickup, so a global verb ordering mis-classified the very case it was meant to
    catch -- and a per-task ordering is exactly the benchmark-tuned heuristic this method must
    not contain. Disagreement with the step in progress needs no ordering at all.
    """
    if not pred or not state._current:
        return False
    return canon(pred) != canon(state._current)


def correct_phase(subgoal: str, state: "ProgressState", template: Optional[str] = None) -> str:
    """Fall back to the step actually in progress when the prediction names a different one.

    ButtonUnmask's mismatches are entirely of this kind -- the agent proposes picking up the
    container before the button has been pressed, so the episode stalls and never reaches the
    container. The coordinate is left to the detector, so no object identity comes from the state.
    """
    if not subgoal or not out_of_sync(subgoal, state):
        return subgoal
    return template if template else (state._current or subgoal)
