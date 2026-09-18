"""When does a grounded subgoal ask memory for a coordinate?

The released gate keys on three literal strings the composer produces on this benchmark --
"highlight", "correct cube", "correct target". Those strings are a property of one benchmark's
phrasing, not of the method, and a reviewer is right to say so.

The generic gate (opt-in, ``AGENTMEM_GATE=generic``) states the same decision without naming any
task text:

    a subgoal reads memory when the noun phrase carrying its coordinate has a modifier that is
    NOT an appearance or position attribute, and its head noun is a type memory holds an answer
    for.

"Appearance or position attribute" is the vocabulary the module already uses elsewhere: the three
colour words, the ordinals the repetition memory counts, and the side and distance words.
Everything else -- "highlighted", "correct", "same", "marked", "previous" -- is a
memory-determined reference, because nothing in the current frame tells you which one is meant.

The head-noun types are not a list in this file: each caller passes the types its own store can
answer for, derived from what it actually recorded.

Replayed over every subgoal the campaign logged, the two gates read exactly the same subgoals
(campaign/sg/gate_replay.py). This module is therefore a restatement, not a behaviour change, and
it ships behind a flag so that claim can be re-measured live before it becomes the default.
"""
from typing import Iterable, List, Optional, Tuple
import os
import re

COORD = re.compile(r"<\s*\d+\s*,\s*\d+\s*>")
_AT_COORD = re.compile(r"\bat\s*<\s*\d+\s*,\s*\d+\s*>")

#: attributes the current frame can settle on its own
APPEARANCE = {"red", "green", "blue"}
POSITION = {"first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
            "tenth", "left-side", "right-side", "same-side", "near", "far", "left", "right",
            "nearest", "top"}


def mode() -> str:
    """"keyword" (released) or "generic"."""
    return "generic" if os.environ.get("AGENTMEM_GATE", "keyword").lower() == "generic" else "keyword"


def grounded_np(subgoal: str) -> Optional[Tuple[str, List[str]]]:
    """(head noun, modifiers) of the noun phrase carrying the subgoal's first coordinate.

    The phrase is the text between the last determiner before the coordinate and the coordinate
    itself: in "pick up the highlighted cube at <72, 86>" it is "highlighted cube", so the head is
    "cube" and the modifiers are ["highlighted"]. Returns None when the subgoal carries no
    coordinate, which is also the only case in which there is nothing to substitute.
    """
    s = (subgoal or "").lower().strip()
    m = _AT_COORD.search(s)
    if not m:
        return None
    pre = s[:m.start()].rstrip()
    k = pre.rfind("the ")
    np_ = pre[k + 4:] if k >= 0 else pre.split()[-1] if pre.split() else ""
    toks = np_.split()
    if not toks:
        return None
    return toks[-1], toks[:-1]


def memory_determined(subgoal: str, types: Iterable[str]) -> bool:
    """True when the subgoal's grounded phrase names one of ``types`` and distinguishes it by
    something the current frame cannot supply."""
    g = grounded_np(subgoal)
    if g is None:
        return False
    head, mods = g
    if head not in set(types):
        return False
    return any(m not in APPEARANCE and m not in POSITION for m in mods)


def asks(subgoal: str, types: Iterable[str], keyword_test: bool) -> bool:
    """The gate. ``keyword_test`` is the released behaviour's answer for this subgoal; under
    ``AGENTMEM_GATE=generic`` it is ignored and the linguistic rule decides."""
    if not subgoal or not COORD.search(subgoal):
        return False
    return memory_determined(subgoal, types) if mode() == "generic" else bool(keyword_test)
