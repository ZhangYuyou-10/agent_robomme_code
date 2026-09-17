"""Parse a step plan from the task goal.

Measured on real episodes: the goal states the plan for the seven tasks that carry nearly all
of the semantic error ("first press both buttons, then pick up the container hiding the red
cube"; "put two red cubes and three green cubes into the bin, then press the button"). The eight
"watch the video carefully" tasks do not -- their plan content is in the demo -- but those
already run at 92.6-100% subgoal accuracy, so the plan is available exactly where it is needed.

This parser is a HEADROOM INSTRUMENT. The method will have an LLM parse the goal, which
generalises; this deterministic version exists to bound what perfect plan-parsing is worth
before that is built. It is deliberately small and it is allowed to fail -- `parse` returns
None when the goal does not state a plan, and that case is reported rather than patched.
"""
import re
from typing import List, Optional, NamedTuple

WORD_NUM = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,
            "nine":9,"ten":10,"once":1,"twice":2}
ORD_NUM  = {"first":1,"second":2,"third":3,"fourth":4,"fifth":5,"sixth":6}


class Step(NamedTuple):
    action: str          # press | pick | place | move
    obj: str             # noun phrase as stated in the goal
    repeats: int         # how many times the GROUP this step belongs to repeats
    group: int = 0       # steps sharing a group id repeat together, interleaved


def expand(plan: List["Step"]) -> List[str]:
    """Flatten a plan to the action sequence it implies.

    Steps in the same group repeat TOGETHER: "pick up the blue cube and place it on the target,
    repeating this action five times" is (pick, place) x5, not pick x5 then place x5. Expanding
    per-step instead of per-group was worth 40+ points of agreement on BinFill and PickXtimes.
    """
    out: List[str] = []
    i = 0
    while i < len(plan):
        g = plan[i].group
        block = [s for s in plan[i:] if s.group == g]
        i += len(block)
        n = max((s.repeats for s in block), default=1) or 1
        for _ in range(max(n, 1)):
            out += [s.action for s in block]
    return out


def _num(tok: str) -> Optional[int]:
    tok = tok.strip().lower()
    if tok.isdigit():
        return int(tok)
    return WORD_NUM.get(tok) or ORD_NUM.get(tok)


def parse(goal: str) -> Optional[List[Step]]:
    """Return the stated step plan, or None if the goal does not state one."""
    g = (goal or "").lower().strip()
    if not g or "watch the video" in g:
        return None                      # plan content lives in the demo, not the text

    steps: List[Step] = []

    # "first press the button" / "first press both buttons"
    m = re.search(r"press (both|the|all) (buttons?|button)", g)
    if m and g.index(m.group(0)) < len(g) // 2:
        steps.append(Step("press", "button", 2 if m.group(1) == "both" else 1, len(steps)))

    # "put two red cubes and three green cubes into the bin"
    fills = re.findall(r"(\w+) (red|green|blue) cubes?", g)
    if "into the bin" in g and fills:
        for cnt, col in fills:
            n = _num(cnt) or 1
            gid = len(steps)                    # one group per colour: (pick, place) x n
            steps.append(Step("pick", f"{col} cube", n, gid))
            steps.append(Step("place", "bin", n, gid))

    # "pick up the blue cube and place it on the target, repeating this action five times"
    m = re.search(r"pick up the (red|green|blue) cube.*?place it on the target", g)
    if m:
        r = re.search(r"repeating this action (\w+) times", g)
        n = (_num(r.group(1)) if r else 1) or 1
        gid = len(steps)                        # (pick, place) x n, not pick x n then place x n
        steps.append(Step("pick", f"{m.group(1)} cube", n, gid))
        steps.append(Step("place", "target", n, gid))

    # "move it to the top of the right-side target, then ... left-side ... repeating ... three times"
    if "right-side target" in g and "left-side target" in g:
        r = re.search(r"repeating this back and forth motion (\w+) times", g)
        n = (_num(r.group(1)) if r else 1) or 1
        c = re.search(r"pick up the (red|green|blue) cube", g)
        if c:
            steps.append(Step("pick", f"{c.group(1)} cube", 1, len(steps)))
        gid = len(steps)                        # (right, left) x n
        steps.append(Step("move", "right-side target", n, gid))
        steps.append(Step("move", "left-side target", n, gid))

    # "pick up all cubes that have been highlighted"
    if "highlight" in g:
        gid = len(steps)                        # (pick, place) repeated an unknown number of times
        steps.append(Step("pick", "highlighted cube", 0, gid))
        steps.append(Step("place", "table", 0, gid))

    # "pick up the container hiding the red cube [, finally pick up another container ...]"
    for col in re.findall(r"hiding the (red|green|blue) cube", g):
        steps.append(Step("pick", f"container hiding the {col} cube", 1, len(steps)))

    # terminal "then press the button to stop"
    if re.search(r"press the button( to (stop|finish))?$", g) or "then press the button" in g:
        if not steps or steps[-1].action != "press":
            steps.append(Step("press", "button", 1, len(steps)))

    return steps or None


def expand_steps(plan: List["Step"]) -> List[str]:
    """Flatten to readable step descriptions, group-interleaved and repetition-labelled.

    Two failures this avoids, both hit in the first position experiment:
      - per-step expansion gives "pick, pick, place, place" when the robot does
        "pick, place, pick, place" -- the model was asked to locate itself in a plan that did
        not match the rollout.
      - identical adjacent options ("1. pick red cube / 2. pick red cube") cannot be told apart
        from a single frame, so the question was partly unanswerable. Labelling the repetition
        makes each option distinguishable.
    """
    out: List[str] = []
    i = 0
    while i < len(plan):
        g = plan[i].group
        block = [s for s in plan[i:] if s.group == g]
        i += len(block)
        n = max((s.repeats for s in block), default=1) or 1
        for rep in range(max(n, 1)):
            for s in block:
                label = f"{s.action} {s.obj}"
                if n > 1:
                    label += f"  ({rep+1} of {n})"
                out.append(label)
    return out
