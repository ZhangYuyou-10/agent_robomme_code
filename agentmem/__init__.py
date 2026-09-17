"""Agentic memory for grounded-subgoal prediction on RoboMME.

The method is one VLM agent, driven only by the task prompt and the current image, deciding four
things: which objects to watch, whether their evidence will still be visible later, what to
record, and how to compose the subgoal. `AgentMemory` holds the memory and its read/write policy;
`AgentMemorySubgoalPredictor` is the predictor the benchmark calls.

These modules import the benchmark's own `env_runner` / `utils` and the Qwen-VL wrapper, so they
run from inside a RoboMME checkout rather than standing alone.

Placement (the benchmark's `examples/robomme/` directory):

    subgoal_predictor.py       -> examples/robomme/subgoal_predictor.py
    agent_memory.py            -> examples/robomme/subgoal_prediction/agent_memory.py
    detector_refine.py         -> examples/robomme/subgoal_prediction/detector_refine.py
    plan_parse.py              -> examples/robomme/subgoal_prediction/plan_parse.py
    progress_state.py          -> examples/robomme/subgoal_prediction/progress_state.py

`build_subgoal_predictor` dispatches on `use_agentmem`; the reported configuration is

    WRITEMEM_TH=0.30 AGENTMEM_KW=1 AGENTMEM_STALE=150 AGENTMEM_CYCLE=0 AGENTMEM_DEMO=1
    AGENTMEM_REVERT=1 AGENTMEM_REFINE=verbs AGENTMEM_PATH=1 AGENTMEM_ORD=1 AGENTMEM_TRACK=1
    AGENTMEM_VIDEO2=1

with the eval arguments `--subgoal_type=grounded_subgoal --use_agentmem --use_qwenvl
--model_seed=7 --model_ckpt_id=79999 --max_steps=1300 --episodes_per_task=50`. Two post-hoc
guards, `AGENTMEM_VERBS=contact` and `AGENTMEM_LOCGUARD=1`, are opt-in and OFF: both are measured
nulls and neither is in the reported table. Everything else defaults to the reported behaviour.
"""
