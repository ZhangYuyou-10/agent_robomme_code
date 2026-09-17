"""Agentic memory for grounded-subgoal prediction on RoboMME.

The method is one VLM agent, driven only by the task prompt and the current image, deciding four
things: which objects to watch, whether their evidence will still be visible later, what to
record, and how to compose the subgoal. `AgentMemory` holds the memory and its read/write policy;
`AgentMemorySubgoalPredictor` is the predictor the benchmark calls.

These modules import the benchmark's own `env_runner` / `utils` and the Qwen-VL wrapper, so they
run from inside a RoboMME checkout rather than standing alone. See the repository README.
"""
