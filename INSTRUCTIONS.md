# Running the agentic memory on RoboMME

Everything below was used to produce the reported numbers (60.50% and 61.25% average task success
over 16 tasks × 50 episodes, against 32.70% for the same policy without memory). It assumes you can
run the upstream RoboMME evaluation already; the method is five Python files that drop into that
codebase plus one small patch to its evaluation flags.

## 0. What you need

| item | where |
|---|---|
| RoboMME benchmark (environments, episode metadata, demonstration videos) | https://github.com/RoboMME/robomme_benchmark |
| RoboMME policy-learning codebase (policy server, `examples/robomme` evaluation harness) | https://github.com/RoboMME/robomme_policy_learning — we are on commit `ecf086c` (2026-04-08); `third_party/robomme_benchmark` is its submodule |
| GroundSG policy checkpoint (π0.5, symbolic grounded subgoals) | `runs/ckpts/symbolic-grounded-subgoal/79999` — upstream's released checkpoint |
| Grounded-subgoal composer (Qwen3-VL-4B LoRA) | `runs/ckpts/vlm_subgoal_predictor/qwenvl/grounded_subgoal/checkpoint-1200` — upstream's released adapter |
| Base VLM and detector | `Qwen/Qwen3-VL-4B-Instruct`, `IDEA-Research/grounding-dino-base` from the Hugging Face hub (downloaded on first use; set `HF_HOME`) |

Two Python environments, as upstream: the policy server runs in the repo's own `uv` environment
(JAX / openpi); the evaluator runs in a conda-style environment. Ours (`robomme`) is Python 3.11.16,
torch 2.9.1+cu128, torchvision 0.24.1, transformers 4.57.3, ms-swift 3.11.1, PyAV 18.1.0,
scipy 1.17.1, numpy 1.26.4, imageio 2.37.4, qwen-vl-utils. The method adds no dependency beyond
upstream's: it needs `swift` (the composer's engine), `transformers` (Grounding DINO) and `scipy`
(`ndimage.label` for the pixel cue readers).

GPU budget: a policy server takes 36–43 GB; each evaluator ~11 GB (composer + detector + simulator).
On a 144 GB GPU we ran two servers plus at most **five** evaluators; a sixth crashes a running
evaluator inside SAPIEN's renderer with `RuntimeError: Resource temporarily unavailable`.

## 1. Install the method

Inside your `robomme_policy_learning` checkout:

```
# 1. the five method files (subgoal_predictor.py REPLACES upstream's file: it contains upstream's
#    predictors unchanged plus ours, built on commit ecf086c)
cp agentmem/subgoal_predictor.py   examples/robomme/subgoal_predictor.py
cp agentmem/agent_memory.py        examples/robomme/subgoal_prediction/agent_memory.py
cp agentmem/detector_refine.py     examples/robomme/subgoal_prediction/detector_refine.py
cp agentmem/plan_parse.py          examples/robomme/subgoal_prediction/plan_parse.py
cp agentmem/progress_state.py      examples/robomme/subgoal_prediction/progress_state.py
cp agentmem/scene_graph.py         examples/robomme/subgoal_prediction/scene_graph.py   # optional reader, AGENTMEM_SG=1

# 2. the evaluation harness we launch with (episode ranges / lists on top of upstream eval.py)
cp run_subset_eval.py examples/robomme/run_subset_eval.py

# 3. the flags upstream's eval.py needs (five booleans in Args, plus QWEN_BASE / QWEN_ATTN
#    environment overrides in subgoal_prediction/qwenvl/api.py, an optional render-backend
#    override and a relaxed obs-horizon assert; all additive)
git apply --check patches/robomme_policy_learning_examples.patch && git apply patches/robomme_policy_learning_examples.patch
```

`examples/robomme/subgoal_prediction/__init__.py` must exist (upstream ships an empty
`__ini__.py` — a typo; create `__init__.py` if imports of `subgoal_prediction.*` fail).

**If you will run more than two evaluators on one machine**, patch torchvision's video reader
(the composer reads the demonstration video through `qwen_vl_utils` → `torchvision.io.read_video`;
upstream torchvision converts every decoded frame with a fresh swscale context whose thread pool
is sized to all CPU cores, and several concurrent readers exhaust the process/thread limit, which
shows up as `libav.swscaler: Failed initializing scaling graph (Resource temporarily unavailable)`
followed by `KeyError: 'video_fps'` and an `"error"` outcome). In
`site-packages/torchvision/io/video.py`, function `read_video`, replace

```python
        vframes_list = [frame.to_rgb().to_ndarray() for frame in video_frames]
```
with
```python
        from av.video.reformatter import VideoReformatter as _VR
        _rf = _VR()
        vframes_list = [_rf.reformat(frame, format="rgb24").to_ndarray() for frame in video_frames]
```
The frames are bit-identical (we checked with `torch.equal` on a 1,077-frame demonstration); only
the thread churn goes away.

## 2. Start a policy server

From the repo root, one server per port (we used 8301–8304, two per GPU):

```
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
  uv run scripts/serve_policy.py --seed=7 --port=8301 policy:checkpoint \
  --policy.dir=runs/ckpts/symbolic-grounded-subgoal/79999 --policy.config=mme_vla_suite
```

Wait for `server listening on 0.0.0.0:8301` in its output (about a minute). `--seed` is the
policy's sampling seed; it is the only seed in the pipeline (episode initial states come from the
benchmark's per-episode metadata). Note that a server serves many evaluators concurrently, so two
runs with the same seed still differ: identical configurations flip on 11–14% of episodes. Compare
runs on matched episodes and treat fewer than three discordant episodes as noise.

## 3. Run the method

From `examples/robomme`, with the environment that runs the evaluator:

```
export HF_HOME=/path/to/hf_cache            # Qwen3-VL-4B-Instruct and grounding-dino-base land here
export QWEN_ATTN=sdpa                       # flash-attention is not required
export TMPDIR=/path/with/space              # the memory writes small scratch images here
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4  # with several evaluators per box, uncapped BLAS threads crash

WRITEMEM_TH=0.30 \
AGENTMEM_KW=1 AGENTMEM_STALE=150 AGENTMEM_CYCLE=0 AGENTMEM_DEMO=1 AGENTMEM_REVERT=1 \
AGENTMEM_REFINE=verbs AGENTMEM_PATH=1 AGENTMEM_ORD=1 AGENTMEM_TRACK=1 AGENTMEM_VIDEO2=1 \
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python run_subset_eval.py \
  --args.model_seed=7 --args.port=8301 \
  --args.policy_name=agentmem_seed7 --args.model_ckpt_id=79999 \
  --args.subgoal_type=grounded_subgoal --args.use_agentmem --args.use_qwenvl \
  --args.qwenvl_groundSG_adapter_path=runs/ckpts/vlm_subgoal_predictor/qwenvl/grounded_subgoal/checkpoint-1200 \
  --args.only_tasks=ButtonUnmask,VideoUnmaskSwap --args.episodes_per_task=50 --args.max_steps=1300
```

The environment variables above are the reported configuration; set nothing else. `only_tasks`
takes any subset of the sixteen (`BinFill, PickXtimes, SwingXtimes, StopCube, VideoUnmask,
ButtonUnmask, VideoUnmaskSwap, ButtonUnmaskSwap, PickHighlight, VideoRepick, VideoPlaceButton,
VideoPlaceOrder, MoveCube, InsertPeg, PatternLock, RouteStick`). To shard a task across
processes use `--args.episode_start=25 --args.episodes_per_task=50` (episodes 25–49) or
`--args.episode_list=3,7,11`; give every process its own `policy_name`, because two processes
writing one `progress.json` corrupt it. `--args.model_seed` only names the output directory.

Outputs land in `runs/evaluation/<policy_name>/ckpt79999/seed<model_seed>/qwenvl/`:
`progress.json` (per task, per episode: `true`, `false`, or `"error"`), `videos/` (one mp4 per
episode, its name ending in `success`, `fail` or `timeout` and the episode's difficulty), and the
composer's per-call inputs. The memory prints its own decisions to stdout with the `[agentmem]`
prefix: the policy it chose (`watching [...] in the live frames (3/3 votes)`), every note taken,
every read (`step is identified by ... -- reading the note`), and a per-episode summary.

Per-episode wall-clock (median, one evaluator among ten on a shared box): most tasks 1–5 min;
VideoRepick 6, VideoPlaceOrder 8, VideoPlaceButton 9, InsertPeg 32 (cap-length episodes). The
full 16 × 50 table is roughly 75 evaluator-hours.

## 4. Reading results

Success for a task is the count of `true` over the episodes run. An `"error"` entry is an
infrastructure fault (the video decode above), not a task failure: it is dropped when the run is
resumed, so simply run the same command again and it re-runs those episodes and skips the rest.
Never pass `--args.overwrite` on a directory whose episodes you want to keep.

What to expect at 50 episodes per task, seed 7, two independent runs:

| task | run 1 | run 2 | no memory (published) |   | task | run 1 | run 2 | no memory |
|---|---|---|---|---|---|---|---|---|
| BinFill | 64 | 66 | 52.00 | | VideoRepick | 64 | 66 | 25.33 |
| PickXtimes | 92 | 94 | 92.67 | | VideoPlaceButton | 46 | 44 | 54.00 |
| SwingXtimes | 60 | 68 | 7.33 | | VideoPlaceOrder | 88 | 90 | 31.78 |
| StopCube | 0 | 0 | 0.00 | | MoveCube | 62 | 70 | 71.56 |
| VideoUnmask | 92 | 98 | 88.67 | | InsertPeg | 0 | 0 | 3.33 |
| ButtonUnmask | 94 | 86 | 24.00 | | PatternLock | 96 | 92 | 6.67 |
| VideoUnmaskSwap | 80 | 82 | 30.67 | | RouteStick | 46 | 52 | 6.00 |
| ButtonUnmaskSwap | 66 | 50 | 14.00 | | **AVG** | **60.50** | **61.25** | 32.70 |

A 50-episode cell carries a ±13-point Wilson interval; ButtonUnmaskSwap in particular flips on
half its episodes between runs.

## 5. Baselines and ablations

- No memory (the published GroundSG+QwenVL row): same command without `--args.use_agentmem`.
- Detector refinement only: `--args.use_detector_refine --args.use_qwenvl` instead of `--args.use_agentmem`.
- Ablation switches (all opt-in; the default is the reported configuration):

| switch | removes |
|---|---|
| `AGENTMEM_POLICY=rules` / `=all` | the agent's memory policy → keyword rule / always take notes on all three colours |
| `AGENTMEM_VOTE=1` | the three-wrapping vote → a single wrapping |
| `AGENTMEM_COLOURRULE=0` | the rule that only colour-named phrases are noted |
| `AGENTMEM_RETRIEVAL=loose` / `=agent` | structural retrieval → token overlap / ask the agent per subgoal |
| `AGENTMEM_KW=0` | agent-named covers → the shared phrase `white cube` |
| `AGENTMEM_TRACK=0`, `AGENTMEM_DEMOFOLLOW=0` | live cover tracking / demonstration cover following |
| `AGENTMEM_DEMO=0`, `AGENTMEM_VIDEO2=0`, `AGENTMEM_PATH=0`, `AGENTMEM_ORD=0` | demonstration notes / demonstrated objects and places / route memory / flash-counted repetition |
| `AGENTMEM_REVERT=0` | the stale-request revert (`AGENTMEM_STALE` sets its patience, 150) |
| `AGENTMEM_REFINE=0` / `=1` | selective refinement → none / every verb |
| `AGENTMEM_VERBS=contact`, `AGENTMEM_LOCGUARD=1` | two post-hoc guards, both measured nulls; off in the table |
| `AGENTMEM_SG=1` | adds the scene-graph readers (`agentmem/scene_graph.py`, also copied next to `agent_memory.py`), switched on by keyword tests on the prompt: the relation *cube on a white highlight disc* recorded as an event and used to answer "the highlighted cube" (PickHighlight 30/50 vs 11/50, p = 0.0002), the demonstration read as events for "the block that was previously picked up" (VideoRepick) and "the target right after/before the button was pressed" (VideoPlaceButton); not in the reported table |
| `AGENTMEM_SG=agent` | the same readers, but chosen by the write-time agent: a separate call after the plan asks, for each WATCH item, how the task description picks the thing out (appearance / mark / handled / sequence) and each answer maps to one reader (three more text calls per episode; the plan's own decisions are untouched — appended to the plan prompt the question flipped SOURCE on two tasks). Offline routing accuracy: see the review log (`campaign/sg/route_study_v4.py`) |

## 6. Things that bit us

- Killing an evaluator: kill the `python run_subset_eval.py` process by its exact PID; a
  `micromamba run` / shell wrapper left behind keeps the grandchild alive, and `pkill -f` matches
  your own shell. Check `nvidia-smi` for orphans.
- The benign line `Resource temporarily unavailable` at the top of every log comes from the
  environment manager's lock; the real failure signature is the `RuntimeError` from SAPIEN or the
  `libav.swscaler` line above.
- Do not edit `agent_memory.py` / `subgoal_predictor.py` while evaluators are starting: a
  syntax error is imported by the next process to launch.
- Everything is `/workspace`-style paths in our logs; nothing in the method assumes them except
  the scratch-directory fallback in `AgentMemory.__init__`, which honours `TMPDIR`.

Questions: yuyouz@andrew.cmu.edu.
