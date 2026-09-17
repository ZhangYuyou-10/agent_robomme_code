"""
Run a subset of RoboMME evaluation: a chosen task list, capped episodes per task.

Reuses eval.py's own Args / EpisodeEvaluator / save-dir / log-dict machinery unmodified,
so the per-episode rollout logic is identical to the official reproduction path. Only the
outer loop is different: it caps episodes per task instead of always running all of them,
which eval.py has no flag for.
"""
import dataclasses
import json
import os
import time
from pathlib import Path

import eval as eval_mod


@dataclasses.dataclass
class SubsetArgs(eval_mod.Args):
    episodes_per_task: int = 10
    episode_start: int = 0  # shard a task by episode range [episode_start, episodes_per_task)
    episode_list: str = ""  # exact episode ids, comma-separated; overrides the range. Used by
                            # refill runs so only the episodes that failed are redone.


def evaluate_subset(args: SubsetArgs):
    eval_mod.check_args(args)

    save_dir = eval_mod.setup_save_directory(args)
    video_save_dir = save_dir / "videos"

    log_dict = eval_mod.setup_log_dict(save_dir, args)

    if args.only_tasks:
        task_names = args.only_tasks.split(",")
    else:
        task_names = eval_mod.TASK_NAME_LIST

    subgoal_predictor = eval_mod.build_subgoal_predictor(args, save_dir)
    evaluator = eval_mod.EpisodeEvaluator(args, save_dir)

    for task_name in task_names:
        if task_name not in log_dict:
            log_dict[task_name] = {}

        env_runner = eval_mod.EnvRunner(task_name, video_save_dir, max_steps=args.max_steps)
        num_episodes = min(env_runner.num_episodes, args.episodes_per_task)

        if args.episode_list:
            ids = [int(x) for x in args.episode_list.split(",") if x.strip()]
        else:
            ids = list(range(args.episode_start, num_episodes))
        for episode_id in ids:
            if str(episode_id) in log_dict[task_name]:
                print(f"[robomme] episode {episode_id} already evaluated, skipping...")
                continue

            env_runner.make_env(episode_id)
            print(f"\n[robomme] env for task {task_name} episode {episode_id} setup finished")

            try:
                success_flag = evaluator.eval_each_episode(env_runner, subgoal_predictor, video_save_dir)
                if success_flag == "unknown":
                    log_dict[task_name][episode_id] = "error"
                else:
                    log_dict[task_name][episode_id] = success_flag == "success"
            except Exception as e:
                print(f"Error evaluating episode {episode_id} for task {task_name}: {e}")
                log_dict[task_name][episode_id] = "error"

            env_runner.close_env()
            with open(save_dir / "progress.json", "w") as f:
                json.dump(log_dict, f, indent=2)

        del env_runner
        time.sleep(1)

    final_results = {}
    final_results["success_rate"] = {
        task_name: sum(v is True for v in log_dict[task_name].values()) / len(log_dict[task_name].values())
        for task_name in log_dict.keys()
        if len(log_dict[task_name]) > 0
    }
    final_results["total_success_rate"] = (
        sum(final_results["success_rate"].values()) / len(final_results["success_rate"].values())
    )
    with open(save_dir / "log.json", "w") as f:
        json.dump(final_results, f, indent=2)
    print(json.dumps(final_results, indent=2))


if __name__ == "__main__":
    import tyro
    tyro.cli(evaluate_subset)
