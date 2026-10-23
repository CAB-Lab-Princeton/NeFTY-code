"""Train one model per specimen in parallel across GPUs.

Example (full 1,000-sample benchmark, NeFTY):
    python train_batch.py --config configs/nefty.txt --data_dir ./data/synthetic_1000 --gpus 0,1

Example (bundled mini dataset, two specimens):
    python train_batch.py --config configs/quick/nefty.txt --data_dir ./data/synthetic_test \
        --indices 0 7 --dataset_size 8 --gpus 0 --max_per_gpu 1

Runs are written to <output_root>/<setting>/idx_<i>/ and can then be evaluated with
evaluate_batch.py (NeFTY, Grid Opt.) or evaluate_pinn_batch.py (PINN).
"""
import argparse
import os
import shlex
import sys
import time

from utils.batch import (add_task_selection_args, parse_gpus, report_failures, run_tasks,
                         load_task_list, save_task_list, select_tasks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch per-specimen training")
    parser.add_argument("--script", type=str, default="train.py", choices=["train.py", "train_pinn.py"],
                        help="Per-specimen training script (train.py: NeFTY / Grid Opt.; train_pinn.py: PINN)")
    parser.add_argument("--config", type=str, default="configs/nefty.txt")
    parser.add_argument("--data_dir", type=str, default="./data/synthetic_1000")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Root output dir. Defaults to ./logs/output_batch_<timestamp>")
    parser.add_argument("--gpus", type=str, default="0,1", help="Comma-separated GPU ids")
    parser.add_argument("--max_per_gpu", type=int, default=20, help="Concurrent runs per GPU")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--extra_args", type=str, default="",
                        help="Extra arguments forwarded to the training script, e.g. \"--N_iters 100\"")
    parser.add_argument("--dry_run", action="store_true", help="Print the commands without running them")
    add_task_selection_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Dataset size: explicit flag, else the size recorded in a reused output root, else 1000
    if args.dataset_size is None and args.output_root:
        args.dataset_size = load_task_list(args.output_root)[1]
    if args.dataset_size is None:
        args.dataset_size = 1000
    tasks = select_tasks(args.indices, args.settings, args.dataset_size)

    output_root = args.output_root or os.path.join("./logs", f"output_batch_{time.strftime('%Y%m%d_%H%M%S')}")
    if not args.dry_run:
        save_task_list(output_root, tasks, args.dataset_size)  # evaluation and aggregation reuse this list
    gpus = parse_gpus(args.gpus)

    def make_cmd(task, gpu_id):
        setting_dir = os.path.join(output_root, task.setting_name)
        expname = f"idx_{task.eval_idx}"
        cmd = [
            sys.executable, "-u", args.script,
            "--config", args.config,
            "--datadir", args.data_dir,
            "--basedir", setting_dir,
            "--expname", expname,
            "--eval_idx", str(task.eval_idx),
            "--seed", str(args.seed),
            "--gpu", str(gpu_id),
            "--disable_tqdm",
        ]
        if args.extra_args:
            cmd.extend(shlex.split(args.extra_args))
        return cmd, os.path.join(setting_dir, expname, "train.log")

    print(f"Total tasks: {len(tasks)}")
    print(f"Output root: {output_root}")
    print(f"GPUs: {gpus} (max {args.max_per_gpu} per GPU)")

    results = run_tasks(tasks, make_cmd, gpus, args.max_per_gpu, args.dry_run, desc="Training")
    if report_failures(results):
        sys.exit(1)
    print("All training tasks completed successfully.")


if __name__ == "__main__":
    main()
