"""Evaluate one trained U-Net checkpoint on every benchmark specimen and aggregate.

    python evaluate_unet_batch.py --checkpoint ./logs/unet_sound_only/best.tar \
        --datadir ./data/synthetic_1000 --gpus 0,1

Writes <output_root>/<setting>/idx_<i>/output_visualization/metrics.txt per specimen,
<output_root>/<setting>/metrics_report.txt per category, and a consolidated
<output_root>/metrics_report.txt (layered vs. not layered; mean +/- 95% CI).
"""
import argparse
import os
import sys
import time

from utils.batch import (UNET_METRIC_KEYS, add_eval_selection_args, parse_gpus, report_failures,
                         resolve_eval_tasks, run_tasks, save_task_list, write_summaries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch evaluation for the U-Net baselines")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to best.tar/final.tar or a directory containing them")
    parser.add_argument("--args_file", type=str, default=None,
                        help="Optional args.txt path (defaults to <checkpoint dir>/args.txt)")
    parser.add_argument("--datadir", type=str, default=None,
                        help="Evaluation dataset (default in evaluate_unet.py: ./data/synthetic_1000)")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Root output dir. Defaults to ./logs/output_unet_batch_<timestamp>")
    parser.add_argument("--gpus", type=str, default="0,1", help="Comma-separated GPU ids")
    parser.add_argument("--max_per_gpu", type=int, default=20, help="Concurrent runs per GPU")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the evaluation commands; run and write nothing")
    parser.add_argument("--metrics_only", action="store_true",
                        help="Only summarize existing metrics.txt files under --output_root")
    add_eval_selection_args(parser)
    return parser.parse_args()


def resolve_checkpoint(checkpoint_path: str) -> str:
    if os.path.isdir(checkpoint_path):
        for name in ("best.tar", "final.tar"):
            path = os.path.join(checkpoint_path, name)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def main() -> None:
    args = parse_args()

    if args.metrics_only:
        if args.output_root is None:
            raise ValueError("--output_root is required with --metrics_only")
        tasks, _ = resolve_eval_tasks(args.output_root, args.indices, args.settings, args.dataset_size,
                                      args.all_found)
        if args.dry_run:
            print(f"Dry run: would summarize {len(tasks)} tasks under {args.output_root}")
            return
        write_summaries(args.output_root, tasks, UNET_METRIC_KEYS, sectioned=False)
        print("Metrics-only mode: summaries updated.")
        return

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --metrics_only is set")
    if args.dry_run and not os.path.exists(args.checkpoint):
        checkpoint_path = args.checkpoint  # not trained yet; only print the commands
    else:
        checkpoint_path = resolve_checkpoint(args.checkpoint)
    args_file = args.args_file
    if args_file is None:
        candidate = os.path.join(os.path.dirname(checkpoint_path), "args.txt")
        if os.path.exists(candidate):
            args_file = candidate

    output_root = args.output_root or os.path.join("./logs", f"output_unet_batch_{time.strftime('%Y%m%d_%H%M%S')}")
    # Default: the full 1,000-sample benchmark (or the explicit selection)
    tasks, dataset_size = resolve_eval_tasks(output_root, args.indices, args.settings, args.dataset_size,
                                             args.all_found, default_dataset_size=1000)
    if not args.dry_run:
        save_task_list(output_root, tasks, dataset_size)
    gpus = parse_gpus(args.gpus)

    def make_cmd(task, gpu_id):
        exp_dir = os.path.join(output_root, task.setting_name, f"idx_{task.eval_idx}")
        cmd = [
            sys.executable, "-u", "evaluate_unet.py",
            "--checkpoint", checkpoint_path,
            "--eval_idx", str(task.eval_idx),
            "--output_dir", os.path.join(exp_dir, "output_visualization"),
        ]
        if args_file:
            cmd.extend(["--args_file", args_file])
        if args.datadir:
            cmd.extend(["--datadir", args.datadir])
        return cmd, os.path.join(exp_dir, "evaluate_unet.log")

    print(f"Total eval tasks: {len(tasks)}")
    print(f"Output root: {output_root}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"GPUs: {gpus} (max {args.max_per_gpu} per GPU)")

    results = run_tasks(tasks, make_cmd, gpus, args.max_per_gpu, args.dry_run, desc="Evaluation")
    report_failures(results)
    failed_tasks = [r[0] for r in results if r[1] != 0]
    if args.dry_run:
        print("Dry run: no summaries written.")
        return

    # Failed evaluations are reported as failed; any old metrics.txt of theirs is ignored
    write_summaries(output_root, tasks, UNET_METRIC_KEYS, sectioned=False, failed_tasks=failed_tasks)
    print(f"Summaries written to {os.path.join(output_root, 'metrics_report.txt')}")
    if failed_tasks:
        sys.exit(1)


if __name__ == "__main__":
    main()
