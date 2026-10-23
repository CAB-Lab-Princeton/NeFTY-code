"""Evaluate the runs of a train_batch.py output root and aggregate the metrics.

    python evaluate_batch.py --output_root ./logs/output_batch_<timestamp> --gpus 0,1

The task list is the one recorded by train_batch.py (<output_root>/tasks.json), unless
--indices / --settings / --dataset_size select tasks explicitly or --all_found evaluates
every <setting>/idx_<i> directory found. Writes per-specimen
<output_root>/<setting>/idx_<i>/output_visualization/metrics.txt, per-setting
<output_root>/<setting>/metrics_report.txt, and a consolidated
<output_root>/metrics_report.txt (layered vs. not layered; mean +/- 95% CI).
"""
import argparse
import os
import sys

from utils.batch import (SECTIONED_METRIC_KEYS, add_eval_selection_args, find_checkpoint, parse_gpus,
                         report_failures, resolve_eval_tasks, run_tasks, write_summaries)


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output_root", type=str, required=True,
                        help="Root output dir written by train_batch.py")
    parser.add_argument("--gpus", type=str, default="0,1", help="Comma-separated GPU ids")
    parser.add_argument("--max_per_gpu", type=int, default=20, help="Concurrent runs per GPU")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the evaluation commands; run and write nothing")
    parser.add_argument("--metrics_only", action="store_true",
                        help="Only summarize existing metrics.txt files")
    add_eval_selection_args(parser)
    return parser


def run(evaluator: str, log_name: str, description: str, preview_checkpoint: str = "best.tar") -> None:
    """``preview_checkpoint``: checkpoint name shown by --dry_run before training has run
    (best.tar for NeFTY / Grid Opt., final.tar for the PINN)."""
    args = build_parser(description).parse_args()
    # A dry run before training (no tasks.json yet) previews the full dataset
    tasks, _ = resolve_eval_tasks(args.output_root, args.indices, args.settings, args.dataset_size,
                                  args.all_found, default_dataset_size=1000 if args.dry_run else None)
    print(f"Total eval tasks: {len(tasks)}")
    print(f"Output root: {args.output_root}")

    failed_tasks = []
    if not args.metrics_only:
        gpus = parse_gpus(args.gpus)

        def make_cmd(task, gpu_id):
            exp_dir = os.path.join(args.output_root, task.setting_name, f"idx_{task.eval_idx}")
            try:
                checkpoint = find_checkpoint(exp_dir)
            except FileNotFoundError:
                if not args.dry_run:
                    raise
                checkpoint = os.path.join(exp_dir, preview_checkpoint)  # preview before training
            cmd = [
                sys.executable, "-u", evaluator,
                "--checkpoint", checkpoint,
                "--output_dir", os.path.join(exp_dir, "output_visualization"),
            ]
            return cmd, os.path.join(exp_dir, log_name)

        print(f"GPUs: {gpus} (max {args.max_per_gpu} per GPU)")
        results = run_tasks(tasks, make_cmd, gpus, args.max_per_gpu, args.dry_run, desc="Evaluation")
        report_failures(results)
        failed_tasks = [r[0] for r in results if r[1] != 0]

    if args.dry_run:
        print("Dry run: no summaries written.")
        return

    # Failed evaluations are reported as failed; any old metrics.txt of theirs is ignored
    write_summaries(args.output_root, tasks, SECTIONED_METRIC_KEYS, sectioned=True,
                    failed_tasks=failed_tasks)
    print(f"Summaries written to {os.path.join(args.output_root, 'metrics_report.txt')}")
    if failed_tasks:
        sys.exit(1)


if __name__ == "__main__":
    run("evaluate.py", "evaluate.log", "Batch evaluation for NeFTY / Grid Opt.")
