"""Shared helpers for the batch train/evaluate drivers.

A batch run is a directory tree ``<output_root>/<setting>/idx_<i>/`` with one
per-specimen run per leaf. ``<setting>`` is the benchmark category of the sample
(``1_defect`` ... ``4_defect`` for the homogeneous split, ``3_layer`` / ``4_layer``
for the layered split).
"""
import math
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm


@dataclass(frozen=True)
class Task:
    setting_name: str
    eval_idx: int


SETTINGS = ["1_defect", "2_defect", "3_defect", "4_defect", "3_layer", "4_layer"]
LAYERED_SETTINGS = {"3_layer", "4_layer"}


def setting_of_index(idx: int, dataset_size: int) -> str:
    """Benchmark category of sample ``idx`` (same schedule and arithmetic as
    dataset/synthetic.py). For the 1,000-sample benchmark this gives
    1_defect 0-199, 2_defect 200-399, 3_defect 400-600, 4_defect 601-799,
    3_layer 800-900, 4_layer 901-999 (the generator's floating-point bucket edges)."""
    p = idx / max(dataset_size, 1)
    if p < 0.8:
        sub_p = p / 0.8
        n_defects = 1 if sub_p < 0.25 else 2 if sub_p < 0.5 else 3 if sub_p < 0.75 else 4
        return f"{n_defects}_defect"
    sub_p_layered = (p - 0.8) / 0.2
    return "3_layer" if sub_p_layered < 0.5 else "4_layer"


def build_default_tasks(dataset_size: int = 1000) -> List[Task]:
    """Every sample of the dataset (the full 1,000-sample benchmark by default),
    each assigned to its category."""
    return [Task(setting_of_index(i, dataset_size), i) for i in range(dataset_size)]


def select_tasks(indices: Optional[Sequence[int]] = None,
                 settings: Optional[Sequence[str]] = None,
                 dataset_size: int = 1000) -> List[Task]:
    """All ``dataset_size`` samples by default; ``indices`` restricts the run to those
    samples. ``settings`` keeps only the listed categories."""
    if dataset_size is None or dataset_size <= 0:
        raise ValueError(f"--dataset_size must be a positive integer (got {dataset_size})")
    if indices:
        bad = [i for i in indices if i < 0 or i >= dataset_size]
        if bad:
            raise ValueError(f"--indices out of range for a dataset of {dataset_size} samples: {bad}")
        dupes = sorted({i for i in indices if list(indices).count(i) > 1})
        if dupes:
            raise ValueError(f"--indices contains duplicates: {dupes}")
        tasks = [Task(setting_of_index(i, dataset_size), i) for i in indices]
    else:
        tasks = build_default_tasks(dataset_size)
    if settings:
        unknown = set(settings) - set(SETTINGS)
        if unknown:
            raise ValueError(f"Unknown settings {sorted(unknown)}; choose from {SETTINGS}")
        tasks = [t for t in tasks if t.setting_name in set(settings)]
    return tasks


def add_task_selection_args(parser) -> None:
    parser.add_argument("--indices", type=int, nargs="+", default=None,
                        help="Sample indices to run (default: every sample, 0 .. dataset_size-1)")
    parser.add_argument("--settings", type=str, nargs="+", default=None, choices=SETTINGS,
                        help="Only run these categories")
    parser.add_argument("--dataset_size", type=int, default=None,
                        help="Number of samples in the dataset; sets the default task list and "
                             "the index -> category assignment (default: the size recorded in "
                             "<output_root>/tasks.json, else 1000)")


TASKS_FILE = "tasks.json"


def save_task_list(output_root: str, tasks: List[Task], dataset_size: Optional[int]) -> None:
    """Record the selected tasks (and the dataset size used to assign their categories)
    so that evaluation and aggregation use the same list. An unknown size (None) never
    replaces a recorded one."""
    import json
    if dataset_size is None:
        dataset_size = load_task_list(output_root)[1]
    os.makedirs(output_root, exist_ok=True)
    with open(os.path.join(output_root, TASKS_FILE), "w") as f:
        json.dump({"dataset_size": dataset_size,
                   "tasks": [{"setting": t.setting_name, "index": t.eval_idx} for t in tasks]},
                  f, indent=0)


def load_task_list(output_root: str) -> Tuple[Optional[List[Task]], Optional[int]]:
    """(tasks, dataset_size) recorded in <output_root>/tasks.json, or (None, None)."""
    import json
    path = os.path.join(output_root, TASKS_FILE)
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        record = json.load(f)
    tasks = [Task(d["setting"], int(d["index"])) for d in record["tasks"]]
    return tasks, record.get("dataset_size")


def add_eval_selection_args(parser) -> None:
    """Task selection for the evaluators (see resolve_eval_tasks)."""
    parser.add_argument("--indices", type=int, nargs="+", default=None,
                        help="Evaluate these sample indices")
    parser.add_argument("--settings", type=str, nargs="+", default=None, choices=SETTINGS,
                        help="Only evaluate these categories")
    parser.add_argument("--dataset_size", type=int, default=None,
                        help="Number of samples in the dataset (with --indices / --settings, "
                             "or alone to select all samples)")
    parser.add_argument("--all_found", action="store_true",
                        help="Evaluate every <setting>/idx_<i> directory found under --output_root")


def resolve_eval_tasks(output_root: str, indices=None, settings=None, dataset_size=None,
                       all_found: bool = False, default_dataset_size: Optional[int] = None
                       ) -> Tuple[List[Task], Optional[int]]:
    """(tasks, dataset_size) for evaluation and aggregation, in order of precedence:
    --all_found (every run directory found), an explicit selection (--indices /
    --settings / --dataset_size), the list recorded by train_batch.py
    (<output_root>/tasks.json), or the full dataset if ``default_dataset_size`` is given.
    An explicit selection without --dataset_size uses the recorded dataset size
    (else 1000) to assign categories."""
    recorded, recorded_size = load_task_list(output_root)
    if all_found:
        return discover_tasks(output_root), (dataset_size if dataset_size is not None else recorded_size)
    if indices or settings or dataset_size is not None:
        size = dataset_size if dataset_size is not None else (recorded_size or default_dataset_size or 1000)
        return select_tasks(indices, settings, size), size
    if recorded is not None:
        return recorded, recorded_size
    if default_dataset_size is not None:
        return select_tasks(None, None, default_dataset_size), default_dataset_size
    raise ValueError(f"No {TASKS_FILE} in {output_root}: pass --indices / --settings / "
                     "--dataset_size to select the tasks, or --all_found to evaluate every run "
                     "directory found there.")


def discover_tasks(output_root: str) -> List[Task]:
    """Find ``<output_root>/<setting>/idx_<i>`` run directories."""
    tasks: List[Task] = []
    if not os.path.exists(output_root):
        print(f"Warning: Output root {output_root} does not exist")
        return []

    for setting_name in os.listdir(output_root):
        setting_path = os.path.join(output_root, setting_name)
        if not os.path.isdir(setting_path):
            continue
        for exp_name in os.listdir(setting_path):
            exp_path = os.path.join(setting_path, exp_name)
            if not os.path.isdir(exp_path) or not exp_name.startswith("idx_"):
                continue
            try:
                tasks.append(Task(setting_name=setting_name, eval_idx=int(exp_name.split("_")[1])))
            except (IndexError, ValueError):
                continue

    return sorted(tasks, key=lambda t: (t.setting_name, t.eval_idx))


def parse_gpus(gpus: str) -> List[str]:
    gpu_list = [gpu.strip() for gpu in gpus.split(",") if gpu.strip()]
    if not gpu_list:
        raise ValueError("No GPUs provided via --gpus")
    return gpu_list


def check_max_per_gpu(max_per_gpu: int) -> None:
    if max_per_gpu <= 0:
        raise ValueError(f"--max_per_gpu must be positive (got {max_per_gpu})")


def find_checkpoint(exp_dir: str) -> str:
    """Prefer best.tar, fall back to final.tar."""
    for name in ("best.tar", "final.tar"):
        path = os.path.join(exp_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No checkpoint found in {exp_dir}")


# ---------------------------------------------------------------------------
# Subprocess pool: one process per task, at most `max_per_gpu` per GPU
# ---------------------------------------------------------------------------

_process_lock = threading.Lock()
_active_processes: List[subprocess.Popen] = []


def _register(proc: subprocess.Popen) -> None:
    with _process_lock:
        _active_processes.append(proc)


def _unregister(proc: subprocess.Popen) -> None:
    with _process_lock:
        if proc in _active_processes:
            _active_processes.remove(proc)


def _terminate(proc: subprocess.Popen, timeout: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=timeout)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _terminate_all() -> None:
    with _process_lock:
        processes = list(_active_processes)
    for proc in processes:
        _terminate(proc)


def run_tasks(tasks: List[Task],
              make_cmd: Callable[[Task, str], Tuple[List[str], str]],
              gpus: List[str],
              max_per_gpu: int,
              dry_run: bool = False,
              desc: str = "Tasks") -> List[tuple]:
    """Run ``make_cmd(task, gpu) -> (cmd, log_path)`` for every task.

    Tasks are assigned to GPUs round-robin; each GPU runs at most ``max_per_gpu``
    processes at a time. Returns a list of ``(task, return_code, log_path, cmd)``.
    On SIGINT/SIGTERM the running processes are terminated and the program exits with
    status 128 + signal number (130 / 143).
    """
    check_max_per_gpu(max_per_gpu)
    scheduled = [(task, gpus[i % len(gpus)]) for i, task in enumerate(tasks)]

    if dry_run:
        # Print the commands in order, without starting any process
        results = []
        for task, gpu_id in scheduled:
            cmd, _ = make_cmd(task, gpu_id)
            print(" ".join(cmd))
            results.append((task, 0, "DRY_RUN", cmd))
        return results

    stop_event = threading.Event()
    received = {"signum": None}
    semaphores: Dict[str, threading.Semaphore] = {gpu: threading.Semaphore(max_per_gpu) for gpu in gpus}

    def handle_interrupt(signum, frame):
        received["signum"] = signum
        stop_event.set()
        print("\nInterrupt received. Terminating running tasks...")
        _terminate_all()

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    def worker(task: Task, gpu_id: str):
        semaphore = semaphores[gpu_id]
        while True:
            if stop_event.is_set():
                return (task, 130, "INTERRUPTED", [])
            if semaphore.acquire(timeout=0.2):
                break
        try:
            cmd, log_path = make_cmd(task, gpu_id)
            if stop_event.is_set():
                return (task, 130, "INTERRUPTED", cmd)

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "w") as f:
                process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
                _register(process)
                try:
                    while True:
                        if stop_event.is_set():
                            _terminate(process)
                            return (task, 130, log_path, cmd)
                        return_code = process.poll()
                        if return_code is not None:
                            return (task, return_code, log_path, cmd)
                        time.sleep(0.2)
                finally:
                    _unregister(process)
        except Exception as e:
            return (task, 1, f"worker exception: {e}", None)
        finally:
            semaphore.release()

    results = []
    executor = ThreadPoolExecutor(max_workers=max(1, len(gpus) * max_per_gpu))
    try:
        futures = [executor.submit(worker, task, gpu_id) for task, gpu_id in scheduled]
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            if stop_event.is_set():
                break
            results.append(future.result())
    except KeyboardInterrupt:
        stop_event.set()
        _terminate_all()
        executor.shutdown(wait=False, cancel_futures=True)
        print("Interrupted. Cleaned up running subprocesses.")
        sys.exit(130)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    if stop_event.is_set():
        _terminate_all()
        signum = received["signum"] or signal.SIGINT
        print(f"Cancelled by signal {signum}; {len(results)}/{len(tasks)} tasks had finished.")
        sys.exit(128 + int(signum))
    return results


def report_failures(results: List[tuple]) -> bool:
    """Print failed tasks; return True if any failed."""
    failures = [r for r in results if r[1] != 0]
    if failures:
        print(f"{len(failures)} tasks failed:")
        for task, code, log_path, _ in failures:
            print(f"  - {task.setting_name} idx={task.eval_idx}: exit={code}, log={log_path}")
    return bool(failures)


# ---------------------------------------------------------------------------
# Metric aggregation: mean +/- 95% CI (1.96 * sample std / sqrt(n))
# ---------------------------------------------------------------------------

def _mean_ci(values: List[float]) -> Tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n > 1:
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        ci = 1.96 * math.sqrt(variance) / math.sqrt(n)
    else:
        ci = 0.0
    return mean, ci


def _format_metric(key: str, value: float) -> str:
    if "psnr" in key.lower():
        return f"{value:.2f}"
    if "ssim" in key.lower():
        return f"{value:.4f}"
    return f"{value:.6f}"


def _parse_metrics_file(metrics_path: str, sectioned: bool) -> Dict[str, float]:
    """Parse a per-specimen metrics.txt into {key: value}.

    ``sectioned=True``: lines "Section:" open a section and keys become "Section|Key"
    (evaluate.py / evaluate_pinn.py format). Otherwise keys are used as-is
    (evaluate_unet.py format).
    """
    metrics: Dict[str, float] = {}
    current_section = None
    with open(metrics_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if sectioned and line.endswith(":") and not line.startswith("MSE"):
                current_section = line[:-1]
                continue
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip()
            if sectioned:
                key = f"{current_section}|{key}"
            try:
                metrics[key] = float(val.strip())
            except ValueError:
                continue
    return metrics


def load_excluded(output_root: str) -> set:
    """Optional ``<output_root>/outliers.txt``: one run per line (``<setting>/idx_<i>`` or a
    path to its metrics.txt, relative to output_root); '#' starts a comment. Listed runs
    are left out of the summaries. Without the file, every run is included."""
    path = os.path.join(output_root, "outliers.txt")
    excluded = set()
    if not os.path.exists(path):
        return excluded
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            candidate = line if os.path.isabs(line) else os.path.join(output_root, line)
            if os.path.isdir(candidate):
                candidate = os.path.join(candidate, "output_visualization", "metrics.txt")
            excluded.add(os.path.abspath(candidate))
    if excluded:
        print(f"Excluding {len(excluded)} run(s) listed in {path} from the summaries")
    return excluded


@dataclass
class _Coverage:
    requested: int = 0
    evaluated: int = 0
    missing: int = 0
    failed: int = 0
    excluded: int = 0


def _collect(output_root: str, tasks: List[Task], keys: List[str], sectioned: bool,
             excluded: set = frozenset(), failed: frozenset = frozenset()
             ) -> Tuple[Dict[str, List[float]], _Coverage]:
    """Gather the values of ``keys`` over ``tasks``. Values may include NaN/inf;
    those are excluded (and counted) when rendering."""
    records: Dict[str, List[float]] = {k: [] for k in keys}
    cov = _Coverage(requested=len(tasks))
    for task in tasks:
        metrics_path = os.path.join(output_root, task.setting_name, f"idx_{task.eval_idx}",
                                    "output_visualization", "metrics.txt")
        if os.path.abspath(metrics_path) in excluded:
            cov.excluded += 1
            continue
        if task in failed:  # its metrics.txt, if any, is stale
            cov.failed += 1
            continue
        if not os.path.exists(metrics_path):
            cov.missing += 1
            continue
        cov.evaluated += 1
        metrics = _parse_metrics_file(metrics_path, sectioned)
        for key in records:
            if key in metrics:
                records[key].append(metrics[key])
    return records, cov


def _metric_lines(name: str, values: List[float], evaluated: int) -> Tuple[List[str], bool]:
    """Mean +/- 95% CI over the finite values; returns (lines, complete)."""
    finite = [v for v in values if math.isfinite(v)]
    nonfinite = len(values) - len(finite)
    absent = evaluated - len(values)
    lines: List[str] = []
    if finite:
        mean, ci = _mean_ci(finite)
        lines.append(f"{name}: {_format_metric(name, mean)} ± {_format_metric(name, ci)}")
    else:
        lines.append(f"{name}: n/a")
    if nonfinite or absent:
        notes = [f"computed on {len(finite)}/{evaluated} evaluated specimens"]
        if nonfinite:
            notes.append(f"{nonfinite} non-finite excluded")
        if absent:
            notes.append(f"{absent} missing this metric")
        lines.append("  (" + "; ".join(notes) + ")")
    return lines, not (nonfinite or absent)


def _render(records: Dict[str, List[float]], cov: _Coverage, keys: List[str],
            sectioned: bool) -> List[str]:
    body: List[str] = []
    complete = cov.missing == 0 and cov.failed == 0
    if sectioned:
        sections: List[str] = []
        for key in keys:
            section = key.split("|", 1)[0]
            if section not in sections:
                sections.append(section)
        for section in sections:
            body.append(f"{section}:")
            for key in keys:
                if not key.startswith(section + "|"):
                    continue
                lines, ok = _metric_lines(key.split("|", 1)[1], records.get(key, []), cov.evaluated)
                body.extend(lines)
                complete = complete and ok
            body.append("")
    else:
        for key in keys:
            lines, ok = _metric_lines(key, records.get(key, []), cov.evaluated)
            body.extend(lines)
            complete = complete and ok
    header = [f"Specimens: requested {cov.requested}, evaluated {cov.evaluated}, "
              f"missing {cov.missing}, failed {cov.failed}, excluded {cov.excluded}"]
    if not complete:
        header.insert(0, "WARNING: incomplete coverage (missing or failed specimens, missing "
                         "metrics, or non-finite values); see the counts below.")
    return header + body


def write_summaries(output_root: str, tasks: List[Task], keys: List[str], sectioned: bool,
                    failed_tasks=()) -> None:
    """Write ``<setting>/metrics_report.txt`` per category and a consolidated
    ``metrics_report.txt`` (layered vs. not layered) under ``output_root``."""
    excluded = load_excluded(output_root)
    failed = frozenset(failed_tasks)  # evaluations that failed in this run
    by_setting: Dict[str, List[Task]] = {}
    for task in tasks:
        by_setting.setdefault(task.setting_name, []).append(task)

    for setting_name, setting_tasks in by_setting.items():
        records, cov = _collect(output_root, setting_tasks, keys, sectioned, excluded, failed)
        lines = _render(records, cov, keys, sectioned)
        os.makedirs(os.path.join(output_root, setting_name), exist_ok=True)
        with open(os.path.join(output_root, setting_name, "metrics_report.txt"), "w") as f:
            f.write("\n".join(lines).strip() + "\n")

    groups = {
        "layered": [t for t in tasks if t.setting_name in LAYERED_SETTINGS],
        "not_layered": [t for t in tasks if t.setting_name not in LAYERED_SETTINGS],
    }
    titles = {"layered": "Layered (3_layer, 4_layer)", "not_layered": "Not layered (others)"}
    lines: List[str] = []
    for group_key in ["layered", "not_layered"]:
        if not groups[group_key]:
            continue
        records, cov = _collect(output_root, groups[group_key], keys, sectioned, excluded, failed)
        lines.append(f"{titles[group_key]}:")
        lines.extend(_render(records, cov, keys, sectioned))
        lines.append("")
    if lines:
        with open(os.path.join(output_root, "metrics_report.txt"), "w") as f:
            f.write("\n".join(lines).strip() + "\n")


# Metric keys written by evaluate.py / evaluate_pinn.py
SECTIONED_METRIC_KEYS = [
    "Diffusivity Metrics|MSE",
    "Diffusivity Metrics|MAE",
    "Diffusivity Metrics|PSNR",
    "Diffusivity Metrics|SSIM",
    "Diffusivity Metrics|IOU (k<0.03)",
    "Surface Temperature Metrics|MSE (all)",
    "Surface Temperature Metrics|MAE (all)",
    "Surface Temperature Metrics|MSE (final)",
    "Surface Temperature Metrics|PSNR (all)",
    "Surface Temperature Metrics|PSNR (final)",
    "Surface Temperature Metrics|SSIM (final)",
]

# Metric keys written by evaluate_unet.py
UNET_METRIC_KEYS = [
    "k_mse",
    "k_mae",
    "k_max_error",
    "k_psnr",
    "k_ssim",
    "k_iou",
    "surface_mse",
    "surface_psnr_final",
    "surface_psnr_overall",
]
