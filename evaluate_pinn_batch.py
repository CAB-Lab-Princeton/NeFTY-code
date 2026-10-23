"""Evaluate every PINN run under a train_batch.py output root (--script train_pinn.py)
and aggregate the metrics.

    python evaluate_pinn_batch.py --output_root ./logs/output_pinn_<timestamp> --gpus 0,1
"""
from evaluate_batch import run


if __name__ == "__main__":
    run("evaluate_pinn.py", "evaluate_pinn.log", "Batch evaluation for the PINN baseline",
        preview_checkpoint="final.tar")
