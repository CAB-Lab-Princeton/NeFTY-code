"""Fast smoke tests (CPU-friendly, a couple of minutes in total).

    pytest -q tests/test_smoke.py

Everything runs on tiny grids. Tests use the GPU if one is visible; set
CUDA_VISIBLE_DEVICES="" to force CPU.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from network.sim.simulator import DifferentiableHeatSimulator  # noqa: E402
from network.trainer import ThermoNeRFImplicitTrainer  # noqa: E402
from network.trainer_pinn import ThermoNERFPINNTrainer  # noqa: E402
from network.net.thermal_unet import ThermalUNet  # noqa: E402
from dataset.synthetic import (generate_and_save_dataset, generate_synthetic_cases,  # noqa: E402
                               load_synthetic_dataset, phiflow_available, run_custom_simulation)
from utils.batch import build_default_tasks, select_tasks, setting_of_index  # noqa: E402

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RES = (8, 8, 4)
DOMAIN = (10.0, 10.0, 1.0)
DT = 0.05
NUM_STEPS = 4


def _sim(jacobi_iters=50, dtype=torch.float32):
    return DifferentiableHeatSimulator(domain_size=DOMAIN, resolution=RES, dt=DT,
                                       num_steps=NUM_STEPS, jacobi_iters=jacobi_iters, device=DEVICE)


def _defect_field(dtype=torch.float32):
    k = torch.full(RES, 0.15, dtype=dtype, device=DEVICE)
    k[3:5, 3:5, 1:3] = 0.01
    return k


# ---------------------------------------------------------------------------
# Simulator: forward + adjoint gradient
# ---------------------------------------------------------------------------

def test_simulator_forward_shapes_and_finiteness():
    sim = _sim()
    T0 = sim.create_gaussian_heat_source(center=(5.0, 5.0, 0.8), intensity=100.0, radius=0.5)
    surf = sim(_defect_field(), T0)
    assert surf.shape == (NUM_STEPS + 1, RES[0], RES[1])
    assert torch.isfinite(surf).all()
    # Diffusion with zero-flux z and periodic x/y boundaries: the peak decays.
    assert surf[-1].max() < surf[0].max()


def test_adjoint_gradient_matches_bptt_and_finite_differences():
    # Converged inner solves make the adjoint gradient match the BPTT gradient.
    sim = _sim(jacobi_iters=300)
    T0 = sim.create_gaussian_heat_source(center=(5.0, 5.0, 0.8), intensity=100.0, radius=0.5).double()
    target = sim(_defect_field(torch.float64), T0).detach() * 1.05

    def loss_fn(k, use_adjoint):
        return F.mse_loss(sim(k, T0, use_adjoint=use_adjoint), target)

    k0 = torch.full(RES, 0.1, dtype=torch.float64, device=DEVICE)
    k_bptt = k0.clone().requires_grad_(True)
    loss_fn(k_bptt, False).backward()
    k_adj = k0.clone().requires_grad_(True)
    loss_fn(k_adj, True).backward()

    g_bptt, g_adj = k_bptt.grad, k_adj.grad
    assert torch.isfinite(g_adj).all() and g_adj.abs().max() > 0
    rel = (g_adj - g_bptt).norm() / g_bptt.norm()
    assert rel < 1e-4, f"adjoint vs BPTT relative error {rel.item():.2e}"

    # Directional finite-difference check of the adjoint gradient
    gen = torch.Generator().manual_seed(0)
    direction = torch.randn(RES, generator=gen, dtype=torch.float64).to(DEVICE)
    eps = 1e-6
    # Keep both perturbed fields alive: the simulator caches stencil coefficients for
    # no-grad calls keyed on the tensor's storage.
    k_plus, k_minus = k0 + eps * direction, k0 - eps * direction
    with torch.no_grad():
        fd = (loss_fn(k_plus, False) - loss_fn(k_minus, False)) / (2 * eps)
    analytic = (g_adj * direction).sum()
    assert abs(fd.item() - analytic.item()) <= 1e-4 * abs(fd.item()) + 1e-12


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def test_case_generator_and_category_schedule():
    np.random.seed(0)
    k_list, t0_list = generate_synthetic_cases(resolution=RES, domain_size=DOMAIN, num_samples=10, device=DEVICE)
    assert len(k_list) == 10 and k_list[0].shape == RES
    for k in k_list:
        assert float(k.min()) >= 0.005 and float(k.max()) <= 0.2
    # Default batch task list = the full 1,000-sample benchmark
    tasks = build_default_tasks()
    assert [t.eval_idx for t in tasks] == list(range(1000))
    counts = {}
    for t in tasks:
        counts[t.setting_name] = counts.get(t.setting_name, 0) + 1
    assert counts == {"1_defect": 200, "2_defect": 200, "3_defect": 201, "4_defect": 199,
                      "3_layer": 101, "4_layer": 99}
    assert sum(v for k, v in counts.items() if k.endswith("defect")) == 800
    assert len(select_tasks(None, None, 8)) == 8
    assert [t.eval_idx for t in select_tasks([0, 7], None, 8)] == [0, 7]


def test_generate_and_load_dataset(tmp_path):
    np.random.seed(0)
    torch.manual_seed(0)
    generate_and_save_dataset(output_dir=str(tmp_path), num_samples=2, resolution=RES, domain_size=DOMAIN,
                              num_steps=NUM_STEPS, dt=DT, visualize=False, device=DEVICE)
    data = load_synthetic_dataset(str(tmp_path))
    assert data["k_gt"].shape == (2,) + RES
    assert data["surface_gt"].shape == (2, NUM_STEPS + 1, RES[0], RES[1])
    assert data["time_steps"] == NUM_STEPS + 1
    assert data["dt"] == DT and data["num_steps"] == NUM_STEPS
    if phiflow_available():
        assert torch.equal(data["surface_gt"], data["surface_gt_phiflow"])


# ---------------------------------------------------------------------------
# NeFTY / Grid Opt. trainer
# ---------------------------------------------------------------------------

def _nefty_args(**overrides):
    args = dict(
        netdepth=6, netwidth=32, netchunk=1 << 20, multires=4, i_embed=0, anneal_iters=4,
        lrate=1e-3, lrate_decay=500, N_iters=5,
        domain_x=DOMAIN[0], domain_y=DOMAIN[1], domain_z=DOMAIN[2],
        resolution_x=RES[0], resolution_y=RES[1], resolution_z=RES[2],
        dt=DT, num_steps=NUM_STEPS, jacobi_iters=20,
        use_adjoint=True, use_compile=False,
        heat_source_x=5.0, heat_source_y=5.0, heat_source_z=0.8, heat_intensity=100.0, heat_radius=0.5,
        use_grid=False, mean_type="harmonic", output_activation="sigmoid", use_fa=True,
        no_pe=False, no_sl=False, mlp_activation="relu",
        min_diffusivity=0.003, max_diffusivity=0.25,
        reg_type="tv", reg_weight=1e-3, num_loss_frames=512,
    )
    args.update(overrides)
    return argparse.Namespace(**args)


def _observations():
    sim = _sim()
    T0 = sim.create_gaussian_heat_source(center=(5.0, 5.0, 0.8), intensity=100.0, radius=0.5)
    with torch.no_grad():
        return sim(_defect_field(), T0).cpu()


@pytest.mark.parametrize("use_grid", [False, True], ids=["nefty", "grid_opt"])
def test_nefty_and_grid_opt_train_steps(tmp_path, use_grid):
    torch.manual_seed(0)
    args = _nefty_args(use_grid=use_grid)
    trainer = ThermoNeRFImplicitTrainer(args)
    surface = _observations()
    before = [p.detach().clone() for p in trainer.model.parameters()]
    losses = [trainer.train_step(surface)["loss"] for _ in range(args.N_iters)]
    assert all(np.isfinite(losses))
    changed = any(not torch.equal(b, p) for b, p in zip(before, trainer.model.parameters()))
    assert changed
    k = trainer.predict_diffusivity()
    assert k.shape == RES
    k = k.detach()
    assert float(k.min()) >= args.min_diffusivity and float(k.max()) <= args.max_diffusivity
    ckpt = tmp_path / "ckpt.tar"
    trainer.save_checkpoint(str(ckpt))
    fresh = ThermoNeRFImplicitTrainer(args)
    fresh.load_checkpoint(str(ckpt))
    assert fresh.global_step == args.N_iters


# ---------------------------------------------------------------------------
# PINN baseline
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("normalize_coords,use_gradnorm", [(False, False), (True, True)],
                         ids=["fixed_weights", "gradnorm"])
def test_pinn_train_steps(normalize_coords, use_gradnorm):
    torch.manual_seed(0)
    args = _nefty_args(netdepth=6, netwidth=32, lrate=1e-4, N_iters=6,
                       lambda_data=100.0, lambda_ic=100.0, lambda_pde=0.01,
                       lambda_pde_warmup=0, lambda_pde_ramp=0, lambda_pde_start=0.0,
                       use_gradnorm=use_gradnorm, gradnorm_alpha=0.5, gradnorm_warmup=2,
                       gradnorm_update_every=2, n_pde=256, n_ic=64, n_data=64,
                       normalize_coords=normalize_coords,
                       mlp_activation="softplus" if normalize_coords else "relu")
    trainer = ThermoNERFPINNTrainer(args)
    surface = _observations().to(trainer.device)
    for _ in range(args.N_iters):
        metrics = trainer.train_step(surface)
    for key in ("loss", "loss_data", "loss_ic", "loss_pde"):
        assert np.isfinite(metrics[key]), key
    assert metrics["grad_norm_diff"] > 0
    if use_gradnorm:
        total = metrics["lambda_data"] + metrics["lambda_ic"] + metrics["lambda_pde"]
        assert abs(total - (100.0 + 100.0 + 0.01)) < 1e-3


# ---------------------------------------------------------------------------
# U-Net baseline
# ---------------------------------------------------------------------------

def test_unet_forward_backward():
    torch.manual_seed(0)
    model = ThermalUNet(min_diffusivity=0.003, max_diffusivity=0.25).to(DEVICE)
    x = torch.randn(2, 1, 11, 16, 16, device=DEVICE)
    out = model(x)
    assert out.shape == (2, 1, 16, 16, 16)
    assert float(out.min()) >= 0.003 and float(out.max()) <= 0.25
    loss = F.mse_loss(out, torch.full_like(out, 0.1))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


# ---------------------------------------------------------------------------
# End-to-end CLI: train.py -> evaluate.py and train_pinn.py -> evaluate_pinn.py
# ---------------------------------------------------------------------------

def _tiny_dataset(path):
    np.random.seed(1)
    k_list, t0_list = generate_synthetic_cases(resolution=RES, domain_size=DOMAIN, num_samples=2, device=DEVICE)
    surfaces = run_custom_simulation(k_list, t0_list, RES, DOMAIN, NUM_STEPS, DT, device=DEVICE)
    torch.save({
        "k_gt": torch.stack(k_list), "t0_gt": torch.stack(t0_list),
        "surface_gt": torch.stack(surfaces), "surface_gt_ours": torch.stack(surfaces),
        "resolution": {"x": RES[0], "y": RES[1], "z": RES[2]}, "time_steps": NUM_STEPS + 1,
    }, path)


COMMON_CLI = [
    "--resolution_x", str(RES[0]), "--resolution_y", str(RES[1]), "--resolution_z", str(RES[2]),
    "--dt", str(DT), "--num_steps", str(NUM_STEPS), "--N_iters", "4", "--i_print", "2",
    "--netdepth", "6", "--netwidth", "32", "--multires", "4", "--anneal_iters", "2",
    "--use_compile", "False", "--disable_tqdm",
]


def _run(cmd, cwd):
    proc = subprocess.run([sys.executable] + cmd, cwd=cwd, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return proc.stdout


@pytest.mark.parametrize("config", ["configs/quick/nefty.txt", "configs/quick/grid_opt.txt"])
def test_cli_train_and_evaluate(tmp_path, config):
    data = tmp_path / "dataset.pt"
    _tiny_dataset(data)
    _run(["train.py", "--config", config, "--datadir", str(data), "--basedir", str(tmp_path),
          "--expname", "run", "--eval_idx", "1", "--jacobi_iters", "20",
          "--early_stop_patience", "2"] + COMMON_CLI, REPO)
    _run(["evaluate.py", "--checkpoint", str(tmp_path / "run" / "best.tar")], REPO)
    metrics = (tmp_path / "run" / "output_visualization" / "metrics.txt").read_text()
    assert "IOU (k<0.03)" in metrics and "PSNR (final)" in metrics


@pytest.mark.parametrize("config", ["configs/quick/pinn.txt", "configs/quick/pinn_gradnorm.txt"])
def test_cli_pinn_train_and_evaluate(tmp_path, config):
    data = tmp_path / "dataset.pt"
    _tiny_dataset(data)
    _run(["train_pinn.py", "--config", config, "--datadir", str(data), "--basedir", str(tmp_path),
          "--expname", "run", "--eval_idx", "0", "--n_pde", "256", "--n_ic", "64", "--n_data", "64"]
         + COMMON_CLI, REPO)
    _run(["evaluate_pinn.py", "--checkpoint", str(tmp_path / "run" / "final.tar")], REPO)
    assert (tmp_path / "run" / "output_visualization" / "metrics.txt").exists()


def test_download_plan_paths():
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import download_data
    plan = download_data.build_plan(["all"], root="/tmp/nefty")
    assert [local for _, local in plan] == [
        "/tmp/nefty/data/synthetic_1000/dataset.pt",
        "/tmp/nefty/data/synthetic_1000_nodefect/dataset.pt",
        "/tmp/nefty/checkpoints/unet_sound_only.tar",
    ]
    for remote, local in plan:
        assert local.endswith(remote)
    assert download_data.build_plan(["unet"], root="/r") == [
        ("checkpoints/unet_sound_only.tar", "/r/checkpoints/unet_sound_only.tar")]


# ---------------------------------------------------------------------------
# Data-generation failure modes, PINN time convention, batch helpers
# ---------------------------------------------------------------------------

def test_generator_requires_phiflow_unless_disabled(tmp_path, monkeypatch):
    import dataset.synthetic as synth
    monkeypatch.setattr(synth, "phiflow_available", lambda: False)
    with pytest.raises(ImportError):
        synth.generate_and_save_dataset(output_dir=str(tmp_path / "a"), num_samples=2, resolution=RES,
                                        domain_size=DOMAIN, num_steps=NUM_STEPS, dt=DT,
                                        visualize=False, device=DEVICE)
    synth.generate_and_save_dataset(output_dir=str(tmp_path / "b"), num_samples=2, resolution=RES,
                                    domain_size=DOMAIN, num_steps=NUM_STEPS, dt=DT, visualize=False,
                                    use_phiflow=False, device=DEVICE)
    data = load_synthetic_dataset(str(tmp_path / "b"))
    assert "surface_gt_phiflow" not in data
    assert torch.equal(data["surface_gt"], data["surface_gt_ours"])


def test_pinn_frame_times_start_at_zero():
    args = _nefty_args(lambda_data=1.0, lambda_ic=1.0, lambda_pde=1.0, n_pde=8, n_ic=8, n_data=512,
                       normalize_coords=False, mlp_activation="relu")
    trainer = ThermoNERFPINNTrainer(args)
    frames = torch.arange(NUM_STEPS + 1, dtype=torch.float32, device=trainer.device)
    surface = frames[:, None, None].expand(NUM_STEPS + 1, RES[0], RES[1]).contiguous()
    points, values = trainer.sample_data_points(512, surface)
    # Frame i (value i) is observed at t = i * dt; frame 0 is the initial condition at t = 0.
    assert torch.allclose(points[:, 3], values[:, 0] * DT)
    assert float(points[:, 3].min()) == 0.0
    assert float(points[:, 3].max()) <= NUM_STEPS * DT + 1e-6


def test_task_selection_validation():
    with pytest.raises(ValueError):
        select_tasks([0, 8], None, 8)       # out of range
    with pytest.raises(ValueError):
        select_tasks([-1], None, 8)         # negative
    with pytest.raises(ValueError):
        select_tasks([1, 1], None, 8)       # duplicate
    with pytest.raises(ValueError):
        select_tasks(None, None, 0)         # non-positive dataset size


def test_summaries_count_missing_and_nonfinite(tmp_path):
    from utils.batch import Task, write_summaries, SECTIONED_METRIC_KEYS
    values = {0: "0.5", 1: "nan"}            # idx 2 has no metrics.txt
    for idx, psnr in values.items():
        d = tmp_path / "1_defect" / f"idx_{idx}" / "output_visualization"
        d.mkdir(parents=True)
        d.joinpath("metrics.txt").write_text(
            f"Diffusivity Metrics:\nMSE: 0.1\nPSNR: {psnr}\n\nSurface Temperature Metrics:\nMSE (all): 0.2\n")
    tasks = [Task("1_defect", i) for i in range(3)]
    write_summaries(str(tmp_path), tasks, SECTIONED_METRIC_KEYS, sectioned=True)
    report = (tmp_path / "metrics_report.txt").read_text()
    assert report.startswith("Not layered (others):\nWARNING")
    assert "Specimens: requested 3, evaluated 2, missing 1, failed 0, excluded 0" in report
    assert "PSNR: 0.50 ± 0.00" in report and "1 non-finite excluded" in report
    assert "MSE: 0.100000 ± 0.000000" in report


def test_interrupted_batch_exits_nonzero(tmp_path):
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from utils.batch import Task, run_tasks\n"
        "tasks = [Task('1_defect', i) for i in range(4)]\n"
        "run_tasks(tasks, lambda t, g: ([sys.executable, '-c', 'import time; time.sleep(60)'],"
        " %r + '/log_%%d.txt' %% t.eval_idx), ['0'], 2)\n"
        "print('NOT REACHED')\n" % (REPO, str(tmp_path))
    )
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    import time
    time.sleep(3)
    proc.terminate()
    out, _ = proc.communicate(timeout=60)
    assert proc.returncode == 143, out
    assert "NOT REACHED" not in out


def test_generators_reject_nonfinite_phiflow_output(tmp_path, monkeypatch):
    import dataset.synthetic as synth
    import dataset.generate_defect_free as gdf

    def bad_phiflow(k_gt_tensor, T0_tensor, domain_size, dt, num_steps):
        out = torch.zeros(k_gt_tensor.shape[0], num_steps + 1, k_gt_tensor.shape[1], k_gt_tensor.shape[2])
        out[0, 0, 0, 0] = float("nan")
        return out.to(k_gt_tensor.device)

    monkeypatch.setattr(synth, "phiflow_available", lambda: True)
    monkeypatch.setattr(synth, "create_synthetic_data_phiflow", bad_phiflow)
    with pytest.raises(RuntimeError):
        synth.generate_and_save_dataset(output_dir=str(tmp_path / "a"), num_samples=2, resolution=RES,
                                        domain_size=DOMAIN, num_steps=NUM_STEPS, dt=DT,
                                        visualize=False, device=DEVICE)
    monkeypatch.setattr(gdf, "phiflow_available", lambda: True)
    monkeypatch.setattr(gdf, "create_synthetic_data_phiflow", bad_phiflow)
    with pytest.raises(RuntimeError):
        gdf.generate_and_save_nodefect_dataset(output_dir=str(tmp_path / "b"), num_samples=2, resolution=RES,
                                               domain_size=DOMAIN, num_steps=NUM_STEPS, dt=DT, device=DEVICE)
    assert not (tmp_path / "a" / "dataset.pt").exists() and not (tmp_path / "b" / "dataset.pt").exists()


def test_eval_task_list_follows_training_selection(tmp_path):
    from utils.batch import Task, resolve_eval_tasks, save_task_list
    (tmp_path / "1_defect" / "idx_5").mkdir(parents=True)   # stale run from an earlier batch
    with pytest.raises(ValueError):
        resolve_eval_tasks(str(tmp_path))                    # no recorded list, no selection
    save_task_list(str(tmp_path), [Task("1_defect", 0), Task("3_layer", 7)], dataset_size=8)
    assert resolve_eval_tasks(str(tmp_path)) == ([Task("1_defect", 0), Task("3_layer", 7)], 8)
    # An --indices override reuses the recorded dataset size for the category mapping
    assert resolve_eval_tasks(str(tmp_path), indices=[7])[0] == [Task("3_layer", 7)]
    assert resolve_eval_tasks(str(tmp_path), indices=[7], dataset_size=1000)[0] == [Task("1_defect", 7)]
    assert resolve_eval_tasks(str(tmp_path), all_found=True)[0] == [Task("1_defect", 5)]


def test_dry_run_metrics_only_writes_nothing(tmp_path):
    out = tmp_path / "batch"
    d = out / "1_defect" / "idx_0" / "output_visualization"
    d.mkdir(parents=True)
    d.joinpath("metrics.txt").write_text("Diffusivity Metrics:\nMSE: 0.1\n")
    _run(["evaluate_batch.py", "--output_root", str(out), "--metrics_only", "--dry_run",
          "--indices", "0", "--dataset_size", "8"], REPO)
    assert not (out / "metrics_report.txt").exists()
    _run(["evaluate_unet_batch.py", "--output_root", str(out), "--metrics_only", "--dry_run", "--all_found"], REPO)
    assert not (out / "metrics_report.txt").exists()


def test_failed_evaluations_are_not_aggregated(tmp_path):
    from utils.batch import Task, write_summaries, SECTIONED_METRIC_KEYS
    for idx in (0, 1):  # both have a metrics.txt, but idx 1 is stale from an earlier run
        d = tmp_path / "1_defect" / f"idx_{idx}" / "output_visualization"
        d.mkdir(parents=True)
        d.joinpath("metrics.txt").write_text(f"Diffusivity Metrics:\nMSE: {0.1 * (idx + 1)}\n")
    write_summaries(str(tmp_path), [Task("1_defect", 0), Task("1_defect", 1)], SECTIONED_METRIC_KEYS,
                    sectioned=True, failed_tasks=[Task("1_defect", 1)])
    report = (tmp_path / "metrics_report.txt").read_text()
    assert "WARNING" in report
    assert "Specimens: requested 2, evaluated 1, missing 0, failed 1, excluded 0" in report
    assert "MSE: 0.100000 ± 0.000000" in report


def test_dry_run_preview_checkpoint_names(tmp_path):
    from utils.batch import Task, save_task_list
    out = tmp_path / "batch"
    save_task_list(str(out), [Task("1_defect", 0)], dataset_size=8)
    nefty = _run(["evaluate_batch.py", "--output_root", str(out), "--dry_run", "--gpus", "0"], REPO)
    pinn = _run(["evaluate_pinn_batch.py", "--output_root", str(out), "--dry_run", "--gpus", "0"], REPO)
    assert "idx_0/best.tar" in nefty and "evaluate.py" in nefty
    assert "idx_0/final.tar" in pinn and "evaluate_pinn.py" in pinn


def test_recorded_dataset_size_survives_all_found_and_none(tmp_path):
    from utils.batch import Task, load_task_list, resolve_eval_tasks, save_task_list
    (tmp_path / "3_layer" / "idx_7").mkdir(parents=True)
    save_task_list(str(tmp_path), [Task("3_layer", 7)], dataset_size=8)
    tasks, size = resolve_eval_tasks(str(tmp_path), all_found=True)
    assert tasks == [Task("3_layer", 7)] and size == 8
    save_task_list(str(tmp_path), tasks, size)
    save_task_list(str(tmp_path), tasks, None)          # unknown size keeps the recorded one
    assert load_task_list(str(tmp_path))[1] == 8


def test_wrapper_reuses_recorded_dataset_size(tmp_path):
    from utils.batch import Task, save_task_list
    out = tmp_path / "batch"
    save_task_list(str(out), [Task("3_layer", 7)], dataset_size=8)
    env = dict(os.environ, DRY_RUN="1", INDICES="7", OUTPUT_ROOT=str(out), PYTHON=sys.executable,
               CONFIG="configs/quick/nefty.txt", DATA_DIR="./data/synthetic_test", GPUS="0")
    env.pop("DATASET_SIZE", None)
    proc = subprocess.run(["bash", "scripts/train_batch.sh"], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "--dataset_size" not in proc.stdout
    assert "3_layer --expname idx_7" in proc.stdout          # training preview
    assert "3_layer/idx_7/best.tar" in proc.stdout           # evaluation preview
