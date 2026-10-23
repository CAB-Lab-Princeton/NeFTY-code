"""Download the NeFTY benchmark data and the pretrained Sound-Only U-Net from Hugging Face.

    python scripts/download_data.py                 # everything
    python scripts/download_data.py --what benchmark
    python scripts/download_data.py --what benchmark defect_free unet

Files land at:
    data/synthetic_1000/dataset.pt           1,000-sample benchmark (homogeneous + layered)
    data/synthetic_1000_nodefect/dataset.pt  defect-free set (Sound-Only U-Net training)
    checkpoints/unet_sound_only.tar          pretrained Sound-Only U-Net
"""
import argparse
import os

# Hugging Face dataset repository hosting the files.
HF_REPO_ID = "n3il666/NeFTY"
HF_REPO_TYPE = "dataset"

# name -> (path inside the HF repo, local path relative to the repository root)
FILES = {
    "benchmark": ("synthetic_1000/dataset.pt", "data/synthetic_1000/dataset.pt"),
    "defect_free": ("synthetic_1000_nodefect/dataset.pt", "data/synthetic_1000_nodefect/dataset.pt"),
    "unet": ("checkpoints/unet_sound_only.tar", "checkpoints/unet_sound_only.tar"),
}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_plan(what, root=REPO_ROOT):
    """List of (remote filename, absolute local destination) for the requested items."""
    names = list(FILES) if (not what or "all" in what) else what
    return [(FILES[n][0], os.path.join(root, FILES[n][1])) for n in names]


def download(plan, repo_id=HF_REPO_ID, revision=None, force=False):
    from huggingface_hub import hf_hub_download

    for remote, local in plan:
        if os.path.exists(local) and not force:
            print(f"[skip] {local} already exists (use --force to re-download)")
            continue
        # hf_hub_download writes to <local_dir>/<remote>, so local_dir is the destination
        # with the remote relative path stripped.
        assert local.replace(os.sep, "/").endswith(remote)
        local_dir = local[: len(local) - len(remote)]
        os.makedirs(local_dir, exist_ok=True)
        print(f"[get ] {repo_id}:{remote} -> {local}")
        hf_hub_download(repo_id=repo_id, filename=remote, repo_type=HF_REPO_TYPE,
                        revision=revision, local_dir=local_dir, force_download=force)


def main():
    parser = argparse.ArgumentParser(description="Download NeFTY data / checkpoints from Hugging Face")
    parser.add_argument("--what", nargs="+", default=["all"], choices=["all"] + list(FILES),
                        help="Items to download (default: all)")
    parser.add_argument("--repo_id", type=str, default=HF_REPO_ID, help="Hugging Face dataset repo id")
    parser.add_argument("--revision", type=str, default=None, help="Branch, tag or commit of the HF repo")
    parser.add_argument("--root", type=str, default=REPO_ROOT, help="Destination root (default: repository root)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--dry_run", action="store_true", help="Print the plan without downloading")
    args = parser.parse_args()

    plan = build_plan(args.what, args.root)
    if args.dry_run:
        for remote, local in plan:
            print(f"{args.repo_id}:{remote} -> {local}")
        return
    download(plan, repo_id=args.repo_id, revision=args.revision, force=args.force)


if __name__ == "__main__":
    main()
