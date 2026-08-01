import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

RUNS_DIR = os.environ.get("MIRA_RUNS", str(REPO / "ckpts"))
DATA_DIR = os.environ.get("MIRA_DATA", str(REPO / "data"))

DEFAULT_DATASET = "doom"
DATASETS = {
    "doom": {
        "hf_repo": "chrisxx/doom-2players-mp4",
        "root": f"{DATA_DIR}/doom_mira",
        # Explicit, not native: the codec's latent grid is fixed by the trained 384x512 geometry.
        "frame_size": [384, 512],
        "video_fps": 35,
        "n_players_multi": 2,
    },
}


def run_dir(stage: str, dataset: str = DEFAULT_DATASET, tag: str = "") -> str:
    return f"{RUNS_DIR}/{dataset}_{stage}" + (f"_{tag}" if tag else "")


def latest_checkpoint_path(output_dir) -> str | None:
    root = Path(output_dir)
    if not root.is_dir():
        return None
    found = []
    for d in root.glob("checkpoint-*"):
        ckpt = d / "checkpoint.pth"
        if ckpt.is_file():
            try:
                found.append((int(d.name.split("-", 1)[1]), ckpt))
            except ValueError:
                continue
    return str(max(found)[1]) if found else None


def ensure_codec_path() -> None:
    baked = Path("/runs/doom_codec")
    local = Path(RUNS_DIR) / "doom_codec"
    if baked.exists() and baked.resolve() == local.resolve():
        return
    if not local.is_dir():
        print(f"[serve] WARNING: no codec at {local}; the world model will fail to load.")
        return
    try:
        baked.parent.mkdir(parents=True, exist_ok=True)
        if baked.is_symlink() or baked.exists():
            baked.unlink()
        baked.symlink_to(local)
        print(f"[serve] linked {baked} -> {local}")
    except OSError as exc:
        print(f"[serve] WARNING: could not create {baked} ({exc}); "
              f"create it by hand or the model will fail to load.")
