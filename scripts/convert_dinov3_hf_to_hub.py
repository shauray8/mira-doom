
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("convert_dinov3")

# Output filename MIRA's loader looks for, per variant (mira.codec.dino.DINO_WEIGHT_FILENAMES).
OUT_FILENAME = {
    "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
}


def convert_state_dict(
    hf: dict[str, torch.Tensor], hub_ref: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Remap an HF `Dinov3Model` state dict onto hub keys, using `hub_ref` for config-derived buffers."""
    n_layers = 1 + max(int(k.split(".")[1]) for k in hf if k.startswith("layer."))
    out: dict[str, torch.Tensor] = {}

    # Embeddings / stem / final norm.
    out["cls_token"] = hf["embeddings.cls_token"]  # (1, 1, C)
    out["storage_tokens"] = hf["embeddings.register_tokens"]  # (1, R, C)
    out["mask_token"] = hf["embeddings.mask_token"].reshape(1, -1)  # (1,1,C) -> (1, C)
    out["patch_embed.proj.weight"] = hf["embeddings.patch_embeddings.weight"]
    out["patch_embed.proj.bias"] = hf["embeddings.patch_embeddings.bias"]
    out["norm.weight"] = hf["norm.weight"]
    out["norm.bias"] = hf["norm.bias"]

    for i in range(n_layers):
        h, f = f"blocks.{i}.", f"layer.{i}."
        qw, kw, vw = (hf[f + f"attention.{p}_proj.weight"] for p in ("q", "k", "v"))
        out[h + "attn.qkv.weight"] = torch.cat([qw, kw, vw], dim=0)  # (3C, C)
        # K has no bias in DINOv3 -> zeros for the K slice (bias_mask masks it regardless).
        qb, vb = hf[f + "attention.q_proj.bias"], hf[f + "attention.v_proj.bias"]
        out[h + "attn.qkv.bias"] = torch.cat([qb, torch.zeros_like(qb), vb], dim=0)  # (3C,)
        out[h + "attn.proj.weight"] = hf[f + "attention.o_proj.weight"]
        out[h + "attn.proj.bias"] = hf[f + "attention.o_proj.bias"]
        out[h + "ls1.gamma"] = hf[f + "layer_scale1.lambda1"]
        out[h + "ls2.gamma"] = hf[f + "layer_scale2.lambda1"]
        out[h + "mlp.fc1.weight"] = hf[f + "mlp.up_proj.weight"]
        out[h + "mlp.fc1.bias"] = hf[f + "mlp.up_proj.bias"]
        out[h + "mlp.fc2.weight"] = hf[f + "mlp.down_proj.weight"]
        out[h + "mlp.fc2.bias"] = hf[f + "mlp.down_proj.bias"]
        out[h + "norm1.weight"] = hf[f + "norm1.weight"]
        out[h + "norm1.bias"] = hf[f + "norm1.bias"]
        out[h + "norm2.weight"] = hf[f + "norm2.weight"]
        out[h + "norm2.bias"] = hf[f + "norm2.bias"]

    # Config-derived buffers not present in the HF checkpoint: take them from the reference model.
    for k, v in hub_ref.items():
        if k.endswith("attn.qkv.bias_mask") or k == "rope_embed.periods":
            out[k] = v.clone()

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--hf-repo",
        default="facebook/dinov3-vitl16-pretrain-lvd1689m",
        help="HF repo (gated) holding model.safetensors, or a local path to one.",
    )
    ap.add_argument(
        "--model", default="dinov3_vitl16", choices=list(OUT_FILENAME), help="Hub backbone variant to target."
    )
    ap.add_argument(
        "--out-dir", required=True, help="Dir to write the hub .pth into (set RS_DINO_WEIGHTS_DIR here)."
    )
    ap.add_argument(
        "--skip-numeric-check", action="store_true", help="Skip the transformers feature comparison."
    )
    args = ap.parse_args()

    from safetensors.torch import load_file

    # 1. Load HF weights.
    src = Path(args.hf_repo)
    if (src / "model.safetensors").exists():
        st_path = src / "model.safetensors"
    else:
        from huggingface_hub import hf_hub_download

        st_path = Path(hf_hub_download(args.hf_repo, "model.safetensors"))
    hf_sd = load_file(str(st_path))
    logger.info("Loaded HF state dict: %d tensors from %s", len(hf_sd), st_path)

    # 2. Build a reference hub model (random) for the target key set + config-derived buffers.
    ref = torch.hub.load(
        "facebookresearch/dinov3", args.model, pretrained=False, source="github", trust_repo=True
    )
    hub_ref = ref.state_dict()

    # 3. Convert and load back strict=True (this is the structural-exactness check).
    converted = convert_state_dict(hf_sd, hub_ref)
    missing = set(hub_ref) - set(converted)
    unexpected = set(converted) - set(hub_ref)
    if missing or unexpected:
        raise RuntimeError(f"key mismatch: missing={sorted(missing)[:6]} unexpected={sorted(unexpected)[:6]}")
    ref.load_state_dict(converted, strict=True)
    logger.info("strict=True load OK (%d keys) -> structure is exact.", len(converted))

    # 4. Numerical check vs the HF transformers model (optional but on by default).
    if not args.skip_numeric_check:
        _numeric_check(args.hf_repo, converted, args.model)

    # 5. Write the .pth MIRA's loader expects.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / OUT_FILENAME[args.model]
    torch.save(converted, out_path)
    logger.info("Wrote %s\nSet RS_DINO_WEIGHTS_DIR=%s and train the codec.", out_path, out_dir)


def _numeric_check(hf_repo: str, converted: dict[str, torch.Tensor], model: str) -> None:
    """Compare patch features from the hub model (converted weights) vs the HF `Dinov3Model`."""
    try:
        from transformers import AutoModel
    except Exception:  # noqa: BLE001
        logger.warning("transformers not installed; skipping the numeric check (structure was verified).")
        return

    hub = torch.hub.load("facebookresearch/dinov3", model, pretrained=False, source="github", trust_repo=True)
    hub.load_state_dict(converted, strict=True)
    hub.eval()
    hf = AutoModel.from_pretrained(hf_repo).eval()

    torch.manual_seed(0)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        # Compare RAW per-block patch features (norm=False) against the matching HF hidden state.
        # HF hidden_states[0] is the embeddings, so block i's output is hidden_states[i+1]; both are
        # taken pre-final-norm so the two implementations are compared on equal footing.
        last = _n_layers(converted) - 1
        hub_out = hub.get_intermediate_layers(x, n=[last], reshape=False, norm=False)[0]  # (1, Npatch, C)
        hf_hidden = hf(x, output_hidden_states=True).hidden_states[last + 1]  # (1, Nprefix+Npatch, C)
    n_prefix = hf_hidden.shape[1] - hub_out.shape[1]  # HF keeps cls + register tokens as a prefix
    hf_patches = hf_hidden[:, n_prefix:, :]
    diff = (hub_out - hf_patches).abs()
    rel = diff.mean().item() / (hf_patches.abs().mean().item() + 1e-8)
    logger.info(
        "Numeric check: mean_abs_diff=%.3e  rel=%.3e  (prefix tokens dropped=%d)",
        diff.mean().item(),
        rel,
        n_prefix,
    )
    if rel > 1e-3:
        logger.warning(
            "Relative feature diff is high (%.3e); implementations may differ -- inspect before training.",
            rel,
        )
    else:
        logger.info("Numeric check PASSED: converted hub features match the HF model (rel=%.1e).", rel)


def _n_layers(sd: dict[str, torch.Tensor]) -> int:
    return 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))


if __name__ == "__main__":
    main()
