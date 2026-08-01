from __future__ import annotations

import argparse

import torch

from mira.inference.loading import load_world_model
from mira.inference.rollout import measure_rollout_speed
from mira.training.checkpoints import resolve_checkpoint
from mira.world_model.config import WorldModelInferenceConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--n-diffusion-steps", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--schedule-type", type=str, default="linear", choices=["linear", "linear_quadratic"])
    parser.add_argument("--n-frames", type=int, default=32, help="latent frames to unroll")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--noise-level",
        type=lambda s: None if s.lower() == "none" else float(s),
        default=0.2,
        help="kv-cache noise level; 'none' merges the cache update into the last diffusion step.",
    )
    return parser.parse_args()


CONFIG_FILENAME = "world_model_config.yaml"


def main() -> None:
    from omegaconf import OmegaConf

    from mira.data.training_loader import create_loader

    args = parse_args()
    device = torch.device("cuda")
    checkpoint = resolve_checkpoint(args.checkpoint).resolve()
    # The dataset index lives in the run config saved two dirs above the checkpoint (the output dir).
    cfg = OmegaConf.load(checkpoint.parents[1] / CONFIG_FILENAME)
    model, _ = load_world_model(checkpoint, device=device)
    model.eval()
    if args.compile:
        model.world_model.compile()

    frame_size = cfg.dataset.get("frame_size")
    dataloader = create_loader(
        index_path=cfg.dataset.test_index,
        clip_len=model.config.n_context_frames + args.n_frames * model.temporal_downsampling,
        target_fps=model.config.video.fps,
        n_players=getattr(model, "n_players", 1),
        batch_size=1,
        num_workers=0,
        valid_keys=list(model.config.actions.valid_keys),
        action_fps=model.config.actions.target_fps,
        frame_size=tuple(frame_size) if frame_size is not None else None,
        seed=38,
        infinite=True,
    )
    batch, _ = next(iter(dataloader))
    batch = batch.to(device)

    latent_fps = model.config.video.fps / model.temporal_downsampling
    print(
        f"latent fps: {latent_fps:.2f} (video {model.config.video.fps} fps, "
        f"temporal downsampling {model.temporal_downsampling})"
    )

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # Warmup (compile / cudnn autotune) with the largest step count.
        warm_cfg = WorldModelInferenceConfig(
            n_diffusion_steps=max(args.n_diffusion_steps),
            schedule_type=args.schedule_type,
            noise_level=args.noise_level,
        )
        measure_rollout_speed(model, batch, warm_cfg, n_frames=4)

        for steps in args.n_diffusion_steps:
            config = WorldModelInferenceConfig(
                n_diffusion_steps=steps,
                schedule_type=args.schedule_type,
                noise_level=args.noise_level,
            )
            result = measure_rollout_speed(model, batch, config, n_frames=args.n_frames)
            s_per_frame = result["denoise_ms_per_latent_frame"] / 1000
            print(
                f"steps={steps}: {result['denoise_ms_per_latent_frame']:7.1f} ms/latent-frame "
                f"({result['denoise_latent_fps']:6.2f} latent fps, "
                f"{(1 / s_per_frame) / latent_fps:5.2f}x realtime)"
            )


if __name__ == "__main__":
    main()
