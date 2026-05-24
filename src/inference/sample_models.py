"""Sample from one model or a coarse-to-fine pair."""

from __future__ import annotations

import argparse
import os

import torch

from models import DiT_models
from unet_fm import UNet_models

DiT_models.update(UNet_models)

from inference.sampling import (
    default_samples_path,
    generate_single_model,
    generate_two_models,
    save_samples,
    save_visual_grid,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument("--mode", choices=["single", "coarse2fine"], default="single")

    # Main (single model)/coarse checkpoint.
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--epoch", type=str, default="0130000.pt")
    p.add_argument("--results_dir", type=str, default="results")

    # Fine checkpoint for coarse2fine.
    p.add_argument("--ckpt2", type=str, default="")
    p.add_argument("--epoch2", type=str, default="")
    p.add_argument("--results_dir2", type=str, default="results")

    # Fallbacks only. If checkpoint contains args/config, those are used first.
    p.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/2")
    p.add_argument("--model_fine", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    p.add_argument("--dataset_name", type=str, default="celebhq")
    p.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    p.add_argument("--num-classes", type=int, default=1)
    p.add_argument("--transport", type=str, default="fm")
    p.add_argument("--vae_pretrained", type=int, default=0)

    # Sampling.
    p.add_argument("--cfg-scale", type=float, default=4.0)
    p.add_argument("--num-sampling-steps", type=int, default=250) # diffusion step
    p.add_argument("--fm-steps", type=int, default=60) 
    p.add_argument("--sampling-method", type=str, default="euler")
    p.add_argument("--t_split", type=float, default=0.7)

    # VAE.
    p.add_argument("--vae-path", type=str, default="")
    p.add_argument("--encoder-type", type=str, default="vq_gan_taming")

    # Output.
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_iters", type=int, default=2)
    p.add_argument("--samples-path", type=str, default="")
    p.add_argument("--save-images", type=int, default=1)
    p.add_argument("--nrow", type=int, default=6)

    return p


def main(args: argparse.Namespace) -> None:

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if args.mode == "single":
        samples, ckpt_info, cfg, profile = generate_single_model(args, device)
    else:
        if not args.ckpt2 or not args.epoch2:
            raise ValueError("--mode coarse2fine requires --ckpt2 and --epoch2")
        samples, ckpt_info, cfg, profile = generate_two_models(args, device)

    samples_path = args.samples_path or default_samples_path(ckpt_info, args.mode)
    save_samples(samples, samples_path)

    profile.print(
        label=args.mode,
        batch_size=args.batch_size,
        num_iters=args.num_iters,
        num_steps=args.fm_steps if cfg.transport != "diffusion" else args.num_sampling_steps,
    )

    if args.save_images:
        vis_dir = os.path.join(
            ckpt_info.vis_dir,
            f"images_{args.mode}_seed{args.seed}",
        )
        save_visual_grid(samples, vis_dir, nrow=args.nrow)


if __name__ == "__main__":
    main(build_parser().parse_args())
