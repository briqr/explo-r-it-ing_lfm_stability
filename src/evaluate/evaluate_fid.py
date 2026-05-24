"""Evaluate FID/precision/recall for generated samples.

You can either:
  1) pass --samples-path to evaluate existing samples, or
  2) pass --generate-if-missing 1 to generate them first with the same code as sample_models.py.
"""

from __future__ import annotations

import argparse
import os

import torch

from models import DiT_models
from unet_fm import UNet_models

DiT_models.update(UNet_models)

from inference.sampling import (
    calculate_fid_metrics,
    default_samples_path,
    generate_single_model,
    generate_two_models,
    load_checkpoint,
    load_real_images_uint8,
    load_samples,
    resolve_checkpoint,
    resolve_run_config,
    save_samples,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument("--mode", choices=["single", "coarse2fine"], default="single")

    # Main/coarse checkpoint.
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--epoch", type=str, default="0130000.pt")
    p.add_argument("--results_dir", type=str, default="results")

    # Fine checkpoint for coarse2fine model.
    p.add_argument("--ckpt2", type=str, default="")
    p.add_argument("--epoch2", type=str, default="")
    p.add_argument("--results_dir2", type=str, default="results4")

    # Existing generated samples, or generate them if missing.
    p.add_argument("--samples-path", type=str, default="")
    p.add_argument("--generate-if-missing", type=int, default=0)

    # Fallbacks only. If checkpoint contains args/config, those are used first.
    p.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/2")
    p.add_argument("--model_fine", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    p.add_argument("--dataset_name", type=str, default="celebhq")
    p.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    p.add_argument("--num-classes", type=int, default=1)
    p.add_argument("--transport", type=str, default="fm")
    p.add_argument("--vae_pretrained", type=int, default=0)

    # Sampling options, used only if generating.
    p.add_argument("--cfg-scale", type=float, default=4.0)
    p.add_argument("--num-sampling-steps", type=int, default=250)
    p.add_argument("--fm-steps", type=int, default=60)
    p.add_argument("--sampling-method", type=str, default="euler")
    p.add_argument("--t_split", type=float, default=0.7)

    # VAE, used only if generating.
    p.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    p.add_argument("--vae-path", type=str, default="")
    p.add_argument("--encoder-type", type=str, default="vq_gan_taming")

    # Evaluation.
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_iters", type=int, default=128)
    p.add_argument("--real-batch-size", type=int, default=512)
    return p


def samples_path_for_args(args) -> str:
    suffix = f"_twomodels_seed{args.seed}" if args.mode == "coarse2fine" else f"_seed{args.seed}"
    ckpt_info = resolve_checkpoint(args.results_dir, args.ckpt, args.epoch, suffix=suffix)
    return args.samples_path or default_samples_path(ckpt_info, args.mode)


def load_or_generate_samples(args, device: torch.device, samples_path: str):
    if os.path.exists(samples_path):
        print("loading samples:", samples_path)
        return load_samples(samples_path)

    if not args.generate_if_missing:
        raise FileNotFoundError(
            f"Samples not found: {samples_path}. "
            "Pass --generate-if-missing 1 to create them."
        )

    if args.mode == "single":
        samples, _, _, _ = generate_single_model(args, device)
    else:
        if not args.ckpt2 or not args.epoch2:
            raise ValueError("--mode coarse2fine requires --ckpt2 and --epoch2")
        samples, _, _, _ = generate_two_models(args, device)

    save_samples(samples, samples_path)
    return samples


def main(args: argparse.Namespace) -> None:

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load the main checkpoint only to recover dataset/image_size for real validation images.
    ckpt_info = resolve_checkpoint(args.results_dir, args.ckpt, args.epoch)
    checkpoint = load_checkpoint(ckpt_info.ckpt_path)
    cfg = resolve_run_config(args, checkpoint, args.ckpt)

    samples_path = samples_path_for_args(args)
    samples = load_or_generate_samples(args, device, samples_path).to(device)
    print("samples shape:", tuple(samples.shape))
    # Possible to tweak torch_utility to precompute the real distribution features,
    # then no need to keep loading the real images
    real_images = load_real_images_uint8(
        cfg.dataset_name,
        cfg.image_size,
        device,
        batch_size=args.real_batch_size,
    ).to(device)
    print("real images shape:", tuple(real_images.shape))

    metrics = calculate_fid_metrics(samples, real_images, cfg.dataset_name, cuda=True)
    print(metrics)
    print("model used:", ckpt_info.ckpt_path)
    print("number of samples:", len(samples))
    print("seed:", args.seed)


if __name__ == "__main__":
    main(build_parser().parse_args())
