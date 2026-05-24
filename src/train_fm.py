# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

"""Refactored DiT, replaces diffusion with latent flow-matching. This is the training entry point.
e.g. runs

torchrun --nproc_per_node=4 train_fm.py \
  --train-mode standard \
  --dataset_name celebhq_precomputed \
  --model DiT-S/2 \
  --transport fm \
  --vae_pretrained 0 \
  --pruning_method balanced_cluster_nearest \
  --pruning_ratio 0.5 \
  --global-batch-size 128 \
  --results-dir results

torchrun --nproc_per_node=4 train_fm.py \
  --train-mode coarse2fine \
  --dataset_name celebhq_precomputed \
  --model DiT-S/2 \
  --model-fine DiT-XL/2 \
  --fine-ckpt results/fine_model_exp/checkpoints/0100000.pt \
  --transport fm \
  --vae_pretrained 0 \
  --pruning_method balanced_cluster_nearest \
  --pruning_ratio 0.5 \
  --t-split 0.7 \
  --seam-weight 0.1 \
  --global-batch-size 128 \
  --results-dir results


"""

from __future__ import annotations

import argparse

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset.data import build_loader, get_dataset, get_dataset_spec, model_data_config_from_spec
from fm_training.model import build_model, build_transport, maybe_load_checkpoint, model_choices
from fm_training.pruning import maybe_prune_dataset
from fm_training.training import train_loop, build_fine_model, train_coarse_to_fine_loop
from fm_training.distributed import cleanup_distributed, setup_distributed
from fm_training.experiment import setup_experiment
from fm_training.model import unwrap_model
from fm_training.repro import seed_everything


def main(args: argparse.Namespace) -> None:
    env = setup_distributed()
    seed = args.global_seed * env.world_size + env.rank
    seed_everything(seed, deterministic=bool(args.deterministic))
    print(f"Starting rank={env.rank}, local_rank={env.local_rank}, seed={seed}, world_size={env.world_size}, ddp={env.ddp}.")

    method = args.pruning_method
    pr = args.pruning_ratio
    inverse = args.inverse > 0.00001
    vae_pretrained = args.vae_pretrained > 0.00001

    diffusion, is_fm, learn_sigma = build_transport(args)
    dataset_spec = get_dataset_spec(args.dataset_name)
    cfg = model_data_config_from_spec(dataset_spec, vae_pretrained=vae_pretrained)
    print(f"**in channels {cfg.in_channels}, scale factor {cfg.scale_factor:.6f}")

    paths, logger, pr = setup_experiment(
        args,
        env,
        is_fm=is_fm,
        inverse=inverse,
        is_conditional=cfg.is_conditional,
        seed=seed
    )
    args.pruning_ratio = pr

    model, ema = build_model(args, cfg, learn_sigma, env.device)
    #in case a resume checkpoint exists
    checkpoint = maybe_load_checkpoint(args, model, ema, logger)
    model = model.to(env.device)
    if env.ddp:
        model = DDP(model, device_ids=[env.local_rank])

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    if checkpoint is not None:
        opt.load_state_dict(checkpoint["opt"])
        del checkpoint

    param_count = sum(param.numel() for param in unwrap_model(model).parameters())
    logger.info(f"DiT Parameters: {param_count:,}")
    print(f"DiT Parameters: {param_count:,}")
    logger.info(f"args: {args}")
    print(f"args: {args}")

    dataset = get_dataset(
        name=args.dataset_name,
        split="train",
        is_training=True,
        im_size=args.image_size,
    )

    dataset = maybe_prune_dataset(
        args,
        dataset,
        cfg,
        paths,
        env,
        logger,
    )
    print("finished building dataset")

    loader, sampler = build_loader(args, dataset, env)
    logger.info(f"Dataset contains {len(dataset):,} images ({args.dataset_name})")
    print(f"Dataset contains {len(dataset):,} images ({args.dataset_name})")

    try:
        if args.train_mode == "standard":
            train_loop(
                args,
                env=env,
                model=model,
                ema=ema,
                opt=opt,
                loader=loader,
                sampler=sampler,
                dataset_len=len(dataset),
                diffusion=diffusion,
                is_fm=is_fm,
                cfg=cfg,
                paths=paths,
                logger=logger,
            )
            logger.info("Done!")
        
        elif args.train_mode == "coarse2fine":
            assert is_fm, "coarse2fine currently only makes sense for flow matching."
            assert args.fine_ckpt, "--fine-ckpt is required for --train-mode coarse2fine"

            fine_model = build_fine_model(
                args=args,
                cfg=cfg,
                learn_sigma=learn_sigma,
                device=env.device,
            )
            train_coarse_to_fine_loop(
                args,
                env=env,
                model=model,
                ema=ema,
                opt=opt,
                loader=loader,
                sampler=sampler,
                dataset_len=len(dataset),
                diffusion=diffusion,
                cfg=cfg,
                paths=paths,
                logger=logger,
                fine_model=fine_model,
            )
    finally:
        cleanup_distributed(env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-dir", type=str)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=model_choices(), default="DiT-XL/2")
    #coarse2fine related hyperparams
    parser.add_argument("--train-mode", type=str, choices=["standard", "coarse2fine"], default="standard",)
    parser.add_argument("--model-fine", type=str, choices=model_choices(), default="DiT-XL/2")
    parser.add_argument("--fine-ckpt", type=str, default="")
    parser.add_argument("--t-split", type=float, default=0.7)
    parser.add_argument("--seam-weight", type=float, default=0.1)
    parser.add_argument("--inversion-steps-per-unit", type=int, default=60)

    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num_clusters", type=int, default=24)
    parser.add_argument("--dataset_name", type=str, default="celebhq_precomputed")
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=500000)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10_000)
    parser.add_argument("--class-dropout-prob", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pruning_method", type=str, default="random")
    parser.add_argument("--pruning_ratio", type=float, default=0)
    parser.add_argument("--inverse", type=int, default=0)
    parser.add_argument("--transport", type=str, default="fm")
    parser.add_argument("--vae_pretrained", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--deterministic", type=int, default=0)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
