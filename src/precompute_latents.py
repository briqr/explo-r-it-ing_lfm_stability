"""
Precompute VAE latents for a dataset.
single: suitable for smaller datasets (e.g. CelebA-HQ, FFHQ).
shards: for larger dataset (e.g. ImageNet)
Examples
--------
# CelebA-HQ / FFHQ:
python precompute_latents.py \
  --dataset-name celebhq \
  --split train \
  --image-size 256 \
  --batch-size 128 \
  --save-mode single \
  --save-path /path/to/celebhq_latents.pt

# ImageNet:
torchrun --nproc_per_node=4 precompute_latents.py \
  --dataset-name imagenet \
  --split train \
  --image-size 256 \
  --batch-size 128 \
  --vae-pretrained 0 \
  --save-mode sharded \
  --out-dir /path/to/imagenet_latent_shards
"""

from __future__ import annotations

import argparse
import json
import math
import os
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import train_consts as consts
from dataset.data import get_dataset
from fm_training.model import get_autoencoder


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def setup_dist_for_sharded() -> tuple[int, int, int, torch.device]:
    """Initialize torch distributed only when launched with torchrun."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device


def cleanup_dist() -> None:
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


def save_dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unknown save dtype: {name}")


def resolve_save_path(args: argparse.Namespace) -> str:
    """Use --save-path if provided; otherwise choose a common default from train_consts."""
    if args.save_path:
        return args.save_path

    name = args.dataset_name.lower()

    if "celebhq" in name and args.image_size == 512:
        return PRECOMPUTED_LATENTS_PATH_512

    if "celebhq" in name:
        return PRECOMPUTED_LATENTS_PATH

    if "ffhq" in name:
        return consts.PRECOMPUTED_LATENTS_PATH_FFHQ

    raise ValueError("Please pass --save-path for --save-mode single.")


def infer_encoder_type(vae, requested: str) -> str:
    if requested != "auto":
        return requested
    return "vq_gan_taming" if "vq" in str(type(vae)).lower() else "kl_gan_taming"


def load_vae(args: argparse.Namespace, device: torch.device):
    vae = get_autoencoder(
        dataset_name=args.dataset_name,
        vae_path=args.vae_path,
        encoder_type=args.encoder_type if args.encoder_type != "auto" else "vq_gan_taming",
        is_pretrained=bool(args.vae_pretrained),
        device=device,
    )
    vae.eval()
    encoder_type = infer_encoder_type(vae, args.encoder_type)
    print("Using encoder type:", encoder_type)
    return vae, encoder_type


@torch.no_grad()
def encode_images(
    vae,
    images: torch.Tensor,
    *,
    vae_pretrained: bool,
    encoder_type: str,
) -> torch.Tensor:
    """Encode a batch of images into VAE latents."""
    if vae_pretrained:
        z = vae.encode(images).latent_dist.sample()
    elif encoder_type.startswith("vq"):
        z = vae.encode(images)["quantized"]
    else:
        z = vae.encode(images)["quantized"].sample()
    return z.contiguous()


@torch.no_grad()
def encode_with_optional_microbatch(
    args: argparse.Namespace,
    vae,
    encoder_type: str,
    images: torch.Tensor,
) -> torch.Tensor:
    if args.micro_bs <= 0 or images.size(0) <= args.micro_bs:
        return encode_images(
            vae,
            images,
            vae_pretrained=bool(args.vae_pretrained),
            encoder_type=encoder_type,
        )

    chunks = []
    for start in range(0, images.size(0), args.micro_bs):
        z = encode_images(
            vae,
            images[start : start + args.micro_bs],
            vae_pretrained=bool(args.vae_pretrained),
            encoder_type=encoder_type,
        )
        chunks.append(z.cpu())
        del z

    return torch.cat(chunks, dim=0).to(images.device)


def build_dataset(args: argparse.Namespace):
    return get_dataset(
        name=args.dataset_name,
        split=args.split,
        is_training=(args.split=='train'),
        im_size=args.image_size,
    )


def build_loader(
    args: argparse.Namespace,
    dataset,
    *,
    rank: int = 0,
    world_size: int = 1,
    sharded: bool = False,
):
    sampler = None
    if sharded and world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )
    return loader, sampler


def update_stats(stats: dict[str, float], z: torch.Tensor) -> None:
    z = z.float()
    stats["n"] += z.numel()
    stats["sum"] += float(z.sum().item())
    stats["sumsq"] += float((z * z).sum().item())


def std_from_stats(n: int, s: float, s2: float) -> float:
    mean = s / max(n, 1)
    var = max(0.0, s2 / max(n, 1) - mean * mean)
    return math.sqrt(var)


def precompute_single(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = build_dataset(args)
    loader, _ = build_loader(args, dataset)

    vae, encoder_type = load_vae(args, device)

    zs, ys, ids = [], [], []
    images_out = []

    for batch in tqdm(loader, desc="Encoding images"):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"]
        sample_ids = batch["id"]

        z = encode_with_optional_microbatch(args, vae, encoder_type, images)

        zs.append(z.detach().cpu())
        ys.append(labels.detach().cpu().long())
        ids.append(sample_ids.detach().cpu().long())

        if args.include_images:
            images_out.append(images.detach().cpu())

        del images, z

    latents = torch.cat(zs, dim=0)
    labels = torch.cat(ys, dim=0)
    sample_ids = torch.cat(ids, dim=0)

    package = {
        "latents": latents,
        "labels": labels,
        "ids": sample_ids,
    }

    if args.include_images:
        package["images"] = torch.cat(images_out, dim=0)

    scale = latents.std().item()
    save_path = resolve_save_path(args)
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    torch.save(package, save_path)

    print("Saved precomputed latents to", save_path)
    print("latents shape, labels shape, ids shape", latents.shape, labels.shape, sample_ids.shape)
    print(f"{args.dataset_name} latent std: {scale}")
    print(f"{args.dataset_name} scale_factor candidate 1/std: {1.0 / max(scale, 1e-12)}")


def precompute_sharded(args: argparse.Namespace) -> None:
    rank, world_size, local_rank, device = setup_dist_for_sharded()
    is_main = rank == 0

    if not args.out_dir:
        raise ValueError("Please pass --out-dir for --save-mode sharded.")

    out_dir = Path(args.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[precompute] world_size={world_size}, out_dir={out_dir}")

    if is_dist():
        dist.barrier()
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args)
    loader, _ = build_loader(args, dataset, rank=rank, world_size=world_size, sharded=True)

    vae, encoder_type = load_vae(args, device)
    save_dtype = save_dtype_from_name(args.save_dtype)

    stats = {"n": 0, "sum": 0.0, "sumsq": 0.0}
    z_list, y_list, id_list = [], [], []
    shard_idx = 0

    def buffer_count() -> int:
        return sum(t.shape[0] for t in z_list)

    def flush_shard() -> int:
        nonlocal shard_idx

        if not z_list:
            return 0

        z = torch.cat(z_list, dim=0).to(save_dtype)
        y = torch.cat(y_list, dim=0).long()
        ids = torch.cat(id_list, dim=0).long()

        stem = out_dir / f"r{rank:02d}_{shard_idx:05d}"

        if args.shard_format == "pt":
            torch.save(
                {
                    "latents": z,
                    "labels": y,
                    "ids": ids,
                    "dataset_name": args.dataset_name,
                    "split": args.split,
                    "rank": rank,
                    "shard_idx": shard_idx,
                },
                str(stem) + ".pt",
            )
        else:
            np.save(str(stem) + "_latents.npy", z.cpu().numpy())
            np.save(str(stem) + "_labels.npy", y.cpu().numpy().astype(np.int64))
            np.save(str(stem) + "_ids.npy", ids.cpu().numpy().astype(np.int64))

        n = int(z.shape[0])
        z_list.clear()
        y_list.clear()
        id_list.clear()
        shard_idx += 1
        return n

    total_batches = math.ceil(len(dataset) / max(1, args.batch_size * world_size))
    pbar = tqdm(loader, desc=f"rank {rank}", total=total_batches, disable=not is_main)

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        if args.channels_last:
            images = images.to(memory_format=torch.channels_last)

        labels = batch["label"].detach().cpu().long()
        sample_ids = batch["id"].detach().cpu().long()

        z = encode_with_optional_microbatch(args, vae, encoder_type, images)
        z_cpu = z.detach().cpu()

        update_stats(stats, z_cpu)

        z_list.append(z_cpu.to(save_dtype))
        y_list.append(labels)
        id_list.append(sample_ids)

        if buffer_count() >= args.shard_size:
            flush_shard()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        del images, z, z_cpu

    flush_shard()

    local_stats = torch.tensor(
        [float(stats["n"]), float(stats["sum"]), float(stats["sumsq"])],
        dtype=torch.float64,
        device=device,
    )

    if is_dist():
        dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)

    if is_main:
        n = int(local_stats[0].item())
        s = float(local_stats[1].item())
        s2 = float(local_stats[2].item())
        scale = std_from_stats(n, s, s2)

        if args.shard_format == "pt":
            shard_files = sorted(glob(str(out_dir / "r*.pt")))
            manifest = {"format": "pt", "shards": [os.path.basename(p) for p in shard_files]}
            num_shards = len(manifest["shards"])
        else:
            lat_files = sorted(glob(str(out_dir / "r*_latents.npy")))
            manifest = {"format": "npy", "latents": [os.path.basename(p) for p in lat_files]}
            num_shards = len(manifest["latents"])

        meta = {
            "num_elements": n,
            "std": scale,
            "inverse_std": 1.0 / max(scale, 1e-12),
            "world_size": world_size,
            "dataset_name": args.dataset_name,
            "split": args.split,
            "image_size": args.image_size,
            "encoder_type": encoder_type,
            "vae_path": args.vae_path,
            "save_dtype": args.save_dtype,
            "shard_size": args.shard_size,
            "shard_format": args.shard_format,
        }

        with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(f"[rank0] Global latent std: {scale:.10f}, inverse: {1.0 / max(scale, 1e-12):.10f}")
        print(f"[rank0] Wrote {num_shards} shards to {out_dir}")

    cleanup_dist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="celebhq")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--vae-path", type=str, default="")
    parser.add_argument("--encoder-type", type=str, default="vq_gan_taming")
    parser.add_argument("--vae-pretrained", type=int, default=0)

    parser.add_argument("--save-mode", choices=["single", "sharded"], default="single")
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="")

    parser.add_argument("--include-images", type=int, default=0)
    parser.add_argument("--pin-memory", type=int, default=1)
    parser.add_argument("--save-dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument("--shard-size", type=int, default=64_000)
    parser.add_argument("--shard-format", choices=["pt", "npy"], default="pt")
    parser.add_argument("--micro-bs", type=int, default=0)
    parser.add_argument("--channels-last", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.save_mode == "single":
        precompute_single(args)
    else:
        precompute_sharded(args)


if __name__ == "__main__":
    main()
