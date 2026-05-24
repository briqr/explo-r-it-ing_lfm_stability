"""
Evaluate similarity between ImageNet generated samples.
We use DINOv2 feature cosine similarity:

This is a paired metric: sample i in both files should come from the same
initial noise seed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm


def build_samples_path(results_dir: str, ckpt: str, epoch: str, seed: int | None = None) -> Path:
    epoch_stem = Path(epoch).stem

    if seed is None:
        vis_dir = f"vis_{epoch_stem}"
    else:
        vis_dir = f"vis_{epoch_stem}_seed{seed}"

    return Path(results_dir) / ckpt / "vis" / vis_dir / "generated_samples.pth"


def load_samples(path: str | Path, device: torch.device) -> torch.Tensor:
    path = Path(path)
    print("Loading samples:", path)

    x = torch.load(path, map_location="cpu", weights_only=False)

    # Expected shapes: [N, 3, H, W]
    if x.ndim != 4:
        raise ValueError(f"Expected samples with shape [N, 3, H, W], got {tuple(x.shape)}")

    # Convert to float [0, 1].
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    else:
        x = x.float()
        if x.min() < 0:
            x = x * 0.5 + 0.5
        elif x.max() > 1.5:
            x = x / 255.0

    return x.clamp(0, 1).to(device)


def imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def extract_dino_features(
    images: torch.Tensor,
    model: torch.nn.Module,
    batch_size: int,
) -> torch.Tensor:
    feats = []

    for i in tqdm(range(0, len(images), batch_size), desc="Extracting DINO features"):
        x = images[i : i + batch_size]

        # DINOv2 ViT-B/14 usually expects 224x224 ImageNet-normalized images.
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
        x = imagenet_normalize(x)

        f = model(x)

        # torch.hub DINOv2 usually returns [B, D].
        # Keep this guard in case a different wrapper returns a dict.
        if isinstance(f, dict):
            if "x_norm_clstoken" in f:
                f = f["x_norm_clstoken"]
            elif "x_prenorm" in f:
                f = f["x_prenorm"]
            else:
                raise ValueError(f"Unknown DINO output keys: {f.keys()}")

        feats.append(f.float().cpu())

    feats = torch.cat(feats, dim=0)
    feats = F.normalize(feats, dim=1)
    return feats


def load_or_extract_features(
    samples_path: Path,
    images: torch.Tensor,
    model: torch.nn.Module,
    batch_size: int,
    model_name: str,
    recompute: bool = False,
) -> torch.Tensor:
    cache_path = samples_path.with_suffix(f".{model_name}.features.pt")

    if cache_path.exists() and not recompute:
        print("Loading cached features:", cache_path)
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    feats = extract_dino_features(images, model, batch_size=batch_size)
    torch.save(feats, cache_path)
    print("Saved features:", cache_path)
    return feats


def paired_cosine(
    feats1: torch.Tensor,
    feats2: torch.Tensor,
    n: int | None = None,
    unrelated: bool = False,
    seed: int = 0,
) -> tuple[float, float, torch.Tensor]:
    n = min(len(feats1), len(feats2)) if n is None else min(n, len(feats1), len(feats2))

    feats1 = F.normalize(feats1[:n], dim=1)
    feats2 = F.normalize(feats2[:n], dim=1)

    if unrelated:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g)
        feats2 = feats2[perm]

    sims = (feats1 * feats2).sum(dim=1)
    return sims.mean().item(), sims.std(unbiased=False).item(), sims


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    samples_path1 = Path(args.samples1) if args.samples1 else build_samples_path(
        args.results_dir1,
        args.ckpt1,
        args.epoch1,
        seed=args.seed if args.use_seed_dir else None,
    )

    samples_path2 = Path(args.samples2) if args.samples2 else build_samples_path(
        args.results_dir2,
        args.ckpt2,
        args.epoch2,
        seed=args.seed if args.use_seed_dir else None,
    )

    images1 = load_samples(samples_path1, device)
    images2 = load_samples(samples_path2, device)

    print("samples1:", tuple(images1.shape))
    print("samples2:", tuple(images2.shape))

    print("Loading DINOv2:", args.dino_model)
    model = torch.hub.load("facebookresearch/dinov2", args.dino_model)
    model = model.to(device).eval()

    feats1 = load_or_extract_features(
        samples_path1,
        images1,
        model,
        batch_size=args.batch_size,
        model_name=args.dino_model,
        recompute=bool(args.recompute),
    )

    feats2 = load_or_extract_features(
        samples_path2,
        images2,
        model,
        batch_size=args.batch_size,
        model_name=args.dino_model,
        recompute=bool(args.recompute),
    )

    mean, std, sims = paired_cosine(
        feats1,
        feats2,
        n=args.n,
        unrelated=bool(args.unrelated),
        seed=args.seed,
    )

    metric_name = "dino_unmatched_cosine" if args.unrelated else "dino_paired_cosine"

    out = {
        "metric": metric_name,
        "mean": mean,
        "std": std,
        "n": int(len(sims)),
        "dino_model": args.dino_model,
        "samples1": str(samples_path1),
        "samples2": str(samples_path2),
        "unrelated": bool(args.unrelated),
    }

    print(json.dumps(out, indent=2))

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print("Saved:", out_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--samples1", type=str, default="")
    parser.add_argument("--samples2", type=str, default="")

    # ...or pass experiment names.
    parser.add_argument("--results-dir1", type=str, default="imagenet_models")
    parser.add_argument("--results-dir2", type=str, default="imagenet_models")
    parser.add_argument("--ckpt1", type=str, default="")
    parser.add_argument("--ckpt2", type=str, default="")
    parser.add_argument("--epoch1", type=str, default="0200000.pt")
    parser.add_argument("--epoch2", type=str, default="0200000.pt")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-seed-dir", type=int, default=0)

    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)

    parser.add_argument("--dino-model", type=str, default="dinov2_vitb14")
    parser.add_argument("--recompute", type=int, default=0)

    # For unrelated baseline.
    parser.add_argument("--unrelated", type=int, default=0)

    parser.add_argument("--out-json", type=str, default="")

    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())