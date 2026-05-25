"""
Cluster features extracted by clip using k-means. Run after executing extract_features_clip.py. 
Precomputed offline.
  #small datasets, full GPU k-means
  #large datasets, chunked/manual k-means

Output format:
  {
    'labels':  LongTensor [1, N],
    'centers': FloatTensor [1, K, D],
    'inertia': float,
    'x_org':   FloatTensor [N, D] if --save-x-org 1,
    'k':       int,
  }

e.g. run
python cluster_features.py \
--features data/celebhq/clip_features/train/clip_features.pth \
--out data/celebhq/clip_features/train/cluster_clip_24.pth
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

def load_one_feature_file(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        # For files like clip_features_with_metadata.pth
        if "features" in obj:
            x = obj["features"].detach().cpu()
        elif "x" in obj:
            x = obj["x"].detach().cpu()

    else: # list object
        xs = []
        for item in obj:
            xs.append(item.detach().cpu())
        x = torch.stack(xs)

    return x.float().flatten(1)


def expand_feature_paths(paths: Iterable[str]) -> list[str]:
    out: list[str] = []
    for p in paths:
        matches = sorted(glob.glob(p))
        if matches:
            out.extend(matches)
        else:
            out.append(p)
    return out


def load_features(paths: list[str]) -> torch.Tensor:
    paths = expand_feature_paths(paths)
    if len(paths) == 0:
        raise ValueError("No feature files were given.")

    chunks = []
    for path in paths:
        print(f"loading {path}")
        chunks.append(load_one_feature_file(path))

    x = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
    print(f"features: shape={tuple(x.shape)}, dtype={x.dtype}")
    return x


def normalize_features(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), p=2, dim=-1, eps=1e-12)


def init_centers(x: torch.Tensor, k: int, seed: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=g)[:k]
    centers = x[idx].to(device=device, dtype=torch.float32)
    return F.normalize(centers, p=2, dim=-1, eps=1e-12)


def assign_full(x: torch.Tensor, centers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    score = x @ centers.t()
    best_score, labels = score.max(dim=1)

    return labels, best_score

def assign_chunked(
    x_cpu: torch.Tensor,
    centers: torch.Tensor,
    chunk_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = x_cpu.shape[0]
    labels_cpu = torch.empty(n, dtype=torch.long)
    scores_cpu = torch.empty(n, dtype=torch.float32)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        x = x_cpu[start:end].to(device=device, dtype=torch.float32)
        labels, scores = assign_full(x, centers)
        labels_cpu[start:end] = labels.cpu()
        scores_cpu[start:end] = scores.cpu()
        print(f"  assigned {end}/{n}")

    return labels_cpu, scores_cpu


def update_centers_full(
    x: torch.Tensor,
    labels: torch.Tensor,
    old_centers: torch.Tensor,
    k: int) -> torch.Tensor:
    d = x.shape[1]
    sums = torch.zeros(k, d, device=x.device, dtype=torch.float32)
    counts = torch.zeros(k, device=x.device, dtype=torch.float32)

    sums.index_add_(0, labels, x)
    counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))

    new_centers = old_centers.clone()
    nonempty = counts > 0
    new_centers[nonempty] = sums[nonempty] / counts[nonempty, None]

    new_centers = F.normalize(new_centers, p=2, dim=-1, eps=1e-12)

    return new_centers


def update_centers_chunked(
    x_cpu: torch.Tensor,
    labels_cpu: torch.Tensor,
    old_centers: torch.Tensor,
    k: int,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    d = x_cpu.shape[1]
    sums = torch.zeros(k, d, device=device, dtype=torch.float32)
    counts = torch.zeros(k, device=device, dtype=torch.float32)

    n = x_cpu.shape[0]
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        x = x_cpu[start:end].to(device=device, dtype=torch.float32)
        labels = labels_cpu[start:end].to(device=device)
        sums.index_add_(0, labels, x)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))

    new_centers = old_centers.clone()
    nonempty = counts > 0
    new_centers[nonempty] = sums[nonempty] / counts[nonempty, None]

    new_centers = F.normalize(new_centers, p=2, dim=-1, eps=1e-12)

    return new_centers


def cluster_full(
    x_cpu: torch.Tensor,
    k: int,
    iterations: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    x = x_cpu.to(device=device, dtype=torch.float32)
    centers = init_centers(x_cpu, k, seed, device)

    for it in range(iterations):
        labels, scores = assign_full(x, centers)
        new_centers = update_centers_full(x, labels, centers, k)
        shift = (new_centers - centers).norm(dim=1).mean().item()
        centers = new_centers
        print(f"iter {it + 1:03d}/{iterations}: mean center shift={shift:.6f}")
        if shift < 1e-6:
            break

    labels, scores = assign_full(x, centers)
    inertia = compute_inertia_from_scores(scores)
    return centers.cpu(), labels.cpu(), inertia


def cluster_chunked(
    x_cpu: torch.Tensor,
    k: int,
    iterations: int,
    seed: int,
    chunk_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    centers = init_centers(x_cpu, k, seed, device)

    for it in range(iterations):
        print(f"iter {it + 1:03d}/{iterations}")
        labels, scores = assign_chunked(x_cpu, centers, chunk_size, device)
        new_centers = update_centers_chunked(x_cpu, labels, centers, k, chunk_size, device)
        shift = (new_centers - centers).norm(dim=1).mean().item()
        centers = new_centers
        print(f"  mean center shift={shift:.6f}")
        if shift < 1e-6:
            break

    labels, scores = assign_chunked(x_cpu, centers, chunk_size, device)
    inertia = compute_inertia_from_scores(scores)
    return centers.cpu(), labels.cpu(), inertia


def compute_inertia_from_scores(scores: torch.Tensor) -> float:
    return float((1.0 - scores).sum().item())

#in my case, CelebHQ-A and FFHQ fit in memory, and Imagenet, I had to use chunked mode
def choose_mode(args: argparse.Namespace, x: torch.Tensor, k: int, device: torch.device) -> str:
    if args.mode != "auto":
        return args.mode

    n, d = x.shape
    estimated_gb = (n * d + n * k + k * d) * 4 / (1024 ** 3)
    print(f"estimated full-mode tensor memory: {estimated_gb:.2f} GB")
    return "full" if estimated_gb <= args.max_full_gb else "chunked"


def output_path_for_k(out: str, k: int, multi_k: bool) -> str:
    p = Path(out)
    if multi_k or p.suffix == "" or p.is_dir():
        p.mkdir(parents=True, exist_ok=True)
        return str(p / f"cluster_{k}.pth")
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def save_cluster(
    out_path: str,
    x_cpu: torch.Tensor,
    centers: torch.Tensor,
    labels: torch.Tensor,
    inertia: float,
    k: int,
    source_files: list[str],
    save_x_org: bool,
) -> None:
    result = {
        "labels": labels.long().unsqueeze(0),
        "centers": centers.float().unsqueeze(0),
        "inertia": inertia,
        "k": k,
        "source_files": source_files,
    }
    if save_x_org:
        result["x_org"] = x_cpu.float()

    print(f"saving {out_path}")
    torch.save(result, out_path)

    for i in range(k):
        print(f"cluster {i:04d}: {(labels == i).sum().item()} samples")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", nargs="+", required=True, help="Feature .pt files or globs.")
    parser.add_argument("--out", required=True, help="Output .pth file or output directory.")
    parser.add_argument("--k", nargs="+", type=int, default=[24])
    parser.add_argument("--mode", choices=["auto", "full", "chunked"], default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--normalize", type=int, default=1)
    parser.add_argument("--save-x-org", type=int, default=1)
    parser.add_argument("--max-full-gb", type=float, default=96.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_paths = expand_feature_paths(args.features)
    x = load_features(feature_paths)

    x = normalize_features(x)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"device={device}")

    multi_k = len(args.k) > 1
    for k in args.k:
        print(f"\n=== clustering k={k} ===")
        mode = choose_mode(args, x, k, device)
        print(f"mode={mode}")

        if mode == "full":
            centers, labels, inertia = cluster_full(
                x, k=k, iterations=args.iterations, seed=args.seed, device=device
            )
        else:
            centers, labels, inertia = cluster_chunked(
                x,
                k=k,
                iterations=args.iterations,
                seed=args.seed,
                chunk_size=args.chunk_size,
                device=device,
            )

        out_path = output_path_for_k(args.out, k, multi_k)
        save_cluster(
            out_path,
            x_cpu=x,
            centers=centers,
            labels=labels,
            inertia=inertia,
            k=k,
            source_files=feature_paths,
            save_x_org=bool(args.save_x_org),
        )


if __name__ == "__main__":
    main()
