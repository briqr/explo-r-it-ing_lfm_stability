"""
Extract CLIP image features for real datasets or generated sample tensors.

For small datasets, save one tensor:
python extract_features_clip.py \
  --input dataset \
  --dataset-name celebhq \
  --split train \
  --out-dir /p/scratch/multiscale-wm/briq/explfm/data/celebhq/clip_features/train \
  --save-mode single

For ImageNet-scale datasets, save feature shards:
python extract_features_clip.py \
  --input dataset \
  --dataset-name imagenet \
  --split train \
  --out-dir /p/scratch/multiscale-wm/briq/explfm/data/imagenet/clip_features/train \
  --save-mode sharded \
  --shard-size 100000 \
  --feature-prefix imagenet_clip_features
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel

from dataset.data import get_dataset


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class TensorImageDataset(Dataset):
    def __init__(self, data: Any):
        if isinstance(data, torch.Tensor):
            self.data = data.cpu()
        elif isinstance(data, list):
            self.data = [x.cpu() if isinstance(x, torch.Tensor) else x for x in data]
        else:
            raise TypeError(f"Expected Tensor or list, got {type(data)}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {"image": self.data[idx], "id": idx}


def image_tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().cpu()
    if x.ndim == 4:
        x = x[0]
    
    x = x * 0.5 + 0.5
    x = x.float().clamp(0, 1)
    return to_pil_image(x).convert("RGB")


def item_to_pil(item: Any) -> Image.Image:
    if isinstance(item, Image.Image):
        return item.convert("RGB")
    if isinstance(item, torch.Tensor):
        return image_tensor_to_pil(item)
    raise TypeError(f"Unsupported image type: {type(item)}")


def _to_int(value: Any, default: int = -1) -> int:
    if value is None:
        return default
    return int(value)


def make_collate_fn():
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = [item_to_pil(x["image"]) for x in batch]
        ids = [_to_int(x.get("id", i), i) for i, x in enumerate(batch)]
        labels = [_to_int(x.get("label", -1), -1) for x in batch]
        paths = [str(x.get("path", "")) for x in batch]
        return {"images": images, "ids": ids, "labels": labels, "paths": paths}

    return collate


def build_dataset_from_args(args: argparse.Namespace) -> Dataset:
    if args.input == "dataset":
        return get_dataset(
            name=args.dataset_name,
            split=args.split,
            is_training=False,
            im_size=args.image_size,
        )

    data = torch.load(args.data_path, map_location="cpu", weights_only=False)
    return TensorImageDataset(data)


def load_clip(model_name: str, device: torch.device):
    processor = AutoProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model


@torch.no_grad()
def compute_batch_features(batch, processor, model, device, normalize=True):
    inputs = processor(images=batch["images"], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out = model.get_image_features(**inputs)

    # CLIPModel usually returns a tensor.
    # Some model/classes return BaseModelOutputWithPooling.
    if isinstance(out, torch.Tensor):
        features = out
    elif hasattr(out, "pooler_output") and out.pooler_output is not None:
        features = out.pooler_output
    elif hasattr(out, "image_embeds") and out.image_embeds is not None:
        features = out.image_embeds
    else:
        raise TypeError(f"Unexpected image feature output type: {type(out)}")

    features = features.float()

    features = F.normalize(features, p=2, dim=-1)

    return features

def save_single(
    out_dir: str,
    features: torch.Tensor,
    ids: list[int],
    labels: list[int],
    paths: list[str],
    args: argparse.Namespace,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    feature_path = os.path.join(out_dir, args.feature_filename)
    torch.save(features, feature_path)

    torch.save(
        {
            "features": features,
            "ids": ids,
            "labels": labels,
            "paths": paths,
            "model_name": args.model_name,
            "dataset_name": args.dataset_name,
            "split": args.split,
            "input": args.input,
            "data_path": args.data_path,
        },
        os.path.join(out_dir, "clip_features_with_metadata.pth"),
    )

    write_metadata(
        out_dir=out_dir,
        args=args,
        num_samples=int(features.shape[0]),
        feature_dim=int(features.shape[1]),
        feature_files=[os.path.basename(feature_path)],
    )
    print(f"Saved features: {feature_path}")
    print(f"Shape: {tuple(features.shape)}")


def write_metadata(
    out_dir: str,
    args: argparse.Namespace,
    num_samples: int,
    feature_dim: int,
    feature_files: list[str],
) -> None:
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_samples": num_samples,
                "feature_dim": feature_dim,
                "model_name": args.model_name,
                "save_mode": args.save_mode,
                "feature_filename": args.feature_filename,
                "feature_prefix": args.feature_prefix,
                "feature_files": feature_files,
                "dataset_name": args.dataset_name,
                "split": args.split,
                "input": args.input,
                "data_path": args.data_path,
            },
            f,
            indent=2,
        )


def flush_shard(
    out_dir: str,
    shard_idx: int,
    features: list[torch.Tensor],
    ids: list[int],
    labels: list[int],
    paths: list[str],
    args: argparse.Namespace,
) -> tuple[str, int, int]:
    x = torch.cat(features, dim=0)
    feature_name = f"{args.feature_prefix}_{shard_idx:05d}.pth"
    feature_path = os.path.join(out_dir, feature_name)

    # Save the tensor itself because clustering scripts can load tensor shards by glob.
    torch.save(x, feature_path)

    # Save sidecar metadata separately so the feature file stays simple.
    meta_name = f"{args.feature_prefix}_{shard_idx:05d}.meta.json"
    with open(os.path.join(out_dir, meta_name), "w", encoding="utf-8") as f:
        json.dump({"ids": ids, "labels": labels, "paths": paths}, f)

    print(f"Saved shard {shard_idx}: {feature_path} shape={tuple(x.shape)}")
    return feature_name, int(x.shape[0]), int(x.shape[1])


@torch.no_grad()
def extract_to_single_file(
    loader: DataLoader,
    processor: AutoProcessor,
    model: CLIPModel,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    all_features: list[torch.Tensor] = []
    all_ids: list[int] = []
    all_labels: list[int] = []
    all_paths: list[str] = []

    for batch in tqdm(loader, desc="Extracting CLIP features"):
        features = compute_batch_features(batch, processor, model, device)
        all_features.append(features)
        all_ids.extend(batch["ids"])
        all_labels.extend(batch["labels"])
        all_paths.extend(batch["paths"])

    features = torch.cat(all_features, dim=0)
    save_single(args.out_dir, features, all_ids, all_labels, all_paths, args)


@torch.no_grad()
def extract_to_shards(
    loader: DataLoader,
    processor: AutoProcessor,
    model: CLIPModel,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    os.makedirs(args.out_dir, exist_ok=True)

    shard_features: list[torch.Tensor] = []
    shard_ids: list[int] = []
    shard_labels: list[int] = []
    shard_paths: list[str] = []
    feature_files: list[str] = []
    shard_idx = 0
    total = 0
    feature_dim = -1

    for batch in tqdm(loader, desc="Extracting CLIP feature shards"):
        features = compute_batch_features(batch, processor, model, device)
        shard_features.append(features)
        shard_ids.extend(batch["ids"])
        shard_labels.extend(batch["labels"])
        shard_paths.extend(batch["paths"])

        if feature_dim < 0:
            feature_dim = int(features.shape[1])

        current_count = sum(x.shape[0] for x in shard_features)
        if current_count >= args.shard_size:
            name, n, d = flush_shard(
                args.out_dir,
                shard_idx,
                shard_features,
                shard_ids,
                shard_labels,
                shard_paths,
                args,
            )
            feature_files.append(name)
            total += n
            feature_dim = d
            shard_idx += 1
            shard_features = []
            shard_ids = []
            shard_labels = []
            shard_paths = []

    if shard_features:
        name, n, d = flush_shard(
            args.out_dir,
            shard_idx,
            shard_features,
            shard_ids,
            shard_labels,
            shard_paths,
            args,
        )
        feature_files.append(name)
        total += n
        feature_dim = d

    write_metadata(args.out_dir, args, total, feature_dim, feature_files)
    print(f"Saved {len(feature_files)} shards to {args.out_dir}")
    print(f"Use clustering glob: {os.path.join(args.out_dir, args.feature_prefix + '_*.pth')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", choices=["dataset", "pth"], default="dataset")
    parser.add_argument("--dataset-name", type=str, default="celebhq")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--data-path", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="data/celebhq/clip_features/train")
    parser.add_argument("--model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--feature-filename", type=str, default="clip_features.pt")
    parser.add_argument("--feature-prefix", type=str, default="clip_features")
    parser.add_argument("--save-mode", choices=["single", "sharded"], default="single")
    parser.add_argument("--shard-size", type=int, default=100000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = build_dataset_from_args(args)
    print(f"Dataset size: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=make_collate_fn(),
    )

    processor, model = load_clip(args.model_name, device)

    if args.save_mode == "single":
        extract_to_single_file(loader, processor, model, device, args)
    else:
        extract_to_shards(loader, processor, model, device, args)


if __name__ == "__main__":
    main()
