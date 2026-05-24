#!/usr/bin/env python3
"""
classify_face_attributes.py

 script for classifying perceived face attributes on either:
  1) a training dataset, or
  2) a folder / filelist of generated images.

It writes:
  - predictions.jsonl
  - predictions.csv
  - distributions.json
  - kl_discrepancy.json
  - gender_indices/female_indices.json and male_indices.json for datasets
  - gender_indices/female_paths.txt and male_paths.txt for image folders/filelists

Example, training data:
  python classify_face_attributes.py \
    --input dataset \
    --dataset-name celebhq \
    --split train \
    --out-dir attr_train_celebhq \
    --save-gender-splits

Example, generated images:
  python classify_face_attributes.py \
    --input image_dir \
    --image-dir samples/celebhq_pruned05 \
    --out-dir attr_generated_pruned05

Example, compare generated distribution to training distribution:
  python classify_face_attributes.py \
    --input image_dir \
    --image-dir samples/celebhq_pruned05 \
    --out-dir attr_generated_pruned05 \
    --reference-distributions attr_train_celebhq/distributions.json
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from dataset.data import get_dataset
import numpy as np
import torch
from PIL import Image

hf_token = os.environ.get("HF_TOKEN")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

ATTRIBUTE_LABELS = {
    "gender": ["female", "male"],
    "age": ["child", "young", "middle-aged", "elderly"],
    "skintone": ["light", "medium", "dark"],
    "hair_color": ["black", "brown", "blonde", "red", "gray", "white", "bald", "other"],
}

ATTRIBUTE_PROMPTS = {
    "gender": "What is the perceived gender of the person? Answer only one label from: female, male.",
    "age": "What is the perceived age group of the person? Answer only one label from: child, young, middle-aged, elderly.",
    "skintone": "What is the perceived skin tone of the person? Answer only one label from: light, medium, dark.",
    "hair_color": "What is the hair color of the person? Answer only one label from: black, brown, blonde, red, gray, white, bald, other.",
}

LABEL_ALIASES = {
    "gender": {
        "woman": "female",
        "girl": "female",
        "female person": "female",
        "man": "male",
        "boy": "male",
        "male person": "male",
    },
    "age": {
        "kid": "child",
        "baby": "child",
        "teen": "young",
        "teenager": "young",
        "young adult": "young",
        "adult": "middle-aged",
        "middle aged": "middle-aged",
        "old": "elderly",
        "senior": "elderly",
    },
    "skintone": {
        "fair": "light",
        "pale": "light",
        "white": "light",
        "tan": "medium",
        "brown": "medium",
        "olive": "medium",
        "black": "dark",
        "deep": "dark",
    },
    "hair_color": {
        "grey": "gray",
        "gray hair": "gray",
        "grey hair": "gray",
        "black hair": "black",
        "brown hair": "brown",
        "blond": "blonde",
        "blonde hair": "blonde",
        "red hair": "red",
        "ginger": "red",
        "white hair": "white",
        "no hair": "bald",
    },
}


@dataclass
class Item:
    idx: int
    path: str | None
    image: Image.Image


class ImagePathDataset:
    def __init__(self, paths: list[str]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        return {"image": image, "id": idx, "path": path}


def list_images(image_dir: str, recursive: bool = True) -> list[str]:
    root = Path(image_dir)
    pattern = "**/*" if recursive else "*"
    paths = [str(p) for p in root.glob(pattern) if p.suffix.lower() in IMAGE_EXTS]
    return sorted(paths)


def read_filelist(filelist: str) -> list[str]:
    with open(filelist, "r", encoding="utf-8") as f:
        paths = [line.strip() for line in f if line.strip()]
    return paths


def import_function(spec: str):
    module_name, fn_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, fn_name)


def build_input_dataset(args):
    if args.input == "image_dir":
        return ImagePathDataset(list_images(args.image_dir, recursive=not args.no_recursive))

    if args.input == "filelist":
        return ImagePathDataset(read_filelist(args.filelist))

    if args.input == "dataset":
        return get_dataset(
            name=args.dataset_name,
            split=args.split,
            is_training=False,
            is_booster=args.is_booster,
            im_size=args.im_size,
        )

    raise ValueError(f"Unknown input kind: {args.input}")


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().cpu()
    if x.ndim == 4:
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"Expected image tensor with shape CxHxW, got {tuple(x.shape)}")

    # Most project image transforms normalize to [-1, 1]. Undo that when needed.
    if float(x.min()) < -0.05:
        x = (x + 1.0) / 2.0
    x = x.clamp(0, 1)
    x = (x * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(x).convert("RGB")


def sample_to_item(sample: Any, idx: int) -> Item:
    path = None

    if isinstance(sample, dict):
        image = sample.get("image")
        sample_id = sample.get("id", idx)
        path = sample.get("path") or sample.get("file") or sample.get("filename")
        try:
            idx = int(sample_id)
        except Exception:
            idx = int(idx)
    elif isinstance(sample, (tuple, list)):
        image = sample[0]
    else:
        image = sample

    if isinstance(image, Image.Image):
        pil = image.convert("RGB")
    elif isinstance(image, torch.Tensor):
        pil = tensor_to_pil(image)
    elif isinstance(image, np.ndarray):
        pil = Image.fromarray(image).convert("RGB")

    return Item(idx=idx, path=str(path) if path is not None else None, image=pil)


def load_classifier(model_id: str, device: str):
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
    dtype = torch.bfloat16 if torch.cuda.is_available() and device.startswith("cuda") else torch.float32
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype, attn_implementation="eager", token=hf_token)
    model = model.to(device)
    model.eval()
    return processor, model


def normalize_answer(text: str, attr: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z\- ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    aliases = LABEL_ALIASES.get(attr, {})
    for key, value in aliases.items():
        if re.search(rf"\b{re.escape(key)}\b", text):
            return value

    labels = ATTRIBUTE_LABELS[attr]
    for label in labels:
        if re.search(rf"\b{re.escape(label)}\b", text):
            return label

    return "unknown"


@torch.no_grad()
def classify_batch(processor, model, images: list[Image.Image], attr: str, device: str, max_new_tokens: int) -> list[str]:
    prompt = "<image> answer in English: " + ATTRIBUTE_PROMPTS[attr]
    prompts = [prompt] * len(images)

    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    outputs = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    new_tokens = outputs[:, input_len:]
    raw_texts = processor.batch_decode(new_tokens, skip_special_tokens=True)
    return [normalize_answer(t, attr) for t in raw_texts]


def load_done_predictions(path: Path) -> dict[int, dict[str, Any]]:
    done = {}
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            done[int(row["id"])] = row
    return done


def write_json(path: Path, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def write_predictions_csv(path: Path, rows: list[dict[str, Any]], attrs: list[str]):
    fields = ["id", "path"] + attrs
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def distributions_from_rows(rows: list[dict[str, Any]], attrs: list[str]) -> dict[str, Any]:
    out = {}
    for attr in attrs:
        labels = ATTRIBUTE_LABELS[attr]
        counts = {label: 0 for label in labels}
        counts["unknown"] = 0

        for row in rows:
            label = row.get(attr, "unknown") #there are very few unkowns, but including them sways kl by a large margin, so we skip them
            if label not in counts:
                label = "unknown"
            counts[label] += 1

        total_known = sum(counts[label] for label in labels)
        probs = {
            label: (counts[label] / total_known if total_known > 0 else 0.0)
            for label in labels
        }

        out[attr] = {
            "labels": labels,
            "counts": counts,
            "num_known": total_known,
            "num_unknown": counts["unknown"],
            "probs": probs,
        }
    return out


def kl(p: dict[str, float], q: dict[str, float], labels: list[str], eps: float = 1e-12) -> float:
    value = 0.0
    for label in labels:
        pi = float(p.get(label, 0.0))
        qi = float(q.get(label, 0.0))
        if pi > 0:
            value += pi * math.log(pi / max(qi, eps))
    return value


def compute_kl_metrics(distributions: dict[str, Any], reference_path: str | None = None) -> dict[str, Any]:
    metrics = {}
    reference = None
    if reference_path:
        with open(reference_path, "r", encoding="utf-8") as f:
            reference = json.load(f)

    for attr, dist in distributions.items():
        labels = dist["labels"]
        p = dist["probs"]
        uniform = {label: 1.0 / len(labels) for label in labels}

        metrics[attr] = {
            "kl_to_uniform": kl(p, uniform, labels),
        }

        if reference is not None and attr in reference:
            q = reference[attr]["probs"]
            metrics[attr]["kl_to_reference"] = kl(p, q, labels)
            metrics[attr]["kl_reference_to_this"] = kl(q, p, labels)

    return metrics


def save_gender_splits(rows: list[dict[str, Any]], out_dir: Path):
    split_dir = out_dir / "gender_indices"
    split_dir.mkdir(parents=True, exist_ok=True)

    female_indices = [int(row["id"]) for row in rows if row.get("gender") == "female"]
    male_indices = [int(row["id"]) for row in rows if row.get("gender") == "male"]
    female_paths = [row["path"] for row in rows if row.get("gender") == "female" and row.get("path")]
    male_paths = [row["path"] for row in rows if row.get("gender") == "male" and row.get("path")]

    write_json(split_dir / "female_indices.json", female_indices)
    write_json(split_dir / "male_indices.json", male_indices)

    with open(split_dir / "female_paths.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(female_paths))
        if female_paths:
            f.write("\n")

    with open(split_dir / "male_paths.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(male_paths))
        if male_paths:
            f.write("\n")

    print(f"Saved {len(female_indices)} female and {len(male_indices)} male indices to {split_dir}")


def classify_dataset(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    attrs = args.attributes
    for attr in attrs:
        if attr not in ATTRIBUTE_LABELS:
            raise ValueError(f"Unknown attribute {attr}. Available: {sorted(ATTRIBUTE_LABELS)}")

    predictions_path = out_dir / "predictions.jsonl"
    done = load_done_predictions(predictions_path) if args.resume else {}

    dataset = build_input_dataset(args)
    if args.max_samples is not None:
        n_items = min(len(dataset), args.max_samples)
    else:
        n_items = len(dataset)

    print(f"Classifying {n_items} samples from input={args.input}")
    processor, model = load_classifier(args.model_id, args.device)

    rows = [done[k] for k in sorted(done) if k < n_items]
    pending_items: list[Item] = []

    with open(predictions_path, "a", encoding="utf-8") as fout:
        for i in range(n_items):
            if i in done:
                continue

            item = sample_to_item(dataset[i], i)
            pending_items.append(item)

            if len(pending_items) == args.batch_size:
                rows.extend(classify_and_write_batch(pending_items, attrs, processor, model, args, fout))
                pending_items = []

        if pending_items:
            rows.extend(classify_and_write_batch(pending_items, attrs, processor, model, args, fout))

    rows = load_all_rows(predictions_path)
    rows = [row for row in rows if int(row["id"]) < n_items]
    rows = sorted(rows, key=lambda x: int(x["id"]))

    write_predictions_csv(out_dir / "predictions.csv", rows, attrs)

    distributions = distributions_from_rows(rows, attrs)
    write_json(out_dir / "distributions.json", distributions)

    kl_metrics = compute_kl_metrics(distributions, args.reference_distributions)
    write_json(out_dir / "kl_discrepancy.json", kl_metrics)

    if args.save_gender_splits or "gender" in attrs:
        save_gender_splits(rows, out_dir)

    print(f"Wrote predictions to {predictions_path}")
    print(f"Wrote distributions to {out_dir / 'distributions.json'}")
    print(f"Wrote KL metrics to {out_dir / 'kl_discrepancy.json'}")


def classify_and_write_batch(items: list[Item], attrs: list[str], processor, model, args, fout) -> list[dict[str, Any]]:
    images = [item.image for item in items]
    rows = [{"id": item.idx, "path": item.path} for item in items]

    for attr in attrs:
        labels = classify_batch(
            processor,
            model,
            images,
            attr,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
        for row, label in zip(rows, labels):
            row[attr] = label

    for row in rows:
        fout.write(json.dumps(row) + "\n")
    fout.flush()

    if rows:
        print(f"classified id={rows[-1]['id']}")
    return rows


def load_all_rows(predictions_path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", choices=["dataset", "image_dir", "filelist"], required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--filelist", type=str, default=None)
    parser.add_argument("--no-recursive", action="store_true")

    parser.add_argument("--dataset-name", type=str, default="celebhq")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dataset-factory", type=str, default="lfm_training.data:get_dataset")
    parser.add_argument("--is-booster", type=int, default=1)
    parser.add_argument("--im-size", type=int, default=256)

    parser.add_argument("--model-id", type=str, default="google/paligemma-3b-mix-224")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")

    parser.add_argument(
        "--attributes",
        nargs="+",
        default=["gender", "age", "skintone", "hair_color"],
        choices=sorted(ATTRIBUTE_LABELS),
    )
    parser.add_argument("--reference-distributions", type=str, default=None)
    parser.add_argument("--save-gender-splits", action="store_true")

    return parser


def main():
    args = build_parser().parse_args()

    if args.input == "image_dir" and not args.image_dir:
        raise ValueError("--image-dir is required when --input image_dir")
    if args.input == "filelist" and not args.filelist:
        raise ValueError("--filelist is required when --input filelist")

    classify_dataset(args)


if __name__ == "__main__":
    main()
