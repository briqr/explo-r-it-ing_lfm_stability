from __future__ import annotations

import os
from dataclasses import dataclass

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from train_consts import * 
from fm_training.distributed import DistEnv
from dataset.transforms import build_transforms
from dataset import build_dataset, force_flip_then
from dataset.precomputed_latent_small import PreencodedLatentsDataset
from dataset.precomputed_latent_big import PreencodedLatentsShards
from dataset.ffhq_dataset import FFHQDataset
@dataclass
class DatasetInfo:
    in_channels: int
    scale_factor: float
    num_classes: int = 1
    class_dropout_prob: float = 0.0
    is_conditional: bool = False
    cluster_path: str | None = None


def _image_transform(im_size: int):
    return transforms.Compose(
        [
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def get_dataset_info(args, method: str) -> DatasetInfo:
    """Dataset metadata used by the model and pruning code."""
    name = args.dataset_name

    if "imagenet" in name:
        base = "imagenet"
        num_classes = 1000
        class_dropout_prob = 0.1
        is_conditional = True
        cluster_path = imagenet_clusters_path
    elif "ffhq" in name:
        base = "ffhq"
        num_classes = args.num_classes
        class_dropout_prob = 0.0
        is_conditional = False
        cluster_path = ffhq_cluster_path
    elif "celebhq" in name:
        base = "celebhq_512" if "512" in name else "celebhq"
        num_classes = args.num_classes
        class_dropout_prob = 0.0
        is_conditional = False
        cluster_path = celebhq_cluster_path
        if args.num_clusters != 24:
            cluster_path = celebhq_cluster_path.replace("cluster_clip_24.pth", f"cluster_clip_{args.num_clusters}.pth")
    else:
        raise ValueError(f"Unknown dataset_name={name}")

    if args.vae_pretrained:
        in_channels = in_channels_map['vae_pretrained']
        scale_factor = scale_factor_map['vae_pretrained']
    else:
        in_channels = in_channels_map[base.replace("_512", "")]
        scale_factor = scale_factor_map[base.replace("_512", "")]

    return DatasetInfo(
        in_channels=in_channels,
        scale_factor=scale_factor,
        num_classes=num_classes,
        class_dropout_prob=class_dropout_prob,
        is_conditional=is_conditional,
        cluster_path=cluster_path,
    )


def get_dataset(name: str, split: str = "train", is_training: bool = False, im_size: int = 256):

    if name == "ffhq":
        if "val" in split:
            split = "val"
        return FFHQDataset(data_dir=FFHQ_DATAROOT, split=split, transform=_image_transform(im_size))

    if name == "ffhq_precomputed":
        return PreencodedLatentsDataset(dataset_path=PRECOMPUTED_LATENTS_PATH_FFHQ, split=split)

    if name == "celebhq_precomputed":
        if "train" not in split:
            return PreencodedLatentsDataset(PRECOMPUTED_LATENTS_PATH_CELEBHQ.replace("train.pt", "valid.pt"))
        return PreencodedLatentsDataset(dataset_path=PRECOMPUTED_LATENTS_PATH_CELEBHQ)

    if name == "celebhq_precomputed_512":
        if "train" not in split:
            return PreencodedLatentsDataset(PRECOMPUTED_LATENTS_PATH_CELEBHQ_512.replace("train.pt", "valid.pt"))
        return PreencodedLatentsDataset(PRECOMPUTED_LATENTS_PATH_CELEBHQ_512)

    if name == "celebhq_precomputed_male":
        return PreencodedLatentsDataset(dataset_path=PRECOMPUTED_LATENTS_PATH_MALE)

    if name == "celebhq_precomputed_female":
        return PreencodedLatentsDataset(dataset_path=PRECOMPUTED_LATENTS_PATH_FEMALE)

    if name == "imagenet_precomputed":
        return PreencodedLatentsShards(split=split)

    if name in {"celebhq", "imagenet"}:
        if name == "celebhq":
            dataset_name = "jxie/celeba-hq"
            cache_dir = CACHE_DIR
        else:
            dataset_name = "imagenet-1k"
            cache_dir = CACHE_DIR

        data_cfg = {
            "path": dataset_name,
            "name": "default",
            "cache_dir": cache_dir,
            "split": split,
            "trust_remote_code": True,
            "token": os.environ.get("HF_TOKEN"),
        }
        transform_cfg = {
            "no_aug": True,
            "is_training": is_training,
            "input_size": im_size,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        }
        data_transform = build_transforms(transform_cfg, dataset_name=dataset_name)
        return build_dataset(
            data_cfg,
            transforms=force_flip_then(data_transform, hflip=False, vflip=False, image_key="image"),
        )

    raise ValueError(f"Unknown dataset: {name}")


def build_loader(args, dataset, env: DistEnv):
    sampler = None
    if env.ddp:
        sampler = DistributedSampler(
            dataset,
            num_replicas=env.world_size,
            rank=env.rank,
            shuffle=True,
            seed=args.global_seed,
        )

    loader = DataLoader(
        dataset,
        batch_size=args.global_batch_size // env.world_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader, sampler




from dataclasses import dataclass
from typing import Optional
from train_consts import (
    scale_factor_map,
    in_channels_map,
    celebhq_cluster_path,
    imagenet_clusters_path,
    ffhq_cluster_path,
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    base_name: str
    in_channels: int
    scale_factor: float
    num_classes: int = 1
    class_dropout_prob: float = 0.0
    is_conditional: bool = False
    cluster_path: Optional[str] = None


@dataclass(frozen=True)
class ModelDataConfig:
    in_channels: int
    scale_factor: float
    num_classes: int
    class_dropout_prob: float
    is_conditional: bool
    cluster_path: Optional[str]


def get_dataset_spec(dataset_name: str) -> DatasetSpec:
    print('*****dataset_name', dataset_name)
    if "imagenet" in dataset_name:
        return DatasetSpec(
            name=dataset_name,
            base_name="imagenet",
            in_channels=in_channels_map["imagenet"],
            scale_factor=scale_factor_map["imagenet"],
            num_classes=1000,
            class_dropout_prob=0.1,
            is_conditional=True,
            cluster_path=imagenet_clusters_path,
        )

    if "celebhq" in dataset_name:
        base_name = "celebhq_512" if "512" in dataset_name else "celebhq"
        return DatasetSpec(
            name=dataset_name,
            base_name=base_name,
            in_channels=in_channels_map["celebhq"],
            scale_factor=scale_factor_map[base_name],
            cluster_path=celebhq_cluster_path,
        )

    if "ffhq" in dataset_name:
        return DatasetSpec(
            name=dataset_name,
            base_name="ffhq",
            in_channels=in_channels_map["ffhq"],
            scale_factor=scale_factor_map["ffhq"],
            cluster_path=ffhq_cluster_path,
        )


    raise ValueError(f"Unknown dataset_name: {dataset_name}")


def model_data_config_from_spec(spec: DatasetSpec, vae_pretrained: bool = False) -> ModelDataConfig:
    if vae_pretrained:
        return ModelDataConfig(
            in_channels=4,
            scale_factor=0.18215,
            num_classes=spec.num_classes,
            class_dropout_prob=spec.class_dropout_prob,
            is_conditional=spec.is_conditional,
            cluster_path=spec.cluster_path,
        )

    return ModelDataConfig(
        in_channels=spec.in_channels,
        scale_factor=spec.scale_factor,
        num_classes=spec.num_classes,
        class_dropout_prob=spec.class_dropout_prob,
        is_conditional=spec.is_conditional,
        cluster_path=spec.cluster_path,
    )