from datasets import load_dataset
import torch
import numpy as np
import torch
from .transforms import build_transforms
import datasets
import sys
from functools import partial
from PIL import Image
from torchvision.transforms import functional as F

def build_dataset(dataset_config, transforms=None):

    dataset = load_dataset(**dataset_config)
    id_col  = np.arange(len(dataset))

    dset_id = datasets.Dataset.from_dict({"id": id_col})
    dataset = datasets.concatenate_datasets([dataset, dset_id], axis=1)
    
    if transforms is not None:
        dataset.set_transform(transforms)
    return dataset


def build_distributed_dataloader(dataset, collate_fn=None, dataloader_config=None):
    dataloader_config.pop("shuffle", None)
    return torch.utils.data.DataLoader(
        dataset,
        sampler=torch.utils.data.distributed.DistributedSampler(dataset),
        collate_fn=collate_fn,
        pin_memory=True,
        **dataloader_config,
    )


def build_dataloader(
    dataset, collate_fn=None, dataloader_config=None, use_distributed_sampler=False
):
    if dataloader_config is None:
        dataloader_config = {}

    if use_distributed_sampler:
        return partial(
            build_distributed_dataloader,
            dataset=dataset,
            collate_fn=collate_fn,
            dataloader_config=dataloader_config,
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collate_fn,
        pin_memory=True,
        **dataloader_config,
    )
    return dataloader


def build_collate_fn(collate_fn_config=None):
    if collate_fn_config is None:
        return None
    from .collate_func import video_data_collate_fn

    return partial(video_data_collate_fn, **collate_fn_config)




def force_flip_then(base_transform, hflip=False, vflip=False, image_key="image"):
    """
    Wrap a dict-based HF transform: flip images first, then call base_transform(examples).
    Works for batched dicts (values are lists) and single examples.
    """
    def _flip_one(im):
        if isinstance(im, np.ndarray):
            im = Image.fromarray(im)
        if hflip:
            im = F.hflip(im)
        if vflip:
            im = F.vflip(im)
        return im

    def _transform(examples):
        imgs = examples[image_key]
        if isinstance(imgs, list):                  # batched dict
            examples[image_key] = [_flip_one(im) for im in imgs]
        else:                                       # single example
            examples[image_key] = _flip_one(imgs)
        return base_transform(examples)             # <- pass dict, not Image

    return _transform