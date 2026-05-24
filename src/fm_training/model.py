from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

# Same behavior as the original DiT script.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from transport import create_transport
from diffusion import create_diffusion
from models import DiT_models
from unet_fm import UNet_models

from timm import create_model
import autoencoders
from diffusers.models import AutoencoderKL
DiT_models.update(UNet_models)
from train_consts import *

def model_choices():
    return list(DiT_models.keys())


def build_transport(args):
    if args.transport == "diffusion":
        return create_diffusion(timestep_respacing=""), False, True
    return create_transport(), True, False


def build_model(args, info, learn_sigma: bool, device: torch.device):
    assert args.image_size % 8 == 0, "Image size must be divisible by 8."
    latent_size = args.image_size // 8

    model = DiT_models[args.model](
        input_size=latent_size,
        in_channels=info.in_channels,
        num_classes=info.num_classes,
        class_dropout_prob=info.class_dropout_prob,
        learn_sigma=learn_sigma,
    )

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    return model, ema


def load_checkpoint_if_needed(args, model, ema, logger):
    if not args.resume:
        return None

    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    ema.load_state_dict(checkpoint["ema"], strict=True)
    logger.info(f"Using checkpoint: {args.resume}")
    print(f"Using checkpoint: {args.resume}")
    return checkpoint


@torch.no_grad()
def update_ema(ema_model, model, decay: float = 0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag: bool = True):
    for param in model.parameters():
        param.requires_grad = flag


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def maybe_load_checkpoint(
    args: argparse.Namespace,
    model: torch.nn.Module,
    ema: torch.nn.Module,
    logger,
) -> Optional[dict[str, Any]]:
    if not args.resume:
        return None

    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    ema.load_state_dict(checkpoint["ema"], strict=True)
    logger.info(f"Using checkpoint: {args.resume}")
    print(f"Using checkpoint: {args.resume}")
    return checkpoint

def resolve_vae_path(dataset_name: str, vae_path: str) -> str:
    if vae_path:
        return vae_path

    if "celebhq" in dataset_name:
        return celebhq_vq_vae_path
    if "ffhq" in dataset_name:
        return ffhq_vq_vae_path
    if "imagenet" in dataset_name:
        return imagenet_vq_vae_path

    raise ValueError(f"Please pass --vae-path for dataset_name={dataset_name!r}")

#if vae_path explicitly passed, use it, otherwise, determine based on dataset name
#is_pretrained refers to stability ai pretrained autoencoder, which we did not use for our final experiments

def get_autoencoder(dataset_name, vae_path=None, encoder_type="vq_gan_taming", is_pretrained=False, device=None):
    if is_pretrained:
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_str}").to(device)
    else:
        vae_path = resolve_vae_path(dataset_name, vae_path)
        print("Loading VAE from:", vae_path)
        vae = create_model(model_name=encoder_type,  path=vae_path, pretrained=True)
    
    if device is not None:
        vae = vae.to(device)
    vae.eval()
    return vae
