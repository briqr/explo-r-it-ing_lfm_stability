"""
Shared sampling and evaluation helpers for DiT/LFM experiments.

This file is intentionally simple:
  1) resolve checkpoint path
  2) read config from checkpoint["config"] or old checkpoint["args"]
  3) build model + sampler + VAE
  4) generate samples / load real images / compute FID metrics
"""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Callable

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from diffusion import create_diffusion
from models import DiT_models
from train_consts import in_channels_map, scale_factor_map
from unet_fm import UNet_models


from transport import Sampler as TransportSampler
from transport import create_transport

from dataset.data import get_dataset
from fm_training.model import get_autoencoder


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclass
class RunConfig:
    dataset_name: str
    model_name: str
    transport: str
    vae_pretrained: bool

    image_size: int
    latent_size: int
    in_channels: int

    scale_factor: float
    num_classes: int
    class_dropout_prob: float
    is_conditional: bool

    vae_path: str 
    encoder_type: str


@dataclass
class CheckpointInfo:
    ckpt_path: str
    exp_dir: str
    epoch_stem: str
    vis_dir: str


@dataclass
class SamplerBundle:
    is_fm: bool
    sample_fn: Callable
    num_steps: int


@dataclass
class SamplingProfile:
    denoise_seconds: float = 0.0
    decode_seconds: float = 0.0
    total_images: int = 0

    def print(self, *, label: str, batch_size: int, num_iters: int, num_steps: int) -> None:
        print(f"\n[PROFILE] {label}")
        print(f"[PROFILE] steps={num_steps}, iters={num_iters}, batch={batch_size}, images={self.total_images}")
        if self.total_images > 0:
            print(f"[PROFILE] denoise: {self.denoise_seconds:.3f}s | {1000*self.denoise_seconds/self.total_images:.2f} ms/image")
            print(f"[PROFILE] decode  : {self.decode_seconds:.3f}s | {1000*self.decode_seconds/self.total_images:.2f} ms/image")
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"[PROFILE] peak memory: {peak_mem:.2f} GB")


# ---------------------------------------------------------------------
# Small utility functions
# ---------------------------------------------------------------------



def get_ckpt_arg(obj, name: str, default=None):
    """Works for argparse.Namespace, dict, or missing old checkpoints."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def infer_dataset_name(ckpt_name: str, fallback: str) -> str:
    for name in ["imagenet", "celebhq", "nuimages", "cityscapes", "ffhq", "anime", "metfaces"]:
        if ckpt_name.startswith(name):
            return name
    return fallback


def infer_model_name(ckpt_name: str, fallback: str) -> str:
    name = ckpt_name.lower()
    if "_s4" in name or "_s-4" in name:
        return "DiT-S/4"
    if "_s2" in name or "_s-2" in name:
        return "DiT-S/2"
    if "_b2" in name or "_b-2" in name:
        return "DiT-B/2"
    if "_xl" in name or "dit-xl" in name:
        return "DiT-XL/2"
    if "_unet" in name:
        return "UNet-M/2"
    return fallback


def infer_transport(ckpt_name: str, fallback: str = "fm") -> str:
    if "_fm_" in ckpt_name or "flowmatch" in ckpt_name:
        return "fm"
    if "diffusion" in ckpt_name or "_ddpm" in ckpt_name:
        return "diffusion"
    return fallback


def infer_vae_pretrained(ckpt_name: str, fallback: int | bool = 0) -> bool:
    name = ckpt_name.lower()
    if "vaepretrained" in name:
        return True
    if "vaeretrained" in name or "vaepr00" in name:
        return False
    return bool(fallback)


def dataset_key_from_name(dataset_name: str) -> str:
    if "imagenet" in dataset_name:
        return "imagenet"
    if "celebhq" in dataset_name:
        return "celebhq_512" if "512" in dataset_name else "celebhq"
    if "ffhq" in dataset_name:
        return "ffhq"
    if "nuimages" in dataset_name:
        return "nuimages"
    if "cityscapes" in dataset_name:
        return "cityscapes"
    if "metfaces" in dataset_name:
        return "metfaces"
    if "anime" in dataset_name:
        return "anime"
    return dataset_name.replace("_precomputed", "")


def raw_dataset_name(dataset_name: str) -> str:
    """FID needs real images, not precomputed latent datasets."""
    if "imagenet" in dataset_name:
        return "imagenet"
    if "celebhq" in dataset_name:
        return "celebhq"
    if "ffhq" in dataset_name:
        return "ffhq"
    if "nuimages" in dataset_name:
        return "nuimages"
    if "cityscapes" in dataset_name:
        return "cityscapes"
    if "metfaces" in dataset_name:
        return "metfaces"
    if "anime" in dataset_name:
        return "anime"
    return dataset_name.replace("_precomputed", "")


def resolve_checkpoint(results_dir: str, ckpt_name: str, epoch: str, *, suffix: str = "") -> CheckpointInfo:
    ckpt_path = os.path.join(results_dir, ckpt_name, "checkpoints", epoch)
    exp_dir = Path(ckpt_path).parts[-3]
    epoch_stem = Path(ckpt_path).stem
    vis_dir = os.path.join(results_dir, exp_dir, "vis", f"vis_{epoch_stem}{suffix}")
    return CheckpointInfo(
        ckpt_path=ckpt_path,
        exp_dir=exp_dir,
        epoch_stem=epoch_stem,
        vis_dir=vis_dir,
    )


def load_checkpoint(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------

def resolve_run_config(args, checkpoint=None, ckpt_name: str = "") -> RunConfig:
    """
    Priority:
      1) new checkpoints: checkpoint["config"]
      2) old checkpoints: checkpoint["args"]
      3) CLI + checkpoint-name inference
    """
    checkpoint = checkpoint or {}
    cfg = checkpoint.get("config", None) if isinstance(checkpoint, dict) else None
    ckpt_args = checkpoint.get("args", None) if isinstance(checkpoint, dict) else None

    if cfg is not None:
        dataset_name = cfg.get("dataset_name", getattr(args, "dataset_name", "celebhq"))
        model_name = cfg.get("model", cfg.get("model_name", getattr(args, "model", "DiT-S/2")))
        image_size = int(cfg.get("image_size", getattr(args, "image_size", 256)))
        num_classes = int(cfg.get("num_classes", getattr(args, "num_classes", 1)))
        transport = cfg.get("transport", getattr(args, "transport", infer_transport(ckpt_name)))
        vae_pretrained = bool(cfg.get("vae_pretrained", getattr(args, "vae_pretrained", infer_vae_pretrained(ckpt_name))))

        is_conditional = bool(cfg.get("is_conditional", "imagenet" in dataset_name))
        class_dropout_prob = float(cfg.get("class_dropout_prob", 0.1 if is_conditional else 0.0))

        dataset_key = dataset_key_from_name(dataset_name)
        in_channels = int(cfg.get("in_channels", 4 if vae_pretrained else in_channels_map[dataset_key]))
        scale_factor = float(cfg.get("scale_factor", 0.18215 if vae_pretrained else scale_factor_map[dataset_key]))

    elif ckpt_args is not None:
        dataset_name = get_ckpt_arg(ckpt_args, "dataset_name", infer_dataset_name(ckpt_name, getattr(args, "dataset_name", "celebhq")))
        model_name = get_ckpt_arg(ckpt_args, "model", infer_model_name(ckpt_name, getattr(args, "model", "DiT-S/2")))
        image_size = int(get_ckpt_arg(ckpt_args, "image_size", getattr(args, "image_size", 256)))
        num_classes = int(get_ckpt_arg(ckpt_args, "num_classes", getattr(args, "num_classes", 1)))
        transport = get_ckpt_arg(ckpt_args, "transport", infer_transport(ckpt_name, getattr(args, "transport", "fm")))
        vae_pretrained = bool(get_ckpt_arg(ckpt_args, "vae_pretrained", infer_vae_pretrained(ckpt_name, getattr(args, "vae_pretrained", 0))))

        dataset_key = dataset_key_from_name(dataset_name)
        is_conditional = "imagenet" in dataset_name
        if is_conditional:
            num_classes = 1000
            class_dropout_prob = 0.1
        else:
            class_dropout_prob = 0.0

        if vae_pretrained:
            in_channels = 4
            scale_factor = 0.18215
        else:
            in_channels = in_channels_map[dataset_key]
            scale_factor = scale_factor_map[dataset_key]

    else: #old checkpoints were saved without a run config 
        dataset_name = infer_dataset_name(ckpt_name, getattr(args, "dataset_name", "celebhq"))
        model_name = infer_model_name(ckpt_name, getattr(args, "model", "DiT-S/2"))
        image_size = int(getattr(args, "image_size", 256))
        num_classes = int(getattr(args, "num_classes", 1))
        transport = infer_transport(ckpt_name, getattr(args, "transport", "fm"))
        vae_pretrained = infer_vae_pretrained(ckpt_name, getattr(args, "vae_pretrained", 0))

        dataset_key = dataset_key_from_name(dataset_name)
        is_conditional = "imagenet" in dataset_name
        if is_conditional:
            num_classes = 1000
            class_dropout_prob = 0.1
        else:
            class_dropout_prob = 0.0

        if vae_pretrained:
            in_channels = 4
            scale_factor = 0.18215
        else:
            in_channels = in_channels_map[dataset_key]
            scale_factor = scale_factor_map[dataset_key]

    return RunConfig(
        dataset_name=dataset_name,
        model_name=model_name,
        transport=transport,
        vae_pretrained=vae_pretrained,
        image_size=image_size,
        latent_size=image_size // 8,
        in_channels=in_channels,
        scale_factor=scale_factor,
        num_classes=num_classes,
        class_dropout_prob=class_dropout_prob,
        is_conditional=is_conditional,
        vae_path = get_ckpt_arg(ckpt_args, "vae_path", getattr(args, "vae_path", "")) or "",
        encoder_type = get_ckpt_arg(ckpt_args, "encoder_type", getattr(args, "encoder_type", "vq_gan_taming")),
    )


# ---------------------------------------------------------------------
# Model, sampler, VAE
# ---------------------------------------------------------------------

def state_dict_from_checkpoint(checkpoint, prefer_ema: bool = True):
    if isinstance(checkpoint, dict):
        if prefer_ema and "ema" in checkpoint:
            return checkpoint["ema"]
        if "model" in checkpoint:
            return checkpoint["model"]
    return checkpoint


def build_model(cfg: RunConfig, learn_sigma: bool, device: torch.device):
    model = DiT_models[cfg.model_name](
        input_size=cfg.latent_size,
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        class_dropout_prob=cfg.class_dropout_prob,
        learn_sigma=learn_sigma,
    ).to(device)
    model.eval()
    return model


def load_model_from_checkpoint(cfg: RunConfig, checkpoint, device: torch.device, *, prefer_ema: bool = True):
    learn_sigma = cfg.transport == "diffusion"
    model = build_model(cfg, learn_sigma, device)
    model.load_state_dict(state_dict_from_checkpoint(checkpoint, prefer_ema=prefer_ema), strict=True)
    model.eval()
    return model


def build_sampler(cfg: RunConfig, args) -> SamplerBundle:
    if cfg.transport != "diffusion":
        transport = create_transport()
        sample_fn = TransportSampler(transport).sample_ode(
            sampling_method=getattr(args, "sampling_method", "euler"),
            num_steps=getattr(args, "fm_steps", 60),
        )
        return SamplerBundle(is_fm=True, sample_fn=sample_fn, num_steps=getattr(args, "fm_steps", 60))

    diffusion = create_diffusion(str(getattr(args, "num_sampling_steps", 250)))

    def sample_fn(z, forward_fn, **model_kwargs):
        samples, _ = diffusion.p_sample_loop(
            forward_fn,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=z.device,
        )
        return samples

    return SamplerBundle(is_fm=False, sample_fn=sample_fn, num_steps=getattr(args, "num_sampling_steps", 250))

def final_state(x):
    # Some samplers return (samples, extra_info)
    if isinstance(x, (tuple, list)):
        x = x[0]
    # FM ODE sampler returns full trajectory: [T, B, C, H, W]
    if x.ndim == 5:
        x = x[-1]

    return x

def load_vae(cfg: RunConfig, device: torch.device):
    """If cfg.vae_path is empty, get_autoencoder resolves it from cfg.dataset_name."""
    vae = get_autoencoder(
        dataset_name=cfg.dataset_name,
        vae_path=cfg.vae_path,
        encoder_type=cfg.encoder_type,
        is_pretrained=cfg.vae_pretrained,
        device=device,
    )
    vae.eval()
    return vae


# ---------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------

def make_latent_batch(args, cfg: RunConfig, device: torch.device):
    n = int(args.batch_size)

    if cfg.is_conditional:
        class_labels = torch.randint(0, cfg.num_classes, (n,), device=device, dtype=torch.long)
        z = torch.randn(n, cfg.in_channels, cfg.latent_size, cfg.latent_size, device=device)

        # Classifier-free guidance: duplicate batch, second half gets null label.
        z = torch.cat([z, z], dim=0)
        y_null = torch.full((n,), cfg.num_classes, device=device, dtype=torch.long)
        y = torch.cat([class_labels, y_null], dim=0)
        return z, {"y": y, "cfg_scale": args.cfg_scale}

    z = torch.randn(n, cfg.in_channels, cfg.latent_size, cfg.latent_size, device=device)
    y = torch.zeros(n, device=device, dtype=torch.long)
    return z, {"y": y}


def decode_latents_to_uint8(samples: torch.Tensor, vae, cfg: RunConfig, ckpt_name: str) -> torch.Tensor:
    if cfg.vae_pretrained:
        samples = vae.decode(samples / cfg.scale_factor).sample
    else:
        samples = vae.decode(samples / cfg.scale_factor)

    samples = samples.clamp(-1.0, 1.0)
    return (((samples * 0.5) + 0.5) * 255).clamp(0, 255).to(torch.uint8)


def sample_loop(model, sampler: SamplerBundle, vae, cfg: RunConfig, args, device: torch.device, ckpt_name: str):
    all_samples = []
    profile = SamplingProfile()

    for _ in tqdm(range(args.num_iters), desc="sampling"):
        z, model_kwargs = make_latent_batch(args, cfg, device)
        forward_fn = model.forward_with_cfg if cfg.is_conditional else model.forward

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time()
        samples = sampler.sample_fn(z, forward_fn, **model_kwargs)
        samples = final_state(samples)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        profile.denoise_seconds += time() - t0

        if cfg.is_conditional:
            samples, _ = samples.chunk(2, dim=0)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time()
        samples = decode_latents_to_uint8(samples, vae, cfg, ckpt_name)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        profile.decode_seconds += time() - t0

        profile.total_images += int(samples.shape[0])
        all_samples.append(samples.cpu())

    return torch.cat(all_samples, dim=0), profile

# when one model evolves the entire trajectory
def generate_single_model(args, device: torch.device):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.reset_peak_memory_stats()

    ckpt_info = resolve_checkpoint(args.results_dir, args.ckpt, args.epoch, suffix=f"_seed{args.seed}")
    checkpoint = load_checkpoint(ckpt_info.ckpt_path)
    cfg = resolve_run_config(args, checkpoint, args.ckpt)

    print("checkpoint:", ckpt_info.ckpt_path)
    print("dataset:", cfg.dataset_name)
    print("model:", cfg.model_name)
    print("transport:", cfg.transport)
    print("scale factor:", cfg.scale_factor)

    sampler = build_sampler(cfg, args)
    model = load_model_from_checkpoint(cfg, checkpoint, device, prefer_ema=True)
    vae = load_vae(cfg, device)

    samples, profile = sample_loop(model, sampler, vae, cfg, args, device, args.ckpt)

    return samples, ckpt_info, cfg, profile

# when two models evolve the trajectory (coarse2fine)
def generate_two_models(args, device: torch.device):
    from fm_training.two_model_wrapper import CoarseFineModel

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.reset_peak_memory_stats()

    ckpt_info = resolve_checkpoint(args.results_dir, args.ckpt, args.epoch, suffix=f"_twomodels_seed{args.seed}")
    fine_info = resolve_checkpoint(args.results_dir2, args.ckpt2, args.epoch2)

    coarse_ckpt = load_checkpoint(ckpt_info.ckpt_path)
    fine_ckpt = load_checkpoint(fine_info.ckpt_path)

    cfg = resolve_run_config(args, coarse_ckpt, args.ckpt)
    fine_cfg = resolve_run_config(args, fine_ckpt, args.ckpt2)
    

    if cfg.transport == "diffusion":
        raise ValueError("coarse-to-fine sampling is intended for flow-matching checkpoints")

    print("coarse checkpoint:", ckpt_info.ckpt_path)
    print("fine checkpoint  :", fine_info.ckpt_path)
    print("coarse model:", cfg.model_name)
    print("fine model  :", fine_cfg.model_name)
    print("t_split:", args.t_split)
    print("scale factor:", cfg.scale_factor)

    coarse = load_model_from_checkpoint(cfg, coarse_ckpt, device, prefer_ema=True)
    fine = load_model_from_checkpoint(fine_cfg, fine_ckpt, device, prefer_ema=True)

    transport = create_transport()
    wrapper = CoarseFineModel(coarse=coarse, fine=fine, transport=transport, t_split=args.t_split)

    sampler = build_sampler(cfg, args)
    vae = load_vae(cfg, device)

    samples, profile = sample_loop(wrapper, sampler, vae, cfg, args, device, args.ckpt)

    if hasattr(wrapper, "timing"):
        t = wrapper.timing
        total = t.coarse_ms + t.fine_ms
        if total > 0:
            print(
                f"[coarse2fine] coarse {t.coarse_ms/1000:.3f}s ({100*t.coarse_ms/total:.1f}%), "
                f"fine {t.fine_ms/1000:.3f}s ({100*t.fine_ms/total:.1f}%)"
            )

    return samples, ckpt_info, cfg, profile


# ---------------------------------------------------------------------
# Saving/loading samples and metrics
# ---------------------------------------------------------------------

def default_samples_path(ckpt_info: CheckpointInfo, mode: str) -> str:
    name = "generated_samples_two_models.pth" if mode == "c2f" else "generated_samples.pth"
    return os.path.join(ckpt_info.vis_dir, name)


def save_samples(samples: torch.Tensor, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(samples.cpu(), path)
    print("saved samples:", path)
    print("shape:", tuple(samples.shape))


def load_samples(path: str) -> torch.Tensor:
    samples = torch.load(path, map_location="cpu", weights_only=False)
    if samples.dtype != torch.uint8:
        samples = samples.clamp(0, 255).to(torch.uint8)
    return samples


def save_visual_grid(samples: torch.Tensor, out_dir: str, *, nrow: int = 6) -> None:
    os.makedirs(out_dir, exist_ok=True)
    grid_path = os.path.join(out_dir, "grid.png")
    save_image(samples.float().div(255.0), grid_path, nrow=nrow, normalize=False)
    print("saved grid:", grid_path)


def load_real_images_uint8(dataset_name: str, image_size: int, device: torch.device, batch_size: int = 512) -> torch.Tensor:
    real_name = raw_dataset_name(dataset_name)
    dataset = get_dataset(split="validation", is_training=False, name=real_name, im_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    reals = []
    for batch in tqdm(loader, desc=f"loading validation images ({real_name})"):
        x = batch["image"].to(device, non_blocking=True)
        x = (((x * 0.5) + 0.5) * 255).clamp(0, 255).to(torch.uint8)
        reals.append(x.cpu())

    return torch.cat(reals, dim=0)


def calculate_fid_metrics(samples: torch.Tensor, reals: torch.Tensor, dataset_name: str, cuda: bool = True):
    import torch_fidelity

    return torch_fidelity.calculate_metrics(
        input1=samples,
        input2=reals,
        cuda=bool(cuda and torch.cuda.is_available()),
        isc=True,
        fid=True,
        kid=False,
        prc=True,
        dataset_name=raw_dataset_name(dataset_name),
        verbose=True,
    )
