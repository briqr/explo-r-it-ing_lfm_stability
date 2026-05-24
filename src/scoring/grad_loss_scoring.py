#pre computed offline
import argparse
from pathlib import Path

import torch
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from models import DiT_models
from download import find_model

from fm_training.distributed import setup_distributed, cleanup_distributed, barrier
from fm_training.model import build_transport, build_model

from dataset.data import get_dataset, get_dataset_spec, model_data_config_from_spec

def make_x0_bank(num_noise, x, seed=2025):
    g = torch.Generator(device=x.device).manual_seed(seed)
    bank = torch.randn(
        (num_noise, *x.shape[1:]),
        generator=g,
        device=x.device,
        dtype=x.dtype,
    )
    return bank

def load_scoring_checkpoint(model, experiment_dir, epoch):
    ckpt_path = Path(experiment_dir) / "checkpoints" / epoch
    print("Loading checkpoint:", ckpt_path)

    state_dict = find_model(str(ckpt_path))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return ckpt_path

def build_scoring_loader(args, env):
    dataset = get_dataset(
        name=args.dataset_name,
        split=args.split,
        is_training=False,
        im_size=args.image_size,
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=env.world_size,
        rank=env.rank,
        shuffle=False,
        drop_last=False,
    )

    # we need batch size 1 for grad scoring, otherwise we get an aggregate over the batch
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if env.is_main:
        print("dataset length:", len(dataset))

    return loader

def compute_one_sample_scores(
    model,
    diffusion,
    x,
    model_kwargs,
    x0_bank,
    t_shared,
    ema_loss_per_t,
    ema_grad_per_t,
    model_params,
    args,
):
    loss_total = 0.0
    grad_total = 0.0

    compute_loss_score = args.score_mode in ["loss", "both"]
    compute_grad_score = args.score_mode in ["grad", "both"]

    for k, tk in enumerate(t_shared):
        t_vec = tk.expand(x.size(0)).to(dtype=x.dtype)

        loss_sum = 0.0
        grad_sum = 0.0

        for m in range(args.num_noise):
            x0_m = x0_bank[m].unsqueeze(0)

            loss_dict = diffusion.training_losses(
                model,
                x,
                t=t_vec,
                x0=x0_m,
                model_kwargs=model_kwargs,
            )

            loss = loss_dict["loss"].mean()
            loss_value = float(loss.detach().item())
            loss_sum += loss_value

            if compute_grad_score:
                grads = grad(
                    loss,
                    model_params,
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=True,
                )

                g2 = 0.0
                for gi in grads:
                    if gi is not None:
                        g2 += float((gi.detach().float() ** 2).sum().item())

                grad_sum += g2

                # clear prev grad memory 
                del grads

        loss_mean = loss_sum / args.num_noise

        if compute_loss_score:
            loss_normed = loss_mean / (float(ema_loss_per_t[k]) + 1e-8)
            loss_total += loss_normed
            ema_loss_per_t[k] = (
                args.ema_alpha * ema_loss_per_t[k]
                + (1.0 - args.ema_alpha) * loss_mean
            )

        if compute_grad_score:
            grad_mean = grad_sum / args.num_noise
            grad_normed = grad_mean / (float(ema_grad_per_t[k]) + 1e-8)
            grad_total += grad_normed
            ema_grad_per_t[k] = (
                args.ema_alpha * ema_grad_per_t[k]
                + (1.0 - args.ema_alpha) * grad_mean
            )

    out = {}

    if compute_loss_score:
        out["loss"] = loss_total / len(t_shared)

    if compute_grad_score:
        out["grad"] = grad_total / len(t_shared)

    return out


def score_dataset(
    model,
    diffusion,
    loader,
    scale_factor,
    is_conditional,
    env,
    args,
):
    model_params = [p for p in model.parameters() if p.requires_grad]

    t_shared = None
    x0_bank = None
    ema_loss_per_t = None
    ema_grad_per_t = None

    loss_scores = {}
    grad_scores = {}

    pbar = tqdm(loader, desc=f"rank {env.rank} scoring", disable=not env.is_main)

    for batch_idx, batch in enumerate(pbar):
        x = batch["image"].to(env.device, non_blocking=True) * scale_factor
        y = batch["label"].to(env.device, non_blocking=True)

        if not is_conditional:
            y[...] = 0

        # batch_size=1
        sample_id = int(batch["id"].item())
        model_kwargs = {"y": y}

        if t_shared is None:
            t_shared = torch.linspace(
                0,
                1,
                steps=args.num_t_steps + 2,
                device=env.device,
                dtype=x.dtype,
            )[1:-1]

            x0_bank = make_x0_bank(args.num_noise, x, seed=args.seed)

            ema_loss_per_t = torch.ones_like(t_shared, dtype=torch.float64)
            ema_grad_per_t = torch.ones_like(t_shared, dtype=torch.float64)

        sample_scores = compute_one_sample_scores(
            model=model,
            diffusion=diffusion,
            x=x,
            model_kwargs=model_kwargs,
            x0_bank=x0_bank,
            t_shared=t_shared,
            ema_loss_per_t=ema_loss_per_t,
            ema_grad_per_t=ema_grad_per_t,
            model_params=model_params,
            args=args,
        )

        if "loss" in sample_scores:
            loss_scores[sample_id] = sample_scores["loss"]

        if "grad" in sample_scores:
            grad_scores[sample_id] = sample_scores["grad"]

    return loss_scores, grad_scores



def rank_score_path(score_dir, name, args, rank):
    return Path(score_dir) / (
        f"dict_score_{name}_iter{Path(args.epoch).stem}_"
        f"{args.num_t_steps}t_rank{rank:02d}.pth"
    )


def merged_score_path(score_dir, name, args):
    return Path(score_dir) / (
        f"dict_score_{name}_iter{Path(args.epoch).stem}_"
        f"{args.num_t_steps}t_all.pth"
    )


def save_rank_scores(scores, score_dir, name, args, env):
    if not scores:
        return None

    path = rank_score_path(score_dir, name, args, env.rank)
    torch.save(scores, path)
    print(f"rank {env.rank} saved {name} scores:", path)
    return path


def merge_rank_score_files(score_dir, name, args, world_size):
    merged = {}

    for rank in range(world_size):
        path = rank_score_path(score_dir, name, args, rank)

        if not path.exists():
            print("missing rank score file:", path)
            continue

        part = torch.load(path, map_location="cpu", weights_only=False)

        overlap = set(merged).intersection(part)
        if overlap:
            print(f"warning: {len(overlap)} overlapping sample ids in {path}")

        merged.update(part)

    out_path = merged_score_path(score_dir, name, args)
    torch.save(merged, out_path)

    print(f"merged {name} scores:", out_path)
    print(f"num samples:", len(merged))

    return out_path

def main(args):
    env = setup_distributed()

    try:
        torch.manual_seed(args.seed + env.rank)

        score_dir = Path(args.experiment_dir) / "scores"
        score_dir.mkdir(parents=True, exist_ok=True)

        if env.is_main:
            print("score dir:", score_dir)
            print("dataset:", args.dataset_name)
            print("score mode:", args.score_mode)

        diffusion, is_fm, learn_sigma = build_transport(args)

        dataset_spec = get_dataset_spec(args.dataset_name)
        cfg = model_data_config_from_spec(dataset_spec)

        if env.is_main:
            print("in_channels:", cfg.in_channels)
            print("scale_factor:", cfg.scale_factor)
            print("is_conditional:", cfg.is_conditional)

        model, _ = build_model(
            args=args,
            info=cfg,
            learn_sigma=learn_sigma,
            device=env.device,
        )

        model = model.to(env.device).eval()
        load_scoring_checkpoint(model, args.experiment_dir, args.epoch)

        loader = build_scoring_loader(args, env)

        loss_scores, grad_scores = score_dataset(
            model=model,
            diffusion=diffusion,
            loader=loader,
            scale_factor=cfg.scale_factor,
            is_conditional=cfg.is_conditional,
            env=env,
            args=args,
        )

        save_rank_scores(loss_scores, score_dir, "loss", args, env)
        save_rank_scores(grad_scores, score_dir, "grad", args, env)

        barrier(env)

        if env.is_main:
            if args.score_mode in ["loss", "both"]:
                merge_rank_score_files(score_dir, "loss", args, env.world_size)

            if args.score_mode in ["grad", "both"]:
                merge_rank_score_files(score_dir, "grad", args, env.world_size)

    finally:
        cleanup_distributed(env)


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment_dir", type=str,
        default="celebhq_models/celebhq_precomputed_unconditionalvaeretrained_fm_unpruned_s2_pruned00_seed0",
    )
    

    parser.add_argument("--epoch", type=str, default="0030000.pt")

    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--dataset_name", type=str, default="celebhq_precomputed")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--transport", type=str, default="fm")

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--score-mode", choices=["loss", "grad", "both"], default="loss")
    parser.add_argument("--num-t-steps", type=int, default=8)
    parser.add_argument("--num-noise", type=int, default=2)
    parser.add_argument("--ema-alpha", type=float, default=0.9)

    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())