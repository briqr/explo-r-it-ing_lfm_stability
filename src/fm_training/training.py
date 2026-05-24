from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import time

import torch

from fm_training.distributed import DistEnv, all_reduce_mean, barrier
from fm_training.experiment import ExperimentPaths
from fm_training.model import unwrap_model, update_ema
from models import DiT_models


@dataclass
class Counters:
    train_steps: int = 0
    log_steps: int = 0
    running_loss: float = 0.0
    start_time: float = 0.0


def get_start_position(args, dataset_len: int, logger):
    if not args.resume:
        return 0, 0

    train_steps = int(Path(args.resume).stem.split("_")[0])
    start_epoch = int(train_steps / (dataset_len / args.global_batch_size))
    logger.info(f"Initial state: step={train_steps}, epoch={start_epoch}")
    print(f"Initial state: step={train_steps}, epoch={start_epoch}")
    return train_steps, start_epoch


def batch_to_inputs(batch, device, scale_factor: float, is_conditional: bool):
    x = batch["image"].to(device, non_blocking=True)
    y = batch["label"].to(device, non_blocking=True)

    if not is_conditional:
        y[...] = 0

    x = x.contiguous() * scale_factor
    return x, {"y": y}


def compute_loss(diffusion, model, x, model_kwargs, is_fm: bool, device):
    if is_fm:
        loss_dict = diffusion.training_losses(model, x, model_kwargs=model_kwargs)
    else:
        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
        loss_dict = diffusion.training_losses(model, x, t, model_kwargs)

    return loss_dict["loss"].mean()


def save_checkpoint(args, model, ema, opt, cfg, paths, train_steps, logger):
    checkpoint = {
        "model": unwrap_model(model).state_dict(),
        "ema": ema.state_dict(),
        "opt": opt.state_dict(),
        "args": args,
        "config": {
            "dataset_name": args.dataset_name,
            "model": args.model,
            "image_size": args.image_size,
            "num_classes": cfg.num_classes,
            "transport": args.transport,
            "vae_pretrained": args.vae_pretrained,
            "in_channels": cfg.in_channels,
            "scale_factor": cfg.scale_factor,
            "class_dropout_prob": cfg.class_dropout_prob,
            "is_conditional": cfg.is_conditional,
        },
    }
    checkpoint_path = f"{paths.checkpoint_dir}/{train_steps:07d}.pt"
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved checkpoint to {checkpoint_path}")


def log_progress(counters: Counters, epoch: int, args, env: DistEnv, paths: ExperimentPaths, logger):
    torch.cuda.synchronize()

    elapsed = time() - counters.start_time
    steps_per_sec = counters.log_steps / elapsed if elapsed > 0 else 0.0

    avg_loss = torch.tensor(counters.running_loss / counters.log_steps, device=env.device)
    avg_loss = all_reduce_mean(avg_loss, env).item()

    logger.info(f"(step={counters.train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
    print(f"(step={counters.train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")

    if env.is_main:
        with open(paths.loss_csv, "a", encoding="utf-8") as f:
            f.write(f"{counters.train_steps},{epoch},{avg_loss:.6f},{steps_per_sec:.4f}\n")

    counters.running_loss = 0.0
    counters.log_steps = 0
    counters.start_time = time()


def train_loop(args, env, model, ema, opt, loader, sampler, dataset_len, diffusion, is_fm, cfg, paths, logger):
    train_steps, start_epoch = get_start_position(args, dataset_len, logger)

    if not args.resume:
        update_ema(ema, unwrap_model(model), decay=0.0)

    model.train()
    ema.eval()

    counters = Counters(train_steps=train_steps, start_time=time())

    logger.info(f"Training for {args.epochs} epochs...")
    print(f"Training for {args.epochs} epochs...")

    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        logger.info(f"Beginning epoch {epoch}...")
        print(f"Beginning epoch {epoch}...")

        for batch in loader:
            x, model_kwargs = batch_to_inputs(batch, env.device, cfg.scale_factor, cfg.is_conditional)
            loss = compute_loss(diffusion, model, x, model_kwargs, is_fm=is_fm, device=env.device)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            update_ema(ema, unwrap_model(model))

            counters.running_loss += loss.item()
            counters.log_steps += 1
            counters.train_steps += 1

            if counters.train_steps % args.log_every == 0:
                log_progress(counters, epoch, args, env, paths, logger)

            if counters.train_steps % args.ckpt_every == 0 and counters.train_steps > 0:
                if env.is_main:
                    save_checkpoint(args, model, ema, opt, cfg, paths, counters.train_steps, logger)
                barrier(env)

    model.eval()


#coarse2fine
def build_fine_model(args, cfg, learn_sigma, device):
    latent_size = args.image_size // 8

    fine_model = DiT_models[args.model_fine](
        input_size=latent_size,
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        class_dropout_prob=cfg.class_dropout_prob,
        learn_sigma=learn_sigma,
    ).to(device)

    ckpt = torch.load(args.fine_ckpt, map_location="cpu", weights_only=False)

    state_dict = ckpt["ema"]

    fine_model.load_state_dict(state_dict, strict=True)
    fine_model.eval()

    for p in fine_model.parameters():
        p.requires_grad = False

    return fine_model

#here model is the coarse (light-weight) model
  
def train_coarse_to_fine_loop(args, env, model, ema, opt, loader, sampler, dataset_len, diffusion, cfg, paths, logger, fine_model):
    train_steps, start_epoch = get_start_position(args, dataset_len, logger)

    if not args.resume:
        update_ema(ema, unwrap_model(model), decay=0.0)
    model.train()
    ema.eval()

    counters = Counters(train_steps=train_steps, start_time=time())

    logger.info(f"Training for {args.epochs} epochs...")
    print(f"Training for {args.epochs} epochs...")

    drift = diffusion.get_drift()
    x_srcs = {} # cache of the diffusion inversion results

    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        for batch in loader:
            x = batch["image"].to(env.device, non_blocking=True)
            y = batch["label"].to(env.device, non_blocking=True)
            sample_ids = batch["id"]

            if not cfg.is_conditional:
                y[...] = 0

            model_kwargs = {"y": y}
            x = x.contiguous() * cfg.scale_factor
            B = x.size(0)

            t_split = args.t_split
            eps = 1e-4
            steps_inv = max(1, int((1.0 - t_split) * args.inversion_steps_per_unit))

            #  Coarse FM loss on time interval [0, t_split)
            t_c = torch.rand(B, device=x.device, dtype=x.dtype) * (t_split - eps) + eps

            loss_fm = diffusion.training_losses(
                model,
                x1=x,
                t=t_c,
                model_kwargs=model_kwargs,
            )["loss"].mean()

            # Seam states obtained from the fine model by backward ODE inversion
            t0 = torch.full((B,), t_split, device=x.device, dtype=x.dtype)

            need_idx = []
            x_src_list = []

            for i, sid in enumerate(sample_ids):
                key = int(sid)
                if key in x_srcs:
                    x_src_list.append(x_srcs[key])
                else:
                    need_idx.append(i)
                    x_src_list.append(None)

            if need_idx:
                y_need = y[need_idx]
                kw_need = {"y": y_need}

                with torch.inference_mode():
                    x_src_need = diffusion.invert_to_t_pf_ode(
                        x[need_idx],
                        fine_model,
                        t0[need_idx],
                        steps=steps_inv,
                        model_kwargs=kw_need,
                    ).cpu()

                for j, i in enumerate(need_idx):
                    key = int(sample_ids[i])
                    x_srcs[key] = x_src_need[j].unsqueeze(0)
                    x_src_list[i] = x_src_need[j:j + 1]

            x_src = torch.cat(x_src_list, dim=0).to(x.device)

            # match both models' velocities at the seam
            with torch.no_grad():
                u_fine = drift(x_src, t0, fine_model, **model_kwargs) # get the model velocity field at (x, t)

            u_coarse = drift(x_src, t0, model, **model_kwargs)

            loss_seam = (u_coarse - u_fine).pow(2).mean()

            loss = loss_fm + args.seam_weight * loss_seam

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            update_ema(ema, unwrap_model(model))

            counters.running_loss += loss.item()
            counters.log_steps += 1
            counters.train_steps += 1

            if counters.train_steps % args.log_every == 0:
                log_progress(counters, epoch, args, env, paths, logger)

            if counters.train_steps % args.ckpt_every == 0 and counters.train_steps > 0:
                if env.is_main:
                    save_checkpoint(args, model, ema, opt, cfg, paths, counters.train_steps, logger)
                barrier(env)
        
        model.eval()