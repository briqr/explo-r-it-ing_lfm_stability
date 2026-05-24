# https://github.com/lucidrains/magvit2-pytorch/blob/main/magvit2_pytorch/trainer.py


from pathlib import Path
from functools import partial
from contextlib import contextmanager, nullcontext

import torch
from torch import nn
from typing import Optional
from autoencoders.trainer.basicTrainer import BasicTrainer
from accelerate import Accelerator
import tqdm
import torch.distributed as dist
from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    ShardingStrategy,
)
from accelerate.utils import DistributedDataParallelKwargs
import copy
import time
from autoencoders.utils.distributed import get_local_rank


def cycle(dl):
    while True:
        for data in dl:
            yield data


class VideoTokenizerTrainer(BasicTrainer):
    def __init__(
        self,
        model,
        ema_update_fn,
        discriminator,
        optimizer,
        disc_optimizer,
        lr_scheduler,
        disc_lr_scheduler,
        train_loader,
        logger,
        eval_metrics,
        num_train_steps=1,
        grad_accum_every=1,
        apply_gradient_penalty_every=1,
        max_grad_norm=None,
        discr_start_after_step=0.0,
        eval_loader=None,
        eval_every_step=100,
        eval_for_steps=1,
        accelerate_configs={},
    ):
        self.logger = logger

        kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
        # kwargs_handlers = []
        accelerate_configs["gradient_accumulation_steps"] = grad_accum_every
        self.accelerator = Accelerator(
            **accelerate_configs,
            kwargs_handlers=kwargs_handlers,
        )

        # model and exponentially moving averaged model
        self.model = model
        self.discriminator = discriminator
        self.ema_model = copy.deepcopy(model)
        self.ema_update_fn = ema_update_fn

        self.ema_update_fn = ema_update_fn

        self.optimizer = optimizer
        self.disc_optimizer = disc_optimizer
        self.lr_scheduler = lr_scheduler
        self.disc_lr_scheduler = disc_lr_scheduler
        # splitting dataset for validation
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        self.eval_metrics = eval_metrics
        self.eval_every_step = eval_every_step
        self.eval_for_steps = eval_for_steps
        self.checkpoint_every_step = self.logger.checkpoint_every_step

        # training related params
        self.num_train_steps = num_train_steps
        self.grad_accum_every = grad_accum_every
        self.max_grad_norm = max_grad_norm
        self.apply_gradient_penalty_every = apply_gradient_penalty_every

        # prepare for maybe distributed

        (
            self.model,
            self.discriminator,
            self.ema_model,
            self.train_loader,
            self.eval_loader,
            self.optimizer,
            self.disc_optimizer,
        ) = self.accelerator.prepare(
            self.model,
            self.discriminator,
            self.ema_model,
            self.train_loader,
            self.eval_loader,
            self.optimizer,
            self.disc_optimizer,
        )

        # only use adversarial training after a certain number of steps

        self.discr_start_after_step = discr_start_after_step
        # checkpoints and sampled results folder

        checkpoints_folder = self.logger.get_directory("checkpoints")
        models_folder = self.logger.get_directory("models")
        results_folder = self.logger.get_directory("results")

        self.checkpoints_folder = checkpoints_folder
        self.models_folder = models_folder
        self.results_folder = results_folder

        # keep track of train step
        self.step = 0
        # move ema to the proper device

        self.post_norm = partial(self.logger.rescale_image_tensor)

    @property
    def ema_tokenizer(self):
        return self.ema_model

    def tokenize(self, *args, **kwargs):
        return self.ema_tokenizer.tokenize(*args, **kwargs)

    def save(self, path, overwrite=True):
        path = Path(path)
        assert overwrite or not path.exists()

        pkg = dict(
            model=self.model.state_dict(),
            ema_model=self.ema_model.state_dict(),
            discriminator=self.discriminator.state_dict(),
            optimizer=self.optimizer.state_dict(),
            disc_optimizer=self.disc_optimizer.state_dict(),
            lr_scheduler=self.lr_scheduler.state_dict(),
            disc_lr_scheduler=self.disc_lr_scheduler.state_dict(),
            step=self.step,
        )

        torch.save(pkg, str(path))

    def load(self, path):
        path = Path(path)
        if not path.exists():
            self.print(f"Warning!!!     checkpoint {path} not found")

        pkg = torch.load(str(path))
        self.model.load_state_dict(pkg["model"])
        self.ema_model.load_state_dict(pkg["ema_model"])
        self.discriminator.load_state_dict(pkg["discriminator"])
        self.optimizer.load_state_dict(pkg["optimizer"])
        self.disc_optimizer.load_state_dict(pkg["disc_optimizer"])
        self.lr_scheduler.load_state_dict(pkg["lr_scheduler"])
        self.disc_lr_scheduler.load_state_dict(pkg["disc_lr_scheduler"])
        self.step = pkg["step"]

        self.print(f"INFO - loaded checkpoint from {path}")

    def train_step(self, dl_iter):
        self.model.train()
        self.discriminator.train()
        step = self.step
        # determine whether to train adversarially
        train_adversarially = (step + 1) > self.discr_start_after_step
        adversarial_loss_weight = 0.0 if not train_adversarially else None

        # main model

        with self.accelerator.accumulate(self.model), self.accelerator.autocast():
            batch = next(dl_iter)
            data = batch["image"].contiguous()
            loss, tokenizer_output = self.model(data)
            recon = tokenizer_output.pop("recon")
            model_for_loss = self.model.module if hasattr(self.model, "module") else self.model
            loss_gan, disc_output = self.discriminator(
                func="get_gan_loss",
                real=self.post_norm(data),
                fake=self.post_norm(recon),
                last_dec_layer=model_for_loss.get_last_dec_layer(),
                lambda_adversarial_loss=adversarial_loss_weight,
            )
            loss += loss_gan
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.logger.log({"ae_loss": loss}, self.step)
        self.logger.log(tokenizer_output, self.step)
        self.logger.log(disc_output, self.step)
        # self.print(f"Train at step -{step} AE loss: {loss.item():.3f}")
        # _ = {k: v.item() for k, v in tokenizer_output.items() if "loss" in k}
        # self.print(f"Train at step -{step} {_}")
        # _ = {k: v for k, v in disc_output.items() if "loss" in k}
        # self.print(f"Train at step -{step} {_}")
        if self.accelerator.sync_gradients:
            self.lr_scheduler.step()

        self.logger.log({"lr_ae": self.lr_scheduler.get_last_lr()[0]}, self.step)

        # update ema model
        self.ema_update_fn(ema_model=self.ema_model, model=self.model)

        # if adversarial loss is turned off, continue
        if not train_adversarially:
            return

        # =====================================================================
        # discriminator
        self.model.train()
        self.discriminator.train()

        self.disc_optimizer.zero_grad()
        apply_gradient_penalty = step % self.apply_gradient_penalty_every == 0

        with self.accelerator.accumulate(
            self.discriminator
        ), self.accelerator.autocast():
            batch = next(dl_iter)
            data = batch["image"].contiguous()
            _ae_loss, tokenizer_output = self.model(
                data,
            )
            disc_loss, disc_output = self.discriminator(
                func="get_disc_loss",
                real=self.post_norm(data),
                fake=self.post_norm(tokenizer_output["recon"]),
                apply_gradient_penalty=apply_gradient_penalty,
            )

            self.accelerator.backward(disc_loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    self.discriminator.parameters(), self.max_grad_norm
                )
            self.disc_optimizer.step()
            self.disc_optimizer.zero_grad()

        self.logger.log({"disc_loss": disc_loss}, self.step)
        self.logger.log(disc_output, self.step)
        self.print(f"Train at step - {step} discr loss: {disc_loss.item():.3f}")

        # if not self.accelerator.optimizer_step_was_skipped:
        if self.accelerator.sync_gradients:
            self.disc_lr_scheduler.step()
        self.logger.log({"lr_disc": self.disc_lr_scheduler.get_last_lr()[0]}, self.step)

    @torch.no_grad()
    def valid_step(self, dl_iter, valid_batch_size=1):
        self.ema_model.eval()
        self.model.eval()
        self.discriminator.eval()
        # recon_loss = 0.0
        ema_recon_loss = 0.0
        self.eval_metrics.reset()
        if self.eval_for_steps <= 0:
            eval_for_steps = valid_batch_size
        else:
            eval_for_steps = min(valid_batch_size, self.eval_for_steps)
        for _ in range(eval_for_steps):
            batch = next(dl_iter)
            valid_video = batch["image"]
            with self.accelerator.autocast():
                # model_return = self.unwrapped_model(
                #     valid_video, return_recon_loss_only=True
                # )
                # loss = model_return["recon_loss"]
                ema_model_return = self.ema_model(
                    valid_video, return_recon_loss_only=True
                )
                ema_loss = ema_model_return["recon_loss"]

                valid_video = self.post_norm(valid_video)
                ema_model_return["recon"] = self.post_norm(ema_model_return["recon"])

            # this should outside of the autocast
            self.eval_metrics.update(real=valid_video, fake=ema_model_return["recon"])

            # recon_loss += loss / self.eval_for_steps
            ema_recon_loss += ema_loss / self.eval_for_steps
        self.logger.log(
            {
                # "valid/recon_loss": recon_loss,
                "valid/ema_recon_loss": ema_recon_loss,
            },
            self.step,
        )
        # self.print(f"Eval at step - {self.step} validation recon loss {recon_loss:.3f}")
        self.print(
            f"Eval at step - {self.step} validation EMA recon loss {ema_recon_loss:.3f}"
        )

        if "random_latent" in ema_model_return:
            # the kl-gan needs special sampling
            random_latent = ema_model_return["random_latent"]
        else:
            random_latent = torch.randn_like(ema_model_return["quantized"])
        with torch.no_grad():
            # sample = self.unwrapped_model.decode(random_latent)
            sample = self.accelerator.unwrap_model(self.ema_model).decode(random_latent)
            sample = self.post_norm(sample)

        self.wait()
        eval_results = self.eval_metrics.compute_and_reduce()
        if self.is_main:
            ema_img_log_kwargs = {
                "step": self.step,
                "prefix": "valid_",
                "suffix": "_ema-model",
            }
            # self.logger.log_ae_training_images(model_return, **img_log_kwargs)
            self.logger.log_ae_training_images(
                ema_model_return, **ema_img_log_kwargs, rescale=False
            )

            img_log_kwargs = {
                "step": self.step,
                "prefix": "valid_",
                "suffix": "_model",
            }
            self.logger.log_ae_training_images(
                {"data": valid_video, "sample": sample},
                rescale=False,
                **img_log_kwargs,
            )
            self.logger.log(eval_results, self.step)
            self.print(f"Eval at step - {self.step} - {eval_results}")
            # self.unwrapped_model.save(
            #     self.models_folder / f"eval_{self.step}.pt", overwrite=True
            # )
            self.accelerator.unwrap_model(self.ema_model).save(
                self.models_folder / f"eval_{self.step}.pt", overwrite=True
            )
        self.wait()

    def fit(self):
        GB = 1024.0 * 1024.0 * 1024.0
        dl_iter = cycle(self.train_loader)
        valid_dl_iter = cycle(self.eval_loader)
        start_step = self.step
        with tqdm.trange(
            start_step, self.num_train_steps, desc="training steps", dynamic_ncols=True
        ) as pbar:
            for step in pbar:
                self.train_step(dl_iter)
                self.wait()
                if step % self.eval_every_step == 0:
                    self.valid_step(valid_dl_iter, len(self.eval_loader))
                self.logger.log({"step": step}, step=self.step, commit=True)
                self.wait()
                if self.is_main and not (step % self.checkpoint_every_step):
                    checkpoint_num = step // self.checkpoint_every_step
                    checkpoint_path = (
                        self.checkpoints_folder / f"checkpoint.{checkpoint_num}.pt"
                    )
                    self.save(str(checkpoint_path))
                self.wait()
                self.step += 1
                pbar.set_postfix(
                    memory=f"{torch.cuda.max_memory_allocated() / GB :.2f} GB"
                )
        self.accelerator.end_training()

        self.valid_step(valid_dl_iter, len(self.eval_loader))
        if self.is_main:
            checkpoint_path = self.checkpoints_folder / f"checkpoint.latest.pt"
            self.save(str(checkpoint_path))
