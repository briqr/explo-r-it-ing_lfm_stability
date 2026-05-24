from __future__ import annotations

import os
import re
from dataclasses import dataclass
from glob import glob
from pathlib import Path

from fm_training.distributed import DistEnv
from fm_training.logging_utils import create_logger


@dataclass
class ExperimentPaths:
    experiment_dir: str | None
    checkpoint_dir: str | None
    selected_index_path: str | None
    loss_csv: str | None


def method_name(args, is_fm: bool, is_conditional: bool) -> str:
    name = args.pruning_method

    if args.inverse:
        name = "inverse" + name
    if is_fm:
        name = "fm_" + name

    name = ("vaePretrained_" if args.vae_pretrained else "vaeretrained_") + name
    name = ("conditional" if is_conditional else "unconditional") + name
    return name


def experiment_dir_from_resume(resume_path: str) -> str:
    path = Path(resume_path)
    if path.parent.name == "checkpoints":
        return str(path.parent.parent)
    return str(path.parent)

def setup_experiment(args, env: DistEnv, is_fm: bool, inverse: bool, is_conditional: bool, seed: int):
    pr = args.pruning_ratio

    # Important: parse resume pruning ratio on all ranks, not only rank 0.
    if args.resume:
        experiment_dir = experiment_dir_from_resume(args.resume)

        match = re.search(r"_pruned(\d+)", Path(experiment_dir).name)
        if match is not None:
            pr = float(match.group(1)) / 100.0
    else:
        experiment_dir = None

    if not env.is_main:
        return (
            ExperimentPaths(None, None, None, None),
            create_logger(None, rank=env.rank),
            pr,
        )

    os.makedirs(args.results_dir, exist_ok=True)

    if args.resume:
        assert experiment_dir is not None
    else:
        pr_str = str(pr).replace(".", "")
        name = method_name(args, is_fm=is_fm, is_conditional=is_conditional)

        experiment_dir = f"{args.results_dir}/{args.dataset_name}_{name}_pruned{pr_str}"
        experiment_index = len(glob(f"{args.results_dir}/*"))
        experiment_dir = f"{experiment_dir}_seed{seed}_{experiment_index:03d}"

    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    selected_index_path = os.path.join(experiment_dir, "selected_index.pth")
    loss_csv = os.path.join(experiment_dir, "train_metrics.csv")

    os.makedirs(checkpoint_dir, exist_ok=True)

    if not os.path.exists(loss_csv):
        with open(loss_csv, "w", encoding="utf-8") as f:
            f.write("step,epoch,loss,steps_per_sec\n")

    logger = create_logger(experiment_dir, rank=env.rank)

    if args.resume:
        logger.info(f"***** pr from resume path: {pr}")

    logger.info(f"Experiment directory created at {experiment_dir}")
    print(f"Experiment directory created at {experiment_dir}")

    return ExperimentPaths(
        experiment_dir,
        checkpoint_dir,
        selected_index_path,
        loss_csv,
    ), logger, pr