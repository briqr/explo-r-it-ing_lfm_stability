from __future__ import annotations

import json
import os

import numpy as np
import torch

from scoring.coreset import kcenter_greedy_coreset
from scoring.kernel_selection import rff_kernel_herding_on_features, equal_cluster_kernel_herding
from scoring.clustering import (
    cluster_balanced_oversampling,
    drop_clusters,
    equal_cluster,
    furthestmid_cluster,
    random_cluster,
)

from fm_training.distributed import DistEnv, broadcast_indices
from fm_training.experiment import ExperimentPaths


def select_pruning_indices(args, dataset_len: int, cluster_path: str | None, logger):
    method = args.pruning_method
    keep_fraction = 1.0 - args.pruning_ratio
    inverse = bool(args.inverse)


    if method.startswith("coreset"):
        return kcenter_greedy_coreset(cluster_path, keep_fraction)

    if method.startswith("kernel"):
        return rff_kernel_herding_on_features(cluster_path, int(dataset_len * keep_fraction))

    if method.startswith("balanced_cluster_nearest_kernel"):
        return equal_cluster_kernel_herding(cluster_path, keep_fraction)

    if method.startswith("loss"):
        score_path = "samples_scores/dict_score_loss_iter30k_10t_all.pth"
        scores = torch.load(score_path, map_location="cpu")
        return score_based_ind(scores, keep_fraction, largest=not inverse)

    if method.startswith("grad"):
        score_path = "samples_scores/dict_score_grad_iter30k_10t_all.pth"
        scores = torch.load(score_path, map_location="cpu")
        return score_based_ind(scores, keep_fraction, largest=not inverse)


    if method.startswith("female"):
        with open("gender_indices/female_indices.json", "r", encoding="utf-8") as f:
            return json.load(f)

    if method.startswith("male"):
        with open("gender_indices/male_indices.json", "r", encoding="utf-8") as f:
            return json.load(f)

    if method.startswith("random"):
        if args.resume and args.selected_index_path is not None and os.path.exists(args.selected_index_path):
            selected_index = torch.load(args.selected_index_path, map_location="cpu")
        else:
            selected_index = random_opt(dataset_len, keep_fraction)[1]

        return selected_index
    #the following two criteria pertain proportional clustering in the paper
    if method.startswith("cluster_furthest"):
        return furthestmid_cluster(keep_fraction, cluster_path, largest=True)

    if method.startswith("cluster_nearest"):
        return furthestmid_cluster(keep_fraction, cluster_path, largest=False)

    if method.startswith("balanced_cluster_furthest"):
        return equal_cluster(cluster_path, keep_fraction, largest=True)

    if method.startswith("balanced_cluster_nearest"):
        return equal_cluster(cluster_path, keep_fraction, largest=False)

    if method.startswith("balanced_cluster_oversampling"):
        return cluster_balanced_oversampling(cluster_path)

    if method.startswith("drop_cluster"):
        selected_index = drop_clusters(keep_fraction, cluster_path=cluster_path)
        return selected_index

    raise ValueError(f"Unknown pruning method: {method}")


def maybe_prune_dataset(args, dataset, cfg, paths: ExperimentPaths, env: DistEnv, logger):
    if args.pruning_ratio <= 0.0000001:
        return dataset

    # Store this on args so random resume can reuse it without passing more arguments around.
    args.selected_index_path = paths.selected_index_path

    selected_index = None
    if env.is_main:
        selected_index = select_pruning_indices(
            args,
            dataset_len=len(dataset),
            cluster_path=cfg.cluster_path,
            logger=logger,
        )

    selected_index = broadcast_indices(selected_index, env)

    if env.is_main and paths.selected_index_path is not None:
        torch.save(selected_index.cpu(), paths.selected_index_path)

    if args.pruning_ratio > 0.000001:
        dataset = torch.utils.data.Subset(dataset, selected_index.cpu().tolist())

    return dataset


def score_based_ind(grad_scores, pr, largest=True):
    keys = list(grad_scores.keys())
    vals = torch.tensor([grad_scores[k] for k in keys], dtype=torch.float32)
    N = len(keys)
    num_samples = max(1, int(N * pr))
    
    idx_in_vals = torch.topk(vals, k=num_samples, largest=largest).indices
    selected_keys = [keys[i] for i in idx_in_vals.tolist()]
    return selected_keys

def random_opt(len_d, pr):
    subset_size = int(len_d * pr )
    pool = torch.rand(len_d)    
    index = pool.topk(subset_size)
    return index

