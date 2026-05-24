#used in online computation when training starts
import torch
import torch.nn.functional as F

def _load_feat_labels(cluster_path, subset_ind=None):
    res = torch.load(cluster_path, map_location="cpu")
    labels = res["labels"][0]
    K = int(res["k"])

    feat = res["x_org"][0]

    if feat.shape[0] == 1:
        feat = feat[0]

    feat = feat.to(torch.float32).contiguous()
    labels = labels.to(torch.long).contiguous()

    if subset_ind is not None:
        subset_ind = torch.as_tensor(subset_ind, dtype=torch.long)
        feat = feat[subset_ind]
        labels = labels[subset_ind]
        global_ids = subset_ind.clone()
    else:
        global_ids = torch.arange(feat.shape[0], dtype=torch.long)

    return feat, labels, K, global_ids


def _allocate_balanced_counts(labels, K, keep_frac):
    N = labels.numel()
    total_keep = int(N * keep_frac)
    sizes = torch.bincount(labels, minlength=K).tolist()

    base = total_keep // K
    rem = total_keep - base * K
    alloc = [base] * K
    order = sorted(range(K), key=lambda i: sizes[i], reverse=True)
    for i in range(rem):
        alloc[order[i % K]] += 1

    deficit = 0
    for k in range(K):
        if alloc[k] > sizes[k]:
            deficit += alloc[k] - sizes[k]
            alloc[k] = sizes[k]

    if deficit > 0:
        spare = [sizes[k] - alloc[k] for k in range(K)]
        spare_order = sorted(range(K), key=lambda i: spare[i], reverse=True)
        idx = 0
        while deficit > 0:
            k = spare_order[idx % K]
            if spare[k] > 0:
                alloc[k] += 1
                spare[k] -= 1
                deficit -= 1
            idx += 1

    assert sum(alloc) == total_keep
    return alloc


@torch.no_grad()
def _kcenter_local(x, m, seed=0, start="random", normalize=True):
    """
    x: [n, d] CPU
    returns: local indices
    """
    n = x.shape[0]
    m = min(int(m), n)
    if m <= 0:
        return []

    if normalize:
        x = torch.nn.functional.normalize(x, dim=1)

    g = torch.Generator(device="cpu").manual_seed(seed)

    if start == "random":
        first_idx = int(torch.randint(n, (1,), generator=g).item())
    elif start == "maxnorm":
        first_idx = int(torch.norm(x, dim=1).argmax().item())
    else:
        raise ValueError(f"Unknown start mode: {start}")

    selected = [first_idx]
    min_dist = torch.cdist(x, x[first_idx:first_idx+1]).squeeze(1)

    for _ in range(1, m):
        j = int(min_dist.argmax().item())
        selected.append(j)
        dist_new = torch.cdist(x, x[j:j+1]).squeeze(1)
        min_dist = torch.minimum(min_dist, dist_new)

    return selected


@torch.no_grad()
def equal_cluster_kcenter_coreset(cluster_path, keep_frac, subset_ind=None,
                                  seed=0, start="random", normalize=True):
    """
    Balanced coreset baseline:
    - same equal-per-cluster quota allocation as balanced clustering
    - within each cluster, use k-center greedy / farthest-first

    Returns:
        selected_global: list of dataset indices
    """
    feat, labels, K, global_ids = _load_feat_labels(cluster_path, subset_ind=subset_ind)
    alloc = _allocate_balanced_counts(labels, K, keep_frac)

    selected_global = []
    for k in range(K):
        idx = torch.where(labels == k)[0]
        quota = alloc[k]
        if idx.numel() == 0 or quota == 0:
            continue

        xk = feat[idx]
        local_sel = _kcenter_local(
            xk, quota, seed=seed + 10000 * k, start=start, normalize=normalize
        )
        local_sel = torch.as_tensor(local_sel, dtype=torch.long)
        selected_global.extend(global_ids[idx[local_sel]].tolist())

    g = torch.Generator(device="cpu").manual_seed(seed)
    selected_global = torch.tensor(selected_global, dtype=torch.long)
    perm = torch.randperm(selected_global.numel(), generator=g)
    selected_global = selected_global[perm].tolist()
    return selected_global


def load_features_only(cluster_path, subset_ind=None):
    """
    Loads feature matrix from the saved cluster/feature file.

    Returns:
        feat: [N, D] float32 CPU
        global_ids: [N] long CPU
    """
    res = torch.load(cluster_path, map_location="cpu")

    
    feat = res["x_org"][0]

    if feat.shape[0] == 1:
        feat = feat[0]

    feat = feat.to(torch.float32).contiguous()

    if subset_ind is not None:
        subset_ind = torch.as_tensor(subset_ind, dtype=torch.long)
        feat = feat[subset_ind]
        global_ids = subset_ind.clone()
    else:
        global_ids = torch.arange(feat.shape[0], dtype=torch.long)

    return feat, global_ids


@torch.no_grad()
def kcenter_greedy_coreset(
    cluster_path,
    keep_frac=None,
    m=None,
    subset_ind=None,
    seed=0,
    start="random",          # "random" or "maxnorm"
    normalize=True,
    device="cpu",            # "cpu" or "cuda"
    block_size=None,         # e.g. 4096 for lower memory usage
    save_path=None,
):
    """
    Vanilla global k-center greedy / farthest-first traversal.

    Args:
        cluster_path: path to the saved clustering file
        keep_frac: fraction of points to keep, e.g. 0.25
        m: exact number of points to keep; overrides keep_frac if provided
        subset_ind: optional subset of candidate indices
        seed: random seed for initial point if start="random"
        start: initialization mode
        normalize: L2-normalize features before distance computation
        device: where to do the distance computations
        block_size: if set, update distances in blocks to reduce memory
        save_path: optional path to save selected indices

    Returns:
        selected_global: list[int], dataset indices
    """
    feat, global_ids = load_features_only(cluster_path, subset_ind=subset_ind)
    N, D = feat.shape

    if m is None:
        assert keep_frac is not None, "Provide either keep_frac or m."
        m = int(N * keep_frac)
    m = min(int(m), N)

    if m <= 0:
        return []

    if normalize:
        feat = F.normalize(feat, dim=1)

    feat = feat.to(device)
    global_ids = global_ids.to("cpu")

    g = torch.Generator(device="cpu").manual_seed(seed)

    # choose first center
    if start == "random":
        first_idx = int(torch.randint(N, (1,), generator=g).item())
    elif start == "maxnorm":
        # if normalized, all norms are ~1, so maxnorm mainly matters when normalize=False
        first_idx = int(torch.norm(feat, dim=1).argmax().item())
    else:
        raise ValueError(f"Unknown start mode: {start}")

    selected = [first_idx]

    # min_dist[i] = distance from point i to nearest selected center
    first_center = feat[first_idx:first_idx+1]
    if block_size is None:
        min_dist = torch.cdist(feat, first_center).squeeze(1)
    else:
        min_dist_blocks = []
        for s in range(0, N, block_size):
            e = min(s + block_size, N)
            min_dist_blocks.append(torch.cdist(feat[s:e], first_center).squeeze(1))
        min_dist = torch.cat(min_dist_blocks, dim=0)

    # greedy farthest-first updates
    for t in range(1, m):
        new_idx = int(min_dist.argmax().item())
        selected.append(new_idx)

        new_center = feat[new_idx:new_idx+1]

        if block_size is None:
            dist_new = torch.cdist(feat, new_center).squeeze(1)
            min_dist = torch.minimum(min_dist, dist_new)
        else:
            for s in range(0, N, block_size):
                e = min(s + block_size, N)
                dist_block = torch.cdist(feat[s:e], new_center).squeeze(1)
                min_dist[s:e] = torch.minimum(min_dist[s:e], dist_block)

        if (t + 1) % 500 == 0 or (t + 1) == m:
            print(f"[kcenter] selected {t+1}/{m}")

    selected = torch.tensor(selected, dtype=torch.long)
    selected_global = global_ids[selected].tolist()

    if save_path is not None:
        torch.save(
            {
                "selected_idx": torch.tensor(selected_global, dtype=torch.long),
                "method": "kcenter_greedy_global",
                "keep_frac": keep_frac,
                "m": m,
                "normalize": normalize,
                "start": start,
                "seed": seed,
                "cluster_path": cluster_path,
                "subset_ind": None if subset_ind is None else torch.as_tensor(subset_ind, dtype=torch.long),
            },
            save_path,
        )
        print(f"Saved coreset indices to: {save_path}")

    return selected_global