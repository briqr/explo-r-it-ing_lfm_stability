#computed online, as long as the clustering file already exists
from torch.utils.data import DataLoader
import torch
import math

import torch
from torch.utils.data import Dataset
import torch, math
from scoring.clustering import allocate_balanced_counts
def flatten(x):
    return x.reshape(x.shape[0], -1)

class RFF_RBF(torch.nn.Module):
    def __init__(self, in_dim, num_features, sigma, device="cuda", dtype=torch.float32):
        super().__init__()
        W = torch.randn(num_features, in_dim, device=device, dtype=dtype) / sigma
        b = 2.0 * math.pi * torch.rand(num_features, device=device, dtype=dtype)
        self.register_buffer("W", W)
        self.register_buffer("b", b)
        self.scale = math.sqrt(2.0 / num_features)

    def forward(self, x):
        x = flatten(x)
        return self.scale * torch.cos(x @ self.W.t() + self.b)

@torch.no_grad()
def median_sigma_from_loader(loader, device="cuda", max_samples=2048):
    xs = []
    for batch in loader:
        xs.append(batch["image"])
        if sum(t.size(0) for t in xs) >= max_samples:
            break
    x = torch.cat(xs, 0).to(device)
    x = flatten(x)
    idx = torch.randperm(x.size(0), device=device)[:min(x.size(0), max_samples)]
    x = x[idx]
    d2 = torch.cdist(x, x).pow(2)
    tri = d2[torch.triu_indices(d2.size(0), d2.size(1), offset=1).unbind()]
    med = tri.median().item()
    return math.sqrt(max(med, 1e-12) / 2.0)

@torch.no_grad()
def rff_kernel_herding_indices(dataset, m, batch_size=256, num_workers=4,
                               device="cuda", rff_dim=1024, sigma=None):
    """
    Greedy herding to match mean embedding (approx MMD minimization).
    Returns list of dataset 'id' values (your dataset provides batch['id']).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, drop_last=False)

    # infer sigma + in_dim
    first = next(iter(loader))
    in_dim = flatten(first["image"]).shape[1]
    if sigma is None:
        sigma = median_sigma_from_loader(loader, device=device)

    rff = RFF_RBF(in_dim=in_dim, num_features=rff_dim, sigma=sigma, device=device)

    # Pass 1: compute mu_full
    mu = torch.zeros(rff_dim, device=device)
    n = 0
    for batch in loader:
        x = batch["image"].to(device)
        z = rff(x)  # [B,D]
        mu += z.sum(0)
        n += z.size(0)
    mu /= max(n, 1)

    # Pass 2: compute z(x) for all points (store on CPU to save GPU mem)
    Z_list, id_list = [], []
    for batch in loader:
        x = batch["image"].to(device)
        z = rff(x).cpu()
        Z_list.append(z)
        id_list.append(batch["id"].cpu())
    Z = torch.cat(Z_list, 0)          # [N,D] on CPU
    ids = torch.cat(id_list, 0)       # [N]

    # Greedy selection: keep running mean of selected embeddings
    selected = []
    selected_mask = torch.zeros(Z.size(0), dtype=torch.bool)
    mean_sel = torch.zeros(rff_dim)

    # objective: minimize ||mu - mean_sel||^2  -> greedily pick point that best reduces it
    # classic herding picks argmax <z_i, r> where r = mu - mean_sel
    for k in range(m):
        r = (mu.cpu() - mean_sel)     # residual in CPU
        scores = (Z @ r)              # [N]
        scores[selected_mask] = -1e9
        j = scores.argmax().item()
        selected.append(int(ids[j].item()))
        selected_mask[j] = True
        mean_sel = (mean_sel * k + Z[j]) / (k + 1)

    return selected#, {"sigma": float(sigma), "rff_dim": rff_dim}


def load_clip_features(cluster_path, subset_ind=None):
    res = torch.load(cluster_path, map_location="cpu")
    labels = res["labels"][0] if "labels" in res else None
    K = int(res["k"]) if "k" in res else None

    feat = res["x_org"][0]
    if feat.shape[0] == 1:
        feat = feat[0]

    feat = feat.to(torch.float32).contiguous()  # [N, D]

    if subset_ind is not None:
        subset_ind = torch.as_tensor(subset_ind, dtype=torch.long)
        feat = feat[subset_ind]
        if labels is not None:
            labels = labels[subset_ind]

    return feat, labels, K

def median_sigma(x, max_samples=4096, seed=0):
    # x: [N,D] CPU float32
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = x.size(0)
    m = min(n, max_samples)
    idx = torch.randperm(n, generator=g)[:m]
    xs = x[idx]
    d2 = torch.cdist(xs, xs).pow(2)
    tri = d2[torch.triu_indices(m, m, offset=1).unbind()]
    med = tri.median().item()
    return math.sqrt(max(med, 1e-12) / 2.0)

class RFF_RBF(torch.nn.Module):
    def __init__(self, in_dim, num_features, sigma, seed=0):
        super().__init__()
        g = torch.Generator(device="cpu").manual_seed(seed)
        W = torch.randn(num_features, in_dim, generator=g, dtype=torch.float32) / sigma
        b = 2.0 * math.pi * torch.rand(num_features, generator=g, dtype=torch.float32)
        self.register_buffer("W", W)
        self.register_buffer("b", b)
        self.scale = math.sqrt(2.0 / num_features)

    def forward(self, x):
        # x: [N,D]
        return self.scale * torch.cos(x @ self.W.t() + self.b)

@torch.no_grad()
def rff_kernel_herding_on_features(cluster_path, m, rff_dim=512, sigma=None, seed=0, block=8192):
    """
    feat: [N,D] CPU float32
    returns: selected indices (0..N-1) length m
    """
    feat = load_clip_features(cluster_path)[0]  # [N,D] CPU
    N, D = feat.shape
    m = min(int(m), N)

    if sigma is None:
        sigma = median_sigma(feat, seed=seed)

    rff = RFF_RBF(in_dim=D, num_features=rff_dim, sigma=sigma, seed=seed)

    # Compute Z in blocks to avoid huge peak RAM if N is large
    Z = []
    for i in range(0, N, block):
        Z.append(rff(feat[i:i+block]))
    Z = torch.cat(Z, 0)  # [N, rff_dim]

    mu = Z.mean(0)              # [rff_dim]
    mean_sel = torch.zeros_like(mu)
    selected = torch.empty(m, dtype=torch.long)
    selected_mask = torch.zeros(N, dtype=torch.bool)

    for k in range(m):
        r = mu - mean_sel        # [rff_dim]
        scores = Z @ r           # [N]
        scores[selected_mask] = -1e9
        j = int(scores.argmax().item())
        selected[k] = j
        selected_mask[j] = True
        mean_sel = (mean_sel * k + Z[j]) / (k + 1)

        if (k+1) % 1000 == 0:
            print(f"[herding] selected {k+1}/{m}")

    return selected.tolist()#, {"sigma": float(sigma), "rff_dim": int(rff_dim)}

def flatten_latents(x):  # [B,C,H,W] -> [B,D]
    return x.reshape(x.shape[0], -1)

def rbf_kernel(x, y, sigma):
    # x: [n,d], y: [m,d]
    d2 = torch.cdist(x, y).pow(2)
    return torch.exp(-d2 / (2.0 * sigma * sigma + 1e-12))


# by equal we mean balanced
@torch.no_grad()
def equal_cluster_kernel_herding(
    cluster_path,
    keep_frac,                 # e.g. keep_frac = 1 - pr (pruning_ratio)
    anchors_per_cluster=128,    # kernel herding anchors per cluster
    rff_dim=256,
    fill_mode="nearest",       # "nearest" | "furthest" | "random"
    subset_ind=None,
    seed=0,
):
    """
    Cluster-balanced selection, but within each cluster:
      - pick 'anchors_per_cluster' by kernel herding in CLIP space (RFF-RBF)
      - fill remaining quota by a cheap rule (nearest/furthest/random wrt centroid)

    Returns global indices (relative to subset_ind if provided, else dataset indices).
    """
    feat, labels, centers, K = _load_cluster_feat_labels(cluster_path, subset_ind=subset_ind)
    #feat = feat[0]
    print('***feat shape', feat.shape)
    N, d = feat.shape
    alloc = allocate_balanced_counts(labels, K, keep_frac)

    g = torch.Generator(device="cpu").manual_seed(seed)
    selected_global = []

    for k in range(K):
        idx = torch.where(labels == k)[0]  # indices into feat/labels arrays
        n_k = idx.numel()
        if n_k == 0:
            continue

        quota = alloc[k]
        if quota == 0:
            continue

        # anchors (cap to quota)
        a = min(int(anchors_per_cluster), quota)
        xk = feat[idx]  # [n_k, d] CPU float32

        # compute anchors by kernel herding
        # use per-cluster sigma heuristic (more stable than global when clusters differ)
        sigma_k = _median_sigma(xk, g=g)
        anchors_local = _kernel_herding_anchor_indices(
            xk, m=a, rff_dim=rff_dim, sigma=sigma_k, seed=seed + 10_000 * k
        )
        anchors_local = torch.as_tensor(anchors_local, dtype=torch.long)
        anchors_global = idx[anchors_local]
        chosen = set(anchors_global.tolist())

        # fill remaining with cheap rule
        remain = quota - len(chosen)
        if remain > 0:
            remaining_pool = idx[~torch.isin(idx, anchors_global)]
            if remaining_pool.numel() > 0:
                if fill_mode == "random":
                    perm = torch.randperm(remaining_pool.numel(), generator=g)
                    fill = remaining_pool[perm[:remain]]
                else:
                    c = centers[k].view(1, -1)  # [1,d]
                    dist = torch.norm(feat[remaining_pool] - c, dim=1)
                    largest = (fill_mode == "furthest")
                    top = dist.topk(k=min(remain, remaining_pool.numel()), largest=largest).indices
                    fill = remaining_pool[top]
                for j in fill.tolist():
                    chosen.add(int(j))

        selected_global.extend(list(chosen))

    # shuffle to avoid any ordering artifacts
    selected_global = torch.tensor(selected_global, dtype=torch.long)
    perm = torch.randperm(selected_global.numel(), generator=g)
    selected_global = selected_global[perm].tolist()

    return selected_global

@torch.no_grad()
def _kernel_herding_anchor_indices(x, m, rff_dim=256, sigma=None, seed=0):
    """
    True greedy herding but only for small m (anchors). Runs on CPU.
    x: [n,d]
    returns: list of local indices length m
    """
    n, d = x.shape
    if m <= 0:
        return []
    m = min(m, n)
    g = torch.Generator(device="cpu").manual_seed(seed)

    if sigma is None:
        sigma = _median_sigma(x, g=g)

    rff = RFF_RBF(in_dim=d, num_features=rff_dim, sigma=sigma, g=g, dtype=torch.float32)
    Z = rff(x)  # [n, rff_dim]
    mu = Z.mean(0)  # [rff_dim]

    selected = []
    selected_mask = torch.zeros(n, dtype=torch.bool)
    mean_sel = torch.zeros_like(mu)

    for k in range(m):
        r = mu - mean_sel  # residual
        scores = Z @ r     # [n]
        scores[selected_mask] = -1e9
        j = int(scores.argmax().item())
        selected.append(j)
        selected_mask[j] = True
        mean_sel = (mean_sel * k + Z[j]) / (k + 1)

    return selected
