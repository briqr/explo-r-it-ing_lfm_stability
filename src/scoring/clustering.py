
"""
#used in online computation when training starts,
as long as the clustering file already exists
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import numpy as np
from collections import OrderedDict
import random


#################################################################################
#                            clustering related functions                         #
#################################################################################

def random_cluster(pr,cluster_path): # return pr of the cluster samples randomly
    res = torch.load(cluster_path, map_location='cpu') 
    labels = res['labels'][0]
    num_cl = res['k']
    centers = res['centers'][0]

    feat = res['x_org'][0]

    all_samples_idx = []
    for l in range(num_cl):
        cluster_sample_idx = torch.where(labels == l)[0]
        if pr < 0.49 and pr > 0.47: #todo
            num_samples = 568
        else:
            num_samples = int(len(cluster_sample_idx) * pr ) 
        pool = torch.rand(len(cluster_sample_idx))
        index = pool.topk(num_samples)[1]
        all_samples_idx.extend(cluster_sample_idx[index])
    return all_samples_idx



def furthestmid_cluster(pr, cluster_path='clusters/cluster_clip_24.pth', largest=True): # return pr of the cluster samples randomly

    res = torch.load(cluster_path, map_location='cpu') 
    labels = res['labels'][0]
    num_cl = res['k']
    centers = res['centers'][0]
    if 'dino' not in cluster_path or 'imagenet' in cluster_path:
        feat = res['x_org']
    else:
        feat = res['x_org'][0]
    if feat.shape[0] == 1:
        feat = feat[0]
    all_samples_idx = []
    for l in range(num_cl):
        cluster_sample_idx = torch.where(labels == l)[0]
        num_samples = int(len(cluster_sample_idx) * (pr) ) 
        dist = torch.norm(feat[cluster_sample_idx] - centers[l], dim=1)
        index = dist.topk(num_samples, largest=largest)[1]
        all_samples_idx.extend(cluster_sample_idx[index])

    return all_samples_idx



def drop_clusters(pr, cluster_path='clusters/cluster_clip_24.pth', seed=42):  # fixed seed added
    res = torch.load(cluster_path, map_location='cpu') 
    labels = res['labels'][0]
    num_cl = res['k']
    centers = res['centers'][0]

    if 'dino' not in cluster_path or 'imagenet' in cluster_path:
        feat = res['x_org']
    else:
        feat = res['x_org'][0]

    if feat.shape[0] == 1:
        feat = feat[0]


    cluster_ids = torch.randperm(num_cl).tolist()

    # Select clusters to keep
    num_keep = int(num_cl * pr)
    keep_clusters = set(cluster_ids[:num_keep])
    print('***************Keeping clusters:', keep_clusters)

    all_samples_idx = []
    for l in range(num_cl):
        if l not in keep_clusters:
            continue
        cluster_sample_idx = torch.where(labels == l)[0]
        num_samples = int(len(cluster_sample_idx) * pr)

        dist = torch.norm(feat[cluster_sample_idx] - centers[l], dim=1)
        index = dist.topk(num_samples, largest=False)[1]
        all_samples_idx.extend(cluster_sample_idx[index])

    return all_samples_idx

def furthestnearest_cluster(pr, cluster_path): 
    # S: dict
    # size: the subset size
    res = torch.load(cluster_path, map_location='cpu')
    labels = res.labels[0]
    num_cl = res.k
    centers = res.centers[0]
    feat = res.x_norm[0]
    if feat.shape[0] == 1:
        feat = feat[0]
    all_samples_idx = []
    rat = 0.2
    for l in range(num_cl):
        cluster_sample_idx = torch.where(labels == l)[0]
        num_samples = int(len(cluster_sample_idx) * (pr-rat) ) 
        dist = torch.norm(feat[cluster_sample_idx] - centers[l], dim=1)
        index = dist.topk(num_samples, largest=True)[1]
        all_samples_idx.extend(cluster_sample_idx[index])

        num_samples = int(len(cluster_sample_idx) * (rat) ) 
        index = dist.topk(num_samples, largest=False)[1]
        all_samples_idx.extend(cluster_sample_idx[index])

    return all_samples_idx


def cluster_balanced_oversampling(cluster_path, seed=0):
    """
    Oversample clusters so that each cluster contributes equally many samples.
    Returns the oversampled indices (with repeats).
    """


    res = torch.load(cluster_path, map_location='cpu')
    labels = res['labels'][0]
    num_cl = res['k']

    # handle features shape differences across formats
    if 'dino' not in cluster_path or 'imagenet' in cluster_path:
        feat = res['x_org']
    else:
        feat = res['x_org'][0]
    if feat.shape[0] == 1:
        feat = feat[0]

    # count samples per cluster
    cluster_indices = [torch.where(labels == l)[0].tolist() for l in range(num_cl)]
    cluster_sizes = [len(idx) for idx in cluster_indices]

    # choose the target cluster size (largest cluster)
    max_size = max(cluster_sizes)

    g = torch.Generator()
    g.manual_seed(seed)

    oversampled_indices = []

    for l, idx in enumerate(cluster_indices):
        if len(idx) == 0:
            continue

        repeat_factor = max_size // len(idx)
        remainder = max_size % len(idx)

        oversampled = idx * repeat_factor
        perm = torch.randperm(len(idx), generator=g).tolist()
        oversampled += [idx[i] for i in perm[:remainder]]
        oversampled_indices.extend(oversampled)

    return oversampled_indices



def one_cluster(cluster_path='clusters/cluster_clip_24.pth', cluster_idx=0): # return pr of the cluster samples randomly
    # S: dict
    # size: the subset size
    #res = torch.load('cluster_dino_22cluster.pth', map_location='cpu')
    res = torch.load(cluster_path, map_location='cpu') 
    labels = res['labels'][0]
    num_cl = res['k']
    centers = res['centers'][0]
    if 'dino' not in cluster_path or 'imagenet' in cluster_path:
        feat = res['x_org']
    else:
        feat = res['x_org'][0]
    if feat.shape[0] == 1:
        feat = feat[0]
    print('**feat shape', feat.shape)



    cluster_sample_idx = torch.where(labels == cluster_idx)[0]

    return all_samples_idx

def equal_cluster(cluster_path='clusters/cluster_clip_24.pth', pr=-1, largest=True, is_random=False, subset_ind=None): # return pr of the cluster samples randomly
    # S: dict
    # size: the subset size
    #res = torch.load('cluster_dino_22cluster.pth', map_location='cpu')
    res = torch.load(cluster_path, map_location='cpu') 
    labels = res['labels'][0]
    num_cl = res['k']
    centers = res['centers'][0]
    if 'dino' not in cluster_path or 'imagenet' in cluster_path:
        feat = res['x_org']
    else:
        feat = res['x_org'][0]
    if feat.shape[0] == 1:
        feat = feat[0]
    print('**feat shape', feat.shape)
    if subset_ind is not None:
        feat = feat[subset_ind]
        labels = labels[subset_ind]
        
    all_samples_idx = []
    min_size = 1000000
    total_samples = 0
    for l in range(num_cl):
        cluster_sample_idx = torch.where(labels == l)[0]
        total_samples += len(cluster_sample_idx)
        if len(cluster_sample_idx) < min_size:
            min_size = len(cluster_sample_idx)
    print('***min size cluster', min_size)
    #print('pr is ', pr)
    if pr > 0:
        samples_per_cluster = int((total_samples/num_cl)*pr)
        print('***samples per cluster', samples_per_cluster)
        repeat = 1
        total_cumul = samples_per_cluster
        while (repeat < 2):
            repeat += 1
            cumulative = 0
            for l in range(num_cl):
                cluster_sample_idx = torch.where(labels == l)[0]
                if len(cluster_sample_idx) < samples_per_cluster:
                    cumulative += samples_per_cluster - len(cluster_sample_idx) 
            total_cumul += cumulative//num_cl
            print('***total_cumul', total_cumul)
            break
        samples_per_cluster = total_cumul
    else:
        samples_per_cluster = min_size
    print('***   min size cluster', min_size)
    for l in range(num_cl):
        cluster_sample_idx = torch.where(labels == l)[0]
        num_samples = min_size
        if pr > 0:
            #num_samples = int(len(cluster_sample_idx) * (pr) ) 
            num_samples = samples_per_cluster
        if not is_random:
            dist = torch.norm(feat[cluster_sample_idx] - centers[l], dim=1)
            index = dist.topk(min(num_samples,len(cluster_sample_idx)) , largest=largest)[1]
        else:
            pool = torch.rand(len(cluster_sample_idx))
            index = pool.topk(min(num_samples,len(cluster_sample_idx)))[1]
        all_samples_idx.extend(cluster_sample_idx[index])
        #break
    return all_samples_idx


def mid_cluster(pr, cluster_path='clusters/cluster_clip_24.pth'): # return pr of the cluster samples randomly
    # S: dict
    # size: the subset size
    #res = torch.load('cluster_dino_22cluster.pth', map_location='cpu')
    res = torch.load(cluster_path, map_location='cpu') 
    labels = res['labels'][0]
    num_cl = res['k']
    centers = res['centers'][0]
    if 'dino' not in cluster_path or 'imagenet' in cluster_path:
        feat = res['x_org']
    else:
        feat = res['x_org'][0]
    if feat.shape[0] == 1:
        feat = feat[0]
    print('**feat shape', feat.shape)
    all_samples_idx = []
    pruning_pr = (1-pr)/2
    for l in range(num_cl):
        cluster_sample_idx = torch.where(labels == l)[0]
        print('***cluster size', len(cluster_sample_idx))
        num_samples = int(len(cluster_sample_idx) * (pr+pruning_pr) ) 
        dist = torch.norm(feat[cluster_sample_idx] - centers[l], dim=1)
        index = dist.topk(num_samples, largest=True)[1]
        print('***index size', len(index))
        number_furthest =  int(len(cluster_sample_idx) * (pruning_pr))
        print('***number_furthest', number_furthest)
        index = index[number_furthest:]
        all_samples_idx.extend(cluster_sample_idx[index])

    return all_samples_idx

import torch

def print_cluster(cluster_path='nuscenes_embeddings/cluster_24.pth'):
    # Load the cluster data
    res = torch.load(cluster_path)
    labels = res['labels'][0]
    num_cl = res['k']
    
    total_samples = 0
    centers = res['centers'][0]
    
    distances = []  # Initialize list to store distances
    
    # Print information about each cluster
    for l in range(num_cl):
        cluster_sample_idx = torch.where(labels == l)[0]
        total_samples += len(cluster_sample_idx)
        print(f"Cluster {l}: ", len(cluster_sample_idx))
    
    # Print total samples
    print('Total samples:', total_samples)
    
    # Calculate and store pairwise distances between cluster centers
    print("\nPairwise distances between cluster centers (distance <= 3):")
    for i in range(num_cl):
        for j in range(i + 1, num_cl):  # Avoid redundant calculations and diagonal
            distance = torch.norm(centers[i] - centers[j])  # Euclidean distance
            distances.append(distance.item())  # Append distance to the list
            if distance <= 3:  # Check if distance is less than or equal to 3
                print(f"Distance between cluster {i} and cluster {j}: {distance.item():.4f}")
    
    # Convert distances list to a tensor
    distances_tensor = torch.tensor(distances)
    
    # Print min, max, mean, and median of distances
    print('\nMin distance, Max distance, Mean distance, Median distance:')
    print(
        f"Min: {torch.min(distances_tensor):.4f}, "
        f"Max: {torch.max(distances_tensor):.4f}, "
        f"Mean: {torch.mean(distances_tensor):.4f}, "
        f"Median: {torch.median(distances_tensor).item():.4f}"
    )




#############kernel cluster##################
import math
import torch

def _load_cluster_feat_labels(cluster_path, subset_ind=None):
    res = torch.load(cluster_path, map_location="cpu")
    labels = res["labels"][0]
    K = int(res["k"])
    centers = res["centers"][0]

    # your file format logic
    if "dino" not in cluster_path or "imagenet" in cluster_path:
        feat = res["x_org"]
    else:
        feat = res["x_org"][0]
    if feat.shape[0] == 1:
        feat = feat[0]

    if subset_ind is not None:
        subset_ind = torch.as_tensor(subset_ind, dtype=torch.long)
        feat = feat[subset_ind]
        labels = labels[subset_ind]

    feat = feat.to(torch.float32).contiguous()
    centers = centers.to(torch.float32).contiguous()
    labels = labels.to(torch.long).contiguous()
    return feat, labels, centers, K

def allocate_balanced_counts(labels, K, keep_frac):
    """
    Allocate per-cluster counts so that:
      - total kept = floor(N * keep_frac)
      - as equal as possible across clusters
      - if a cluster is too small, redistribute its deficit across other clusters
    """
    N = labels.numel()
    total_keep = int(N * keep_frac)
    sizes = torch.bincount(labels, minlength=K).tolist()

    base = total_keep // K
    rem = total_keep - base * K
    alloc = [base] * K
    # distribute remainder (+1) to the largest clusters first (more likely to have capacity)
    order = sorted(range(K), key=lambda i: sizes[i], reverse=True)
    for i in range(rem):
        alloc[order[i % K]] += 1

    # clamp to capacity, compute deficit
    deficit = 0
    for k in range(K):
        if alloc[k] > sizes[k]:
            deficit += alloc[k] - sizes[k]
            alloc[k] = sizes[k]

    # redistribute deficit to clusters with spare capacity
    if deficit > 0:
        spare = [sizes[k] - alloc[k] for k in range(K)]
        # keep adding 1 to clusters with spare capacity (largest spare first)
        # until deficit consumed
        spare_order = sorted(range(K), key=lambda i: spare[i], reverse=True)
        idx = 0
        while deficit > 0:
            k = spare_order[idx % K]
            if spare[k] > 0:
                alloc[k] += 1
                spare[k] -= 1
                deficit -= 1
            idx += 1
            # if no spare anywhere, we stop (shouldn't happen unless total_keep > N)
            if idx > 10_000_000:
                break

    assert sum(alloc) == total_keep, (sum(alloc), total_keep)
    return alloc  # list length K

def _median_sigma(x, max_samples=2048, g=None):
    """
    Median heuristic for RBF sigma on a tensor x: [n,d] on CPU.
    sigma = sqrt(median(||xi-xj||^2)/2)
    """
    n = x.size(0)
    if n <= 2:
        return 1.0
    m = min(n, max_samples)
    if g is None:
        idx = torch.randperm(n)[:m]
    else:
        idx = torch.randperm(n, generator=g)[:m]
    xs = x[idx]
    d2 = torch.cdist(xs, xs).pow(2)
    tri = d2[torch.triu_indices(m, m, offset=1).unbind()]
    med = tri.median().item()
    return math.sqrt(max(med, 1e-12) / 2.0)

class RFF_RBF(torch.nn.Module):
    def __init__(self, in_dim, num_features, sigma, g=None, dtype=torch.float32):
        super().__init__()
        # sample W,b once
        if g is None:
            W = torch.randn(num_features, in_dim, dtype=dtype) / sigma
            b = 2.0 * math.pi * torch.rand(num_features, dtype=dtype)
        else:
            W = torch.randn(num_features, in_dim, generator=g, dtype=dtype) / sigma
            b = 2.0 * math.pi * torch.rand(num_features, generator=g, dtype=dtype)
        self.register_buffer("W", W)
        self.register_buffer("b", b)
        self.scale = math.sqrt(2.0 / num_features)

    def forward(self, x):
        # x: [n,d] float32 CPU
        return self.scale * torch.cos(x @ self.W.t() + self.b)

