# sharded_latents_dataset.py (patched)
import os, numpy as np, torch
from glob import glob

class PreencodedLatentsShards(torch.utils.data.Dataset):
    """
    Load many latent shards with NumPy memmap. Supports both:
      - latents_00000.npy  (old single-rank style)
      - r00_00000_latents.npy (multi-rank style)
    and finds matching labels/ids by suffix replacement.
    """
    def __init__(self, root_dir="/e/scratch/multiscale-wm/briq/imagenet/precomputed_latents/vq_imagenet_shards",
                 split="train"):
        super().__init__()
        # Collect both naming patterns
        lat_paths = sorted(glob(os.path.join(root_dir, "latents_*.npy")))
        lat_paths += sorted(glob(os.path.join(root_dir, "*_latents.npy")))
        # Deduplicate while preserving order
        seen = set(); lat_paths = [p for p in lat_paths if not (p in seen or seen.add(p))]

        # Helper to find label/id path corresponding to a latent path
        def mate(path, kind):
            dirname, fname = os.path.split(path)
            if fname.startswith("latents_") and fname.endswith(".npy"):
                # latents_00000.npy -> labels_00000.npy
                return os.path.join(dirname, fname.replace("latents_", f"{kind}_"))
            if fname.endswith("_latents.npy"):
                # r00_00000_latents.npy -> r00_00000_labels.npy
                return os.path.join(dirname, fname.replace("_latents.npy", f"_{kind}.npy"))
            # Last-resort generic replacement if someone changes the prefix
            if "latents" in fname:
                return os.path.join(dirname, fname.replace("latents", kind))
            raise RuntimeError(f"Cannot derive '{kind}' path from {path}")

        # Keep only complete triplets (latents, labels, ids)
        triples = []
        for lp in lat_paths:
            lab = mate(lp, "labels")
            ids = mate(lp, "ids")
            if os.path.exists(lab) and os.path.exists(ids):
                triples.append((lp, lab, ids))
            else:
                # Skip incomplete shards quietly; uncomment to debug:
                # print(f"[warn] skipping incomplete shard: {lp}")
                pass

        if not triples:
            raise FileNotFoundError(
                f"No complete shards found in {root_dir}. "
                f"Expected files like 'latents_00000.npy' + 'labels_00000.npy' + 'ids_00000.npy' "
                f"or 'r00_00000_latents.npy' + 'r00_00000_labels.npy' + 'r00_00000_ids.npy'."
            )

        # Memmap all shards
        self.latents = [np.load(lp,  mmap_mode="r") for lp,_,_ in triples]
        self.labels  = [np.load(lab, mmap_mode="r") for _,lab,_ in triples]
        self.ids     = [np.load(ids, mmap_mode="r") for _,_,ids in triples]

        # Validate shapes & build index map
        self.sizes = []
        for s, (L, Y, I) in enumerate(zip(self.latents, self.labels, self.ids)):
            if not (L.shape[0] == Y.shape[0] == I.shape[0]):
                raise ValueError(f"Shard {s} size mismatch: {L.shape[0]} vs {Y.shape[0]} vs {I.shape[0]}")
            self.sizes.append(L.shape[0])
        import numpy as _np
        self.cum = _np.cumsum([0] + self.sizes)  # [0, N0, N0+N1, ...]
        self.total = int(self.cum[-1])

    def __len__(self):
        return self.total

    def _locate(self, idx):
        import numpy as _np
        s = _np.searchsorted(self.cum, idx, side="right") - 1
        j = idx - self.cum[s]
        return s, int(j)

    def __getitem__(self, idx):
        s, j = self._locate(idx)
        z = self.latents[s][j]       # (C,H,W) ndarray (memmapped)
        y = self.labels[s][j].item()
        i = self.ids[s][j].item()
        return {
            "image": torch.from_numpy(z),       # zero-copy CPU tensor
            "label": torch.tensor(y, dtype=torch.long),
            "id":    torch.tensor(i, dtype=torch.long),
        }
