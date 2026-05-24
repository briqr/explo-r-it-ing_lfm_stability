import os
import hashlib
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List
from train_consts import FFHQ_DATAROOT

def _stable_sorted_paths(data_dir: str, exts=(".png",)) -> List[str]:
    paths = []
    for root, dirs, files in os.walk(data_dir, topdown=True):
        dirs.sort()   # deterministic traversal
        files.sort()  # deterministic traversal
        for f in files:
            if f.lower().endswith(exts):
                paths.append(os.path.join(root, f))
    if not paths:
        raise FileNotFoundError(f"No images with {exts} under {data_dir}")
    return paths

def _hash_key(relpath: str, seed: int) -> bytes:
    return hashlib.sha1(f"{seed}::{relpath}".encode("utf-8")).digest()

class FFHQDataset(Dataset):
    """
    Deterministic FFHQ dataset with train/val split only (no test).
    Each sample has a stable integer `image_id` in [0, len(split)-1].
    """
    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 transform: Optional[transforms.Compose] = None,
                 size: int = 256,
                 seed: int = 42,
                 val_ratio: float = 0.1):
        super().__init__()
        self.data_dir = data_dir
        self.size = size
        self.split = split
        self.seed = int(seed)
        self.val_ratio = float(val_ratio)

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"FFHQ images directory not found: {self.data_dir}")

        all_paths = _stable_sorted_paths(self.data_dir, exts=(".png",))

        rel = [os.path.relpath(p, self.data_dir) for p in all_paths]
        keyed = sorted(zip(rel, all_paths), key=lambda rp: _hash_key(rp[0], self.seed))
        ordered_paths = [p for _, p in keyed]

        n = len(ordered_paths)
        n_val = int(n * self.val_ratio)
        n_train = n - n_val
        if self.split == 'train':
            self.image_paths = ordered_paths[:n_train]
        elif self.split == 'val':
            self.image_paths = ordered_paths[n_train:]
        else:
            raise ValueError(f"Invalid split '{self.split}'. Use 'train' or 'val'.")

        self.id_map = {p: i for i, p in enumerate(self.image_paths)}

        # Default transform
        self.transform = transform or transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        print(f"Loaded {len(self.image_paths)} images for split='{self.split}' "
              f"(seed={self.seed}, val_ratio={self.val_ratio})")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image) if self.transform else image
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            image = torch.zeros((3, self.size, self.size))

        return {
            'image': image,
            'label': 0,
            'id': self.id_map[image_path],  # stable int in [0, len(split)-1]
        }

