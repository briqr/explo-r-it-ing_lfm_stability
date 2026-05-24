from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


@dataclass
class DistEnv:
    ddp: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed() -> DistEnv:
    ddp = int(__import__("os").environ.get("WORLD_SIZE", "1")) > 1

    if ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(__import__("os").environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return DistEnv(ddp=ddp, rank=rank, local_rank=local_rank, world_size=world_size, device=device)


def cleanup_distributed(env: DistEnv) -> None:
    if env.ddp:
        dist.destroy_process_group()


def barrier(env: DistEnv) -> None:
    if env.ddp:
        dist.barrier()


def all_reduce_mean(x: torch.Tensor, env: DistEnv) -> torch.Tensor:
    if env.ddp:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x = x / env.world_size
    return x


def to_cuda_long(x: Any, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    else:
        t = torch.as_tensor(x)
    return t.to(device=device, dtype=torch.long, non_blocking=True)


def broadcast_indices(indices: Any, env: DistEnv) -> torch.Tensor:
    """Broadcast a 1D index tensor from rank 0. In single GPU, just returns it.
        important when indices have a randomness component, so all GPUs see the same subset
    """
    if env.is_main:
        index_tensor = to_cuda_long(indices, env.device)
        length = torch.tensor([index_tensor.numel()], device=env.device, dtype=torch.long)
    else:
        index_tensor = torch.empty(0, device=env.device, dtype=torch.long)
        length = torch.empty(1, device=env.device, dtype=torch.long)

    if env.ddp:
        dist.broadcast(length, src=0)
        if not env.is_main:
            index_tensor = torch.empty(int(length.item()), device=env.device, dtype=torch.long)
        dist.broadcast(index_tensor, src=0)

    return index_tensor
