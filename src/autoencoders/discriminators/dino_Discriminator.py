# https://github.com/mosaicml/diffusion/blob/main/diffusion/models/autoencoder.py#L403
from .base_discriminator import BaseDiscriminator
from torch import nn, einsum
import torch
from torch.nn.functional import normalize
from autoencoders.layers.discriminator import ProjectedDiscriminator


class DinoDiscriminator(BaseDiscriminator):
    """Defines a PatchGAN discriminator.

    Based on code from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py

    Args:
        input_channels (int): Number of input channels. Default: `3`.
        num_filters (int): Number of filters in the first layer. Default: `64`.
        num_layers (int): Number of layers in the discriminator. Default: `3`.
    """

    def __init__(
        self,
        c_dim: int = 0,
        diffaug: bool = True,
        p_crop: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.ProjectedDiscriminator = ProjectedDiscriminator(
            c_dim=c_dim, diffaug=diffaug, p_crop=p_crop
        )

    def disc_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through the discriminator."""
        return self.ProjectedDiscriminator(x)
