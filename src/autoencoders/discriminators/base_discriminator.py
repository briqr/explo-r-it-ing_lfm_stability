from torch import nn
from torch.nn import functional as F
import torch
from torch.autograd import grad as torch_grad
from einops import rearrange, repeat
from torch import autocast
from autoencoders.discriminators.lpips import LPIPS


def hinge_discr_loss(fake_logits, real_logits):
    return (F.relu(1 - real_logits) + F.relu(1 + fake_logits)).mean()


def hinge_gen_loss(fake):
    return -fake.mean()


def gradient_penalty(images, output):
    batch_size = images.shape[0]

    gradients = torch_grad(
        outputs=output,
        inputs=images,
        grad_outputs=torch.ones(output.size(), device=images.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = rearrange(gradients, "b ... -> b (...)")
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def grad_layer_wrt_loss(loss: torch.Tensor, layer: nn.Parameter):
    with autocast(enabled=False, device_type=loss.device.type):
        return torch_grad(
            outputs=loss,
            inputs=layer,
            grad_outputs=torch.ones_like(loss),
            retain_graph=True,
        )[0].detach()


def get_perceptual_model(perceptual_model="vgg16"):
    
    if perceptual_model == "vgg16":
        return LPIPS().eval()
    else:
        return None


class BaseDiscriminator(nn.Module):
    disciminator_network: nn.Module = nn.Identity()

    default_params = {
        "loss_type": "hinge",
        "with_gradient_penalty": False,
        "lambda_perceptual_loss": 1.0,
        "lambda_disc_loss": 1.0,
        "lambda_adversarial_loss": 1.0,
        "lambda_grad_penalty": 10.0,
        "adaptive_weight_max": 1e3,
        "perceptual_model": "vgg16",
    }

    def __init__(self, **kwargs):
        super(BaseDiscriminator, self).__init__()
        for key, value in self.default_params.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.register_buffer("zero", torch.tensor(0.0), persistent=False)
        self.perceptual_model = get_perceptual_model(self.perceptual_model)

    def compute_disc_loss(self, real_logits, fake_logits):
        if self.loss_type == "hinge":
            return hinge_discr_loss(fake_logits, real_logits)
        else:
            raise NotImplementedError

    def comput_gen_loss(self, fake_logits):
        if self.loss_type == "hinge":
            return hinge_gen_loss(fake_logits)
        else:
            raise NotImplementedError

    def forward(self, func="get_disc_loss", **kwargs):
        # to avoid DDP error
        if func == "get_disc_loss":
            return self.get_disc_loss(**kwargs)
        elif func == "get_gan_loss":
            return self.get_gan_loss(**kwargs)

    def disc_forward(self, x):
        raise NotImplementedError

    def calculate_gradient_penalty(self, real, real_logits):
        if self.check_if_apply_gradient_penalty():
            gradient_penalty_loss = gradient_penalty(real, real_logits)
        else:
            gradient_penalty_loss = self.zero
        return gradient_penalty_loss

    def check_if_apply_gradient_penalty(
        self, lambda_grad_penalty=None, apply_gradient_penalty=False
    ):
        return (
            self.with_gradient_penalty
            and lambda_grad_penalty > 0
            and apply_gradient_penalty
        )

    def get_disc_loss(
        self,
        real,
        fake,
        lambda_disc_loss=None,
        lambda_grad_penalty=None,
        apply_gradient_penalty=False,
    ):
        lambda_disc_loss = lambda_disc_loss or self.lambda_disc_loss
        lambda_grad_penalty = lambda_grad_penalty or self.lambda_grad_penalty

        if self.check_if_apply_gradient_penalty(
            lambda_grad_penalty, apply_gradient_penalty
        ):
            real = real.requires_grad_()

        real_logits = self.disc_forward(real)
        fake_logits = self.disc_forward(fake.detach())
        discr_loss = self.compute_disc_loss(real_logits, fake_logits)
        gradient_penalty_loss = self.calculate_gradient_penalty(real, real_logits)

        loss_sum = (
            discr_loss * lambda_disc_loss + gradient_penalty_loss * lambda_grad_penalty
        )
        return loss_sum, {
            "discr_loss": discr_loss,
            "gradient_penalty_loss": gradient_penalty_loss,
            "lambda_disc_loss": lambda_disc_loss,
            "lambda_grad_penalty": lambda_grad_penalty,
            "disc_loss_sum": loss_sum,
        }

    def get_adaptive_weight(self, gen_loss, perceptual_loss=None, last_dec_layer=None):
        if perceptual_loss != None:
            norm_grad_wrt_perceptual_loss = grad_layer_wrt_loss(
                perceptual_loss, last_dec_layer
            ).norm(p=2)

            norm_grad_wrt_gen_loss = grad_layer_wrt_loss(gen_loss, last_dec_layer).norm(
                p=2
            )
            adaptive_weight = (
                norm_grad_wrt_perceptual_loss / norm_grad_wrt_gen_loss.clamp(min=1e-3)
            )
            adaptive_weight.clamp_(max=self.adaptive_weight_max)

            if torch.isnan(adaptive_weight).any():
                adaptive_weight = 1.0
        else:
            adaptive_weight = 1.0

        return adaptive_weight

    def get_perceptual_loss(self, real, fake):
        if self.perceptual_model is None:
            return None
        self.perceptual_model.eval()
        channels = fake.shape[1]
        if channels == 1:
            real = repeat(real, "b 1 h w -> b c h w", c=3)
            fake = repeat(fake, "b 1 h w -> b c h w", c=3)

        # real_features = self.perceptual_model(real)
        # fake_features = self.perceptual_model(fake)
        # perceptual_loss = 0
        # if isinstance(real_features, list):
        #     perceptual_loss = sum(
        #         F.mse_loss(real_feat, fake_feat)
        #         for real_feat, fake_feat in zip(real_features, fake_features)
        #     ) / len(real_features)
        # else:
        #     perceptual_loss = F.mse_loss(real_features, fake_features)
        # return perceptual_loss

        return self.perceptual_model(real, fake).mean()

    def get_gan_loss(
        self,
        real,
        fake,
        last_dec_layer=None,
        lambda_adversarial_loss=None,
        lambda_perceptual_loss=None,
    ):

        # notice that lambda_adversarial_loss could be 0.0
        if lambda_adversarial_loss is None:
            lambda_adversarial_loss = self.lambda_adversarial_loss

        lambda_perceptual_loss = lambda_perceptual_loss or self.lambda_perceptual_loss

        if lambda_perceptual_loss is None or lambda_perceptual_loss == 0.0:
            perceptual_loss = self.zero
        else:
            perceptual_loss = self.get_perceptual_loss(real, fake)

        if (
            lambda_adversarial_loss is None
            or lambda_perceptual_loss is None
            or lambda_adversarial_loss == 0.0
        ):
            gen_loss = self.zero
            adaptive_weight = 1.0
        else:
            nll_loss = torch.nn.functional.mse_loss(real, fake)
            fake_logits = self.disc_forward(fake)
            gen_loss = self.comput_gen_loss(fake_logits)
            adaptive_weight = self.get_adaptive_weight(
                gen_loss,
                perceptual_loss * lambda_perceptual_loss + nll_loss,
                last_dec_layer,
            )

        loss_sum = (
            perceptual_loss * lambda_perceptual_loss
            + gen_loss * adaptive_weight * lambda_adversarial_loss
        )

        return loss_sum, {
            "perceptual_loss": perceptual_loss.item(),
            "gen_loss": gen_loss.item(),
            "adaptive_weight": adaptive_weight,
            "lambda_perceptual_loss": lambda_perceptual_loss,
            "lambda_adversarial_loss": lambda_adversarial_loss,
            "gan_loss_sum": loss_sum.item(),
        }
