import torch
import torch.nn.functional as F
from einops import pack, unpack


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def safe_matmul(x, y, batch_size):
    # Initialize an empty tensor for the result
    print(x.shape, y.shape)
    result = torch.empty(x.shape[0], y.shape[1], device=x.device, dtype=x.dtype)

    # Perform the matmul in chunks
    for i in range(0, x.shape[0], batch_size):
        end_i = min(i + batch_size, x.shape[0])
        result[i:end_i] = torch.matmul(x[i:end_i], y)
    return result


def compute_dist(x, y):
    # x: n, d
    # y: m, d

    y_t = y.t().type_as(x)

    # |x - y| ^ 2 = x * x ^ t + y * y ^ t - 2 * x * y ^ t
    # dist = (
    #     torch.sum(x**2, dim=-1, keepdim=True)
    #     + torch.sum(y_t**2, dim=0, keepdim=True)
    #     - 2 * safe_matmul(x, y_t, batch_size=2)
    # )

    # sum_up = torch.sum(x**2, dim=-1, keepdim=True) + torch.sum(
    #     y_t.transpose() ** 2, dim=0, keepdim=True
    # )

    dist = (
        torch.sum(x**2, dim=-1, keepdim=True)
        + torch.sum(y_t**2, dim=0, keepdim=True)
        - 2 * torch.matmul(x, y_t)
    )

    return dist


def round_ste(x):
    """Round with straight through gradients."""
    xhat = x.round()
    return x + (xhat - x).detach()


def entropy_loss(affinity, temperature, loss_type="softmax"):
    """
    Increase codebook usage by maximizing entropy

    affinity: 2D tensor of size Dim, n_classes
    """

    n_classes = affinity.shape[-1]

    affinity = torch.div(affinity, temperature)
    probs = F.softmax(affinity, dim=-1)
    log_probs = F.log_softmax(affinity + 1e-5, dim=-1)

    if loss_type == "softmax":
        target_probs = probs
    elif loss_type == "argmax":
        codes = torch.argmax(affinity, dim=-1)
        one_hots = F.one_hot(codes, n_classes).to(codes)
        one_hots = probs - (probs - one_hots).detach()
        target_probs = one_hots
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))

    avg_probs = torch.mean(target_probs, dim=0)
    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    sample_entropy = -torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    return sample_entropy - avg_entropy
