import math
import torch
from torch import einsum, nn
import torch.nn.functional as F
from einops import rearrange


def exists(val):
    """Check if a value exists (is not None)."""
    return val is not None


# Decorators

def eval_decorator(fn):
    """Decorator to temporarily set the model to evaluation mode."""
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out
    return inner


# Tensor Helpers

def log(t, eps=1e-20):
    """Clamp tensor and compute log."""
    return torch.log(t.clamp(min=eps))


def masked_mean(seq, mask=None, dim=1, keepdim=False):
    """
    Compute the mean of a sequence while ignoring masked elements.

    Args:
        seq (Tensor): Input tensor of shape (batch, seq_len, ...).
        mask (Tensor): Boolean mask of shape (batch, seq_len).
        dim (int): Dimension along which to compute the mean.
        keepdim (bool): Whether to retain reduced dimensions.

    Returns:
        Tensor: Masked mean tensor.
    """
    if not exists(mask):
        return seq.mean(dim=dim)

    if seq.ndim == 3:
        mask = rearrange(mask, 'b n -> b n 1')

    masked_seq = seq.masked_fill(~mask, 0.)
    numer = masked_seq.sum(dim=dim, keepdim=keepdim)
    denom = mask.sum(dim=dim, keepdim=keepdim)

    masked_mean = numer / denom.clamp(min=1e-3)
    masked_mean = masked_mean.masked_fill(denom == 0, 0.)
    return masked_mean


# Sampling Helpers

def gumbel_noise(t):
    """
    Generate Gumbel noise for a tensor.

    Args:
        t (Tensor): Input tensor.

    Returns:
        Tensor: Gumbel noise tensor with the same shape as the input.
    """
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1., dim=-1):
    """
    Sample from a distribution using Gumbel-Softmax.

    Args:
        t (Tensor): Logits tensor of shape (..., num_classes).
        temperature (float): Temperature for sampling.
        dim (int): Dimension to sample along.

    Returns:
        Tensor: Indices of sampled classes.
    """
    assert temperature > 0, "Temperature must be greater than zero."
    noise = gumbel_noise(t)
    return ((t / max(temperature, 1e-10)) + noise).argmax(dim=dim)


def top_p(logits, thres=0.9, mask=None):
    """
    Apply nucleus (top-p) filtering to logits.

    Args:
        logits (Tensor): Input logits of shape (batch, num_classes).
        thres (float): Cumulative probability threshold (0 < thres <= 1).
        mask (Tensor): Optional mask to apply before top-p filtering.

    Returns:
        Tensor: Logits after top-p filtering.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Mask logits that exceed the cumulative probability threshold
    sorted_indices_to_remove = cum_probs > thres
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    if exists(mask):
        sorted_indices_to_remove = sorted_indices_to_remove | (~mask)

    # Replace filtered logits with -inf
    sorted_logits[sorted_indices_to_remove] = float('-inf')

    # Scatter back to original ordering
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


def top_k(logits, thres=0.9, mask=None):
    """
    Apply top-k filtering to logits.

    Args:
        logits (Tensor): Input logits of shape (batch, num_classes).
        thres (float): Top-k threshold (0 < thres <= 1).
        mask (Tensor): Optional mask to apply before top-k filtering.

    Returns:
        Tensor: Logits after top-k filtering.
    """
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k, dim=-1)

    if exists(mask):
        mask_expanded = mask.gather(1, ind)
        val = val.masked_fill(~mask_expanded, float('-inf'))

    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


# Advanced Sampling Helpers

def top_k_top_p(logits, top_k_thresh=0.9, top_p_thresh=0.9, mask=None):
    """
    Apply both top-k and top-p filtering to logits.

    Args:
        logits (Tensor): Input logits of shape (batch, num_classes).
        top_k_thresh (float): Top-k threshold (0 < top_k_thresh <= 1).
        top_p_thresh (float): Top-p threshold (0 < top_p_thresh <= 1).
        mask (Tensor): Optional mask to apply before filtering.

    Returns:
        Tensor: Logits after top-k and top-p filtering.
    """
    logits = top_k(logits, thres=top_k_thresh, mask=mask)
    return top_p(logits, thres=top_p_thresh, mask=mask)
