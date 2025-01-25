# Copyright (c) 2025 Senhorita Jaine
# Este código é parte do projeto exclusivo desenvolvido em colaboração com Lumin.
# Proibida a remoção ou alteração deste cabeçalho.
import torch
from torch import nn
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter


# Helper functions
def exists(val):
    """Check if a value exists (is not None)."""
    return val is not None


def default(val, d):
    """Return the value if it exists, otherwise return the default."""
    return val if exists(val) else d


class LoRA(Module):
    """
    LoRA (Low-Rank Adaptation) module for efficient fine-tuning of large models.

    Args:
        dim (int): Input dimensionality.
        dim_out (int): Output dimensionality.
        r (int, optional): Rank of the adaptation matrices. Defaults to 8.
        alpha (float, optional): Scaling factor. Defaults to the rank (r).
        init_method (str, optional): Method to initialize weights ('random', 'zeros', 'xavier', 'he').
        freeze_A (bool, optional): Freeze matrix A during training. Defaults to False.
        freeze_B (bool, optional): Freeze matrix B during training. Defaults to False.
        sparsity (float, optional): Apply sparsity to weights (0 to 1). Defaults to 0 (no sparsity).
        use_linear (bool, optional): Combine LoRA with a linear layer. Defaults to False.
        stochastic_noise (float, optional): Add Gaussian noise to weights during forward pass. Defaults to 0 (no noise).
    """

    def __init__(
        self,
        dim,
        dim_out,
        r=8,
        alpha=None,
        init_method="random",
        freeze_A=False,
        freeze_B=False,
        sparsity=0.0,
        use_linear=False,
        stochastic_noise=0.0
    ):
        super().__init__()
        assert r > 0, "Rank (r) must be a positive integer."
        assert dim > 0 and dim_out > 0, "Input and output dimensions must be positive integers."
        assert 0 <= sparsity <= 1, "Sparsity must be between 0 and 1."

        # Scaling factor
        alpha = default(alpha, r)
        self.scale = alpha / r
        self.stochastic_noise = stochastic_noise

        # Initialize low-rank matrices A and B
        self.A = nn.Parameter(torch.empty(dim, r))
        self.B = nn.Parameter(torch.empty(r, dim_out))
        self._initialize_weights(init_method)

        # Optionally freeze weights
        if freeze_A:
            self.A.requires_grad_(False)
        if freeze_B:
            self.B.requires_grad_(False)

        # Sparsity
        if sparsity > 0:
            self._apply_sparsity(sparsity)

        # Optional linear layer
        self.use_linear = use_linear
        if use_linear:
            self.linear = nn.Linear(dim, dim_out)

    def _initialize_weights(self, method):
        """Initialize weights A and B using the specified method."""
        if method == "random":
            nn.init.normal_(self.A, mean=0, std=0.02)
            nn.init.normal_(self.B, mean=0, std=0.02)
        elif method == "zeros":
            nn.init.zeros_(self.A)
            nn.init.zeros_(self.B)
        elif method == "xavier":
            nn.init.xavier_uniform_(self.A)
            nn.init.xavier_uniform_(self.B)
        elif method == "he":
            nn.init.kaiming_uniform_(self.A, a=0)
            nn.init.kaiming_uniform_(self.B, a=0)
        else:
            raise ValueError(f"Unknown initialization method: {method}")

    def _apply_sparsity(self, sparsity):
        """Apply sparsity to weights A and B."""
        mask_A = torch.bernoulli(torch.full_like(self.A, 1 - sparsity))
        mask_B = torch.bernoulli(torch.full_like(self.B, 1 - sparsity))
        self.A.data.mul_(mask_A)
        self.B.data.mul_(mask_B)

    @property
    def weight(self):
        """
        Compute the low-rank weight matrix.
        Optionally add stochastic noise during the forward pass.
        """
        weight = (self.A @ self.B) * self.scale
        if self.training and self.stochastic_noise > 0:
            noise = torch.randn_like(weight) * self.stochastic_noise
            weight += noise
        return weight

    def forward(self, x):
        """
        Forward pass through the LoRA layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, dim_out).
        """
        assert x.shape[-1] == self.A.shape[0], (
            f"Input tensor's last dimension ({x.shape[-1]}) must match "
            f"the input dimension of LoRA ({self.A.shape[0]})."
        )
        out = x @ self.weight
        if self.use_linear:
            out += self.linear(x)
        return out


# Example usage with TensorBoard and benchmarking
if __name__ == "__main__":
    # Parameters
    dim = 128
    dim_out = 64
    r = 8
    alpha = 16
    sparsity = 0.2
    stochastic_noise = 0.05

    # Initialize LoRA module
    lora = LoRA(
        dim, dim_out, r=r, alpha=alpha,
        init_method="xavier", sparsity=sparsity,
        stochastic_noise=stochastic_noise, use_linear=True
    )

    # Mock input
    x = torch.randn(32, dim)  # Batch of 32 with input dimension 128

    # TensorBoard setup
    writer = SummaryWriter()
    writer.add_histogram("A_weights", lora.A.data)
    writer.add_histogram("B_weights", lora.B.data)

    # Forward pass and benchmarking
    import time
    start = time.time()
    out = lora(x)
    end = time.time()

    print(f"Output shape: {out.shape}")  # Should be (32, 64)
    print(f"Inference time: {end - start:.6f} seconds")
    writer.close()
