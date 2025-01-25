from __future__ import annotations
from copy import deepcopy
import torch
from torch import nn
from torch.nn.functional import logsigmoid
from einops import rearrange


# Utility functions
def exists(value):
    """Check if a value exists."""
    return value is not None


def get_logprob_at(logits, seq):
    """
    Get log probabilities for specific sequences.
    
    Args:
        logits: Tensor of logits from the model.
        seq: Sequence tensor to compute probabilities for.
    
    Returns:
        Log probabilities for the sequence.
    """
    log_probs = logits.log_softmax(dim=-1)
    seq = rearrange(seq, '... -> ... 1')
    log_prob = log_probs.gather(-1, seq)
    return rearrange(log_prob, '... 1 -> ...')


class ImplicitPRM(nn.Module):
    """
    Implicit Process Reward Model (PRM)
    Implements process-level rewards as described in Yuan et al.'s paper.
    
    Args:
        model: Main model to train.
        ref_model: Reference model for comparative rewards. Defaults to a copy of the main model.
        beta: Scaling factor for the implicit rewards.
    """

    def __init__(self, model: nn.Module, ref_model: nn.Module | None = None, beta: float = 0.1):
        super().__init__()
        self.model = model

        # Use a reference model or create a copy of the main model
        if not exists(ref_model):
            ref_model = deepcopy(model)

        self.ref_model = ref_model
        self.ref_model.requires_grad_(False)  # Disable gradient updates for the reference model

        self.beta = beta

    def parameters(self):
        """Ensure only the main model's parameters are trainable."""
        return self.model.parameters()

    def forward(self, seq, labels=None):
        """
        Compute implicit rewards or the loss for training.

        Args:
            seq: Input sequence (batch_size, seq_len).
            labels: Optional labels for training. Defaults to None.

        Returns:
            Loss during training or implicit rewards otherwise.
        """
        source_seq, target_seq = seq[:, :-1], seq[:, 1:]

        # Mask to ignore padding tokens (assume tokens < 0 are padding)
        mask = target_seq >= 0

        # Compute logits for both main and reference models
        model_logits = self.model(source_seq)
        ref_model_logits = self.ref_model(source_seq)

        # Calculate log probabilities for each step
        log_prob = get_logprob_at(model_logits, target_seq)
        ref_log_prob = get_logprob_at(ref_model_logits, target_seq)

        # Compute implicit rewards using the scaling factor beta
        implicit_rewards = self.beta * (log_prob - ref_log_prob)

        # Zero out rewards in padding positions
        implicit_rewards = implicit_rewards.masked_fill(~mask, 0.0)

        # If labels are not provided, return the implicit rewards directly
        if not exists(labels):
            return implicit_rewards

        # Compute the loss using cross-entropy formulation (Eq. 5 from the paper)
        labels = rearrange(labels, 'b -> b 1')  # Ensure labels have the correct shape

        loss = (
            labels * logsigmoid(implicit_rewards) +
            (1.0 - labels) * logsigmoid(-implicit_rewards)
        )

        # Return the average loss, ignoring padding
        return loss[mask].mean()


# Example usage
if __name__ == '__main__':
    from palm_rlhf_pytorch import PaLM

    # Define a main and reference PaLM model
    palm = PaLM(
        num_tokens=256,
        dim=64,
        depth=2
    )
    ref_palm = PaLM(
        num_tokens=256,
        dim=64,
        depth=2
    )

    # Initialize the ImplicitPRM
    implicit_prm = ImplicitPRM(
        model=palm,
        ref_model=ref_palm,
        beta=0.1
    )

    # Generate mock data
    seq = torch.randint(0, 256, (2, 1024))
    labels = torch.randint(0, 2, (2,))

    # Compute the loss during training
    loss = implicit_prm(seq, labels)
    loss.backward()

    # After training, compute implicit rewards
    implicit_rewards = implicit_prm(seq)  # Output: Tensor[2, 1024]

    # Print results
    print(f"Training loss: {loss.item():.4f}")
    print(f"Implicit rewards shape: {implicit_rewards.shape}")
