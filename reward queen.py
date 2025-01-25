import torch
from torch import nn
from torch.nn.functional import mse_loss, cross_entropy
from einops import rearrange, repeat
from pathlib import Path
from palm_rlhf_pytorch.utils import masked_mean, gumbel_sample
from palm_rlhf_pytorch.palm import PaLM
from torch.utils.tensorboard import SummaryWriter

# Helper functions
def exists(val):
    """Check if a value exists."""
    return val is not None


class RewardModel(nn.Module):
    """
    Reward Model built on top of PaLM with support for regression or classification tasks.

    Args:
        palm (PaLM): Pretrained PaLM model.
        dropout (float): Dropout probability for regularization.
        num_binned_output (int): Number of bins for classification tasks. Use 0 for regression.
        use_lora (bool): Whether to use LoRA fine-tuning.
        lora_r (int): Rank for LoRA.
        reward_lora_scope (str): Scope for LoRA fine-tuning.
        log_dir (str): Directory for TensorBoard logs.
    """
    def __init__(
        self,
        palm: PaLM,
        dropout=0.1,
        num_binned_output=0,
        use_lora=True,
        lora_r=8,
        reward_lora_scope="reward",
        log_dir="./logs",
    ):
        super().__init__()

        # Deep copy the PaLM model and set dropout
        self.palm = torch.jit.script(copy.deepcopy(palm))  # Use TorchScript for potential optimization
        self.palm.set_dropout(dropout)

        self.reward_lora_scope = reward_lora_scope if use_lora else None

        if exists(self.reward_lora_scope):
            self.palm.add_finetune_params(reward_lora_scope, lora_r=lora_r)

        dim = palm.dim
        self.binned_output = num_binned_output > 1

        # Embedding parameters for prompt and response
        self.prompt_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.response_embed = nn.Parameter(torch.zeros(1, 1, dim))

        # Output head: Scalar (regression) or Multi-class (classification)
        if self.binned_output:
            self.to_pred = nn.Linear(dim, num_binned_output)
        else:
            self.to_pred = nn.Sequential(
                nn.Linear(dim, 1, bias=False),
                Rearrange("... 1 -> ..."),
            )

        # Initialize weights
        self._initialize_weights()

        # TensorBoard writer for logging
        self.writer = SummaryWriter(log_dir=log_dir)

    def _initialize_weights(self):
        """Initialize weights for the output layers and embeddings."""
        nn.init.zeros_(self.prompt_embed)
        nn.init.zeros_(self.response_embed)
        if isinstance(self.to_pred, nn.Sequential):
            nn.init.kaiming_uniform_(self.to_pred[0].weight, nonlinearity="relu")
        elif isinstance(self.to_pred, nn.Linear):
            nn.init.kaiming_uniform_(self.to_pred.weight, nonlinearity="relu")

    def save(self, path):
        """Save model weights to a file."""
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load model weights from a file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        self.load_state_dict(torch.load(str(path)))

    def finetune_parameters(self):
        """Return parameters for fine-tuning."""
        return [
            *self.to_pred.parameters(),
            *(
                self.palm.finetune_parameters(self.reward_lora_scope)
                if exists(self.reward_lora_scope)
                else self.palm.parameters()
            ),
        ]

    def log_metrics(self, step, **metrics):
        """Log metrics for TensorBoard."""
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

    def forward(
        self,
        x,
        mask=None,
        prompt_mask=None,
        prompt_lengths=None,
        labels=None,
        sample=False,
        sample_temperature=1.0,
        disable_lora=False,
        debug=False,
    ):
        """
        Forward pass for the Reward Model.

        Args:
            x (Tensor): Input tensor of token IDs.
            mask (Tensor): Attention mask for the input.
            prompt_mask (Tensor): Mask for separating prompt and response tokens.
            prompt_lengths (Tensor): Lengths of the prompts in the input.
            labels (Tensor): Ground truth labels for training.
            sample (bool): Whether to sample predictions (useful for binned output).
            sample_temperature (float): Temperature for sampling.
            disable_lora (bool): Disable LoRA fine-tuning during inference.
            debug (bool): Enable debug mode to log intermediate outputs.

        Returns:
            Tensor: Predicted reward values or loss if labels are provided.
        """
        assert not (exists(prompt_mask) and exists(prompt_lengths)), "Provide either prompt_mask or prompt_lengths, not both."

        # Derive prompt mask from prompt lengths
        if exists(prompt_lengths):
            batch, seq_len = x.shape
            arange = torch.arange(seq_len, device=x.device)
            prompt_mask = repeat(arange, "n -> b n", b=batch) < rearrange(prompt_lengths, "b -> b 1")

        # Add embeddings for prompt and response
        extra_embed = None
        if exists(prompt_mask):
            extra_embed = torch.where(
                rearrange(prompt_mask, "b n -> b n 1"),
                self.prompt_embed,
                self.response_embed,
            )

        # Pass through PaLM to get embeddings
        embeds = self.palm(
            x,
            extra_embed=extra_embed,
            return_only_embedding=True,
            disable_lora=disable_lora,
            finetune_scope=self.reward_lora_scope,
        )

        # Pool embeddings and predict rewards
        pooled = masked_mean(embeds, mask, dim=1)
        pred = self.to_pred(pooled)

        if debug:
            print(f"Embeddings shape: {embeds.shape}")
            print(f"Pooled shape: {pooled.shape}")
            print(f"Prediction shape: {pred.shape}")

        # Log embeddings mean and pooled outputs
        self.log_metrics(step=0, pooled_mean=pooled.mean().item(), pooled_std=pooled.std().item())

        # Sampling for binned output
        if sample and self.binned_output:
            assert not exists(labels), "Sampling is only for inference."
            pred = gumbel_sample(pred, temperature=sample_temperature, dim=-1)

        # Return predictions during inference
        if not exists(labels):
            return pred

        # Compute loss during training
        if not self.binned_output:
            return mse_loss(pred, labels)
        return cross_entropy(pred, labels)
