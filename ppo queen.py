import torch
from torch import nn
from torch.nn.functional import pad
from collections import namedtuple, deque
from accelerate import Accelerator
from einops import rearrange, repeat
from torch.utils.data import Dataset, DataLoader
from functools import partial
from random import randrange
from torch.utils.tensorboard import SummaryWriter

# Helper Functions
def exists(val):
    """Check if a value exists."""
    return val is not None

def default(val, d):
    """Return val if it exists, otherwise return the default."""
    return val if exists(val) else d() if callable(d) else d

def shift(t, value=0, shift=1, dim=-1):
    """Shift a tensor along a specified dimension."""
    zeros = (0, 0) * (-dim - 1)
    return pad(t, (*zeros, shift, -shift), value=value)

def log(t, eps=1e-20):
    """Clamp tensor and compute log."""
    return torch.log(t.clamp(min=eps))

def log_prob(prob, indices):
    """Compute log probabilities given probabilities and indices."""
    assert prob.shape[:2] == indices.shape, f"Shapes must match: {prob.shape[:2]} and {indices.shape}"
    return log(prob.gather(-1, indices[..., None])).squeeze(-1)

# Data Utilities
Memory = namedtuple("Memory", ["sequence", "prompt_mask", "mask", "action_prob", "action_log_prob", "reward", "value"])

class ExperienceDataset(Dataset):
    """Dataset wrapper for RLHF training data."""
    def __init__(self, data, device=None):
        super().__init__()
        self.data = data
        self.device = device

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, ind):
        return tuple(map(lambda t: t[ind].to(self.device), self.data))

def create_dataloader(data, batch_size, shuffle=True, device=None, **kwargs):
    """Create a DataLoader for the experience dataset."""
    ds = ExperienceDataset(data, device=device)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, **kwargs)

# RLHF Trainer with PPO
class RLHFTrainer(nn.Module):
    def __init__(
        self,
        *,
        palm,
        reward_model,
        prompts,
        tokenizer,
        actor_lr=1e-4,
        critic_lr=1e-4,
        betas=(0.9, 0.999),
        eps_clip=0.2,
        value_clip=0.4,
        max_norm=None,
        kl_div_loss_weight=0.1,
        minibatch_size=16,
        epochs=1,
        log_dir="./logs",
        checkpoint_dir="./checkpoints",
        accelerate_kwargs={},
    ):
        super().__init__()
        self.accelerator = Accelerator(**accelerate_kwargs)

        # Tokenize prompts
        self.pad_value = 0
        self.prompt_token_ids = tokenizer(prompts)
        self.num_prompts = self.prompt_token_ids.shape[0]

        # Models
        self.palm = palm
        self.reward_model = reward_model.eval()

        self.actor_critic = ActorCritic(palm=palm).to(palm.device)

        # Optimizers
        self.actor_optim = torch.optim.AdamW(self.actor_critic.actor_parameters(), lr=actor_lr, betas=betas)
        self.critic_optim = torch.optim.AdamW(self.actor_critic.critic_parameters(), lr=critic_lr, betas=betas)

        # Hyperparameters
        self.eps_clip = eps_clip
        self.value_clip = value_clip
        self.kl_div_loss_weight = kl_div_loss_weight
        self.max_norm = max_norm
        self.minibatch_size = minibatch_size
        self.epochs = epochs

        # Logging
        self.writer = SummaryWriter(log_dir=log_dir)
        self.checkpoint_dir = checkpoint_dir

        # Prepare models and optimizers with accelerator
        (
            self.actor_critic,
            self.reward_model,
            self.actor_optim,
            self.critic_optim,
        ) = self.accelerator.prepare(
            self.actor_critic, self.reward_model, self.actor_optim, self.critic_optim
        )

    def save_checkpoint(self, step):
        """Save model checkpoint."""
        path = f"{self.checkpoint_dir}/checkpoint_step_{step}.pt"
        torch.save(self.actor_critic.state_dict(), path)

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        self.actor_critic.load_state_dict(torch.load(path))

    def log_metrics(self, step, **metrics):
        """Log metrics for visualization."""
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

    def generate(
        self,
        prompt,
        max_seq_len,
        temperature=1.0,
        num_samples=4,
        eos_token=None,
    ):
        """Generate sequences based on a given prompt."""
        assert prompt.ndim == 1, "Prompt must be 1-dimensional."
        prompt = repeat(prompt, "n -> b n", b=num_samples)

        actor_critic = self.accelerator.unwrap_model(self.actor_critic)
        actor_critic.eval()

        actions, sequences, mask, prompt_mask, action_logits, _ = actor_critic.generate(
            prompt, max_seq_len=max_seq_len, temperature=temperature, eos_token=eos_token
        )

        rewards = self.reward_model(sequences, prompt_mask=prompt_mask, mask=mask, sample=True)
        best_sequence_index = rewards.argmax(dim=-1)
        best_sequence = sequences[best_sequence_index]
        return rearrange(best_sequence, "1 ... -> ...")

    def learn(self, memories, step):
        """Train the model using stored experiences."""
        stacked_memories = list(map(partial(pad_sequence, batch_first=True), zip(*memories)))
        dataloader = create_dataloader(stacked_memories, self.minibatch_size, device=self.accelerator.device)

        self.actor_critic.train()
        for epoch in range(self.epochs):
            for sequences, prompt_masks, masks, old_action_probs, old_log_probs, rewards, old_values in dataloader:
                # Compute new logits and values
                action_logits, values = self.actor_critic(sequences, mask=masks)

                action_probs = action_logits.softmax(dim=-1)
                action_log_probs = log_prob(action_probs, sequences)

                # Calculate policy loss
                ratios = torch.exp(action_log_probs - old_log_probs)
                advantages = rewards - old_values
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Backpropagation
                self.accelerator.backward(policy_loss)
                self.actor_optim.step()
                self.actor_optim.zero_grad()

                # Log metrics
                self.log_metrics(step, policy_loss=policy_loss.item())

                # Update critic
                value_loss = F.mse_loss(values, rewards)
                self.accelerator.backward(value_loss)
                self.critic_optim.step()
                self.critic_optim.zero_grad()

                # Log critic loss
                self.log_metrics(step, critic_loss=value_loss.item())
