# Copyright (c) 2025 Senhorita Jaine
# Este código é parte do projeto exclusivo desenvolvido em colaboração com Lumin.
# Proibida a remoção ou alteração deste cabeçalho.
import torch
from palm_rlhf_pytorch import PaLM, RewardModel, RLHFTrainer
from accelerate import Accelerator
from pathlib import Path

# Setup Accelerator for distributed training
accelerator = Accelerator()
device = accelerator.device

# Initialize PaLM model
palm = PaLM(
    num_tokens=20000,  # Vocabulary size
    dim=512,           # Model dimension
    depth=12           # Number of transformer layers
).to(device)

# Initialize RewardModel
reward_model = RewardModel(
    palm=palm,
    num_binned_output=5  # Number of bins for classification
).to(device)

# Function to save checkpoints
def save_checkpoint(model, optimizer, path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)

# Function to load checkpoints
def load_checkpoint(model, optimizer, path):
    if Path(path).exists():
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Checkpoint loaded from {path}")
    else:
        print(f"No checkpoint found at {path}")

# Training RewardModel on mock data
accelerator.print("Training RewardModel...")
reward_model.train()
optimizer = torch.optim.AdamW(reward_model.parameters(), lr=1e-4)

# Mock data
batch_size = 8
seq_len = 1024
seq = torch.randint(0, 20000, (batch_size, seq_len)).to(device)  # Mock sequences
prompt_mask = torch.zeros(batch_size, seq_len).bool().to(device)  # Mock prompt mask
labels = torch.randint(0, 5, (batch_size,)).to(device)  # Mock labels

# Train RewardModel
epochs = 5
for epoch in range(epochs):
    optimizer.zero_grad()
    loss = reward_model(seq, prompt_mask=prompt_mask, labels=labels)
    accelerator.backward(loss)
    optimizer.step()
    accelerator.print(f"Epoch {epoch + 1}/{epochs}: Loss = {loss.item():.4f}")
    # Save checkpoint
    save_checkpoint(reward_model, optimizer, f"reward_model_epoch_{epoch + 1}.pt")

# Evaluate RewardModel
reward_model.eval()
reward = reward_model(seq, prompt_mask=prompt_mask)
accelerator.print(f"Reward: {reward}")

# Prepare prompts for RLHF training
prompts = torch.randint(0, 256, (4, 512)).to(device)  # 4 prompts of length 512

# Initialize RLHFTrainer
trainer = RLHFTrainer(
    palm=palm,
    reward_model=reward_model,
    prompt_token_ids=prompts
)

# RLHF training
accelerator.print("Training RLHF...")
trainer.train(
    num_episodes=10,           # Number of episodes
    max_timesteps=5,           # Maximum timesteps per episode
    update_timesteps=2,        # Timesteps before policy update
    max_batch_size=4,          # Batch size
    max_seq_len=1024,          # Maximum sequence length
    eos_token=None,            # End-of-sequence token
    temperature=0.7            # Sampling temperature
)

# Generate responses using the trained model
accelerator.print("Generating response...")
num_samples = 10
answers = trainer.generate(
    max_seq_len=1024,          # Maximum sequence length
    prompt=prompts[0],         # First prompt
    num_samples=num_samples    # Number of samples to generate
)

# Evaluate the generated responses with RewardModel
answers = answers.unsqueeze(0)  # Adjust shape for RewardModel
answer_rewards = reward_model(answers)

# Select the best answer
best_answer_idx = answer_rewards.argmax()
best_answer = answers[best_answer_idx]

# Output the results
accelerator.print(f"Best answer: {best_answer}")
accelerator.print(f"Reward scores: {answer_rewards.tolist()}")
