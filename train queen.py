# Salvando o arquivo atualizado como "dataset_de_estilos.py"
file_path = "/mnt/data/dataset_de_estilos.py"

code_content = """
import os
import gzip
import random
from pathlib import Path

import numpy as np
import torch
from lion_pytorch import Lion
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from palm_rlhf_pytorch import PaLM

# Constants
NUM_BATCHES = int(1e5)  # Número total de batches para treinamento
BATCH_SIZE = 4  # Tamanho do batch
GRADIENT_ACCUMULATE_EVERY = 4  # Gradiente acumulado para estabilidade
LEARNING_RATE = 1e-4  # Taxa de aprendizado
VALIDATE_EVERY = 100  # Frequência de validação
GENERATE_EVERY = 500  # Frequência de geração de texto
PRIME_LENGTH = 128  # Comprimento do texto inicial
GENERATE_LENGTH = 512  # Comprimento do texto gerado
SEQ_LEN = 1024  # Comprimento da sequência de entrada
CHECKPOINT_DIR = "./checkpoints"  # Diretório para checkpoints
LOG_DIR = "./logs"  # Diretório para logs

# Funções auxiliares
def cycle(loader):
    """Ciclo infinito de um DataLoader."""
    while True:
        for data in loader:
            yield data

def decode_token(token):
    """Decodifica um único token para um caractere."""
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    """Decodifica uma lista de tokens para uma string."""
    return "".join(list(map(decode_token, tokens)))

def save_checkpoint(model, optimizer, step, checkpoint_dir=CHECKPOINT_DIR):
    """Salva o estado do modelo e otimizador em um checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = Path(checkpoint_dir) / f"checkpoint_step_{step}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
        },
        checkpoint_path,
    )

def load_checkpoint(model, optimizer, checkpoint_dir=CHECKPOINT_DIR):
    """Carrega o estado do modelo e otimizador a partir de um checkpoint."""
    checkpoint_files = list(Path(checkpoint_dir).glob("*.pt"))
    if not checkpoint_files:
        print("Nenhum checkpoint encontrado.")
        return 0

    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Checkpoint carregado de {latest_checkpoint}")
    return checkpoint["step"]

# Inicializa o acelerador
accelerator = Accelerator(gradient_accumulation_steps=GRADIENT_ACCUMULATE_EVERY)
device = accelerator.device

# Inicializa o modelo
model = PaLM(
    num_tokens=256,
    dim=512,
    depth=8,
    flash_attn=True,
).to(device)

# Prepara os dados
def prepare_data(file_path):
    """Prepara os dados do arquivo enwik8."""
    print("🔄 Carregando os dados do arquivo enwik8...")
    with gzip.open(file_path) as file:
        data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
        np_train, np_valid = np.split(data, [int(90e6)])
        print("✅ Dados carregados e divididos em treino e validação!")
        return torch.from_numpy(np_train), torch.from_numpy(np_valid)

# Alteração para refletir domínio personalizado
DATA_SOURCE = "https://queen-ai-datasets.com/enwik8.gz"
data_train, data_val = prepare_data("./data/enwik8.gz")

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq.to(device)

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))
val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE))

# Inicializa o otimizador
optim = Lion(model.palm_parameters(), lr=LEARNING_RATE)

model, optim, train_loader, val_loader = accelerator.prepare(
    model, optim, train_loader, val_loader
)

# Inicializa o logger do TensorBoard
writer = SummaryWriter(log_dir=LOG_DIR)

# Loop de treinamento
start_step = load_checkpoint(model, optim)

for step in tqdm(range(start_step, NUM_BATCHES), mininterval=10.0, desc="Treinando"):
    model.train()

    with accelerator.accumulate(model):
        batch = next(train_loader)
        loss = model(batch, return_loss=True)
        accelerator.backward(loss / GRADIENT_ACCUMULATE_EVERY)

        accelerator.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()

        writer.add_scalar("Loss/Treinamento", loss.item(), step)
        accelerator.print(f"Passo {step}: Loss de Treinamento = {loss.item():.4f}")

    # Validação
    if step % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            batch = next(val_loader)
            val_loss = model(batch, return_loss=True)
            writer.add_scalar("Loss/Validação", val_loss.item(), step)
            accelerator.print(f"Passo {step}: Loss de Validação = {val_loss.item():.4f}")

    # Salva o checkpoint
    if step % VALIDATE_EVERY == 0:
        save_checkpoint(model, optim, step)

    # Geração de texto
    if step % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        prime = decode_tokens(inp)
        accelerator.print(f"Texto inicial: \n{prime}\n{'*' * 50}")

        if hasattr(model, "module"):
            sample = model.module.generate(GENERATE_LENGTH, inp[None, ...])
        else:
            sample = model.generate(GENERATE_LENGTH, inp[None, ...])

        output_str = decode_tokens(sample[0])
        accelerator.print(f"Texto Gerado:\n{output_str}\n{'=' * 50}")

        # Salva o texto gerado
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(f"{LOG_DIR}/generated_text_step_{step}.txt", "w") as f:
            f.write(f"Texto Inicial:\n{prime}\n\nTexto Gerado:\n{output_str}")

writer.close()
"""

with open(file_path, "w") as file:
    file.write(code_content)

file_path
