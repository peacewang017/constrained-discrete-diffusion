"""Training script for SA score surrogate model."""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "discrete-diffusion-guidance"))

from cdd.sascore.model import GPT2SASurrogate
from cdd.sascore.data import QMD5Dataset
from cdd.sascore.utils import extract_embedding_from_udlm
from sascorer import calculateScore
from rdkit import Chem


def compute_sascore(smiles):
    """Compute SA score for a SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 10.0
        return calculateScore(mol)
    except Exception:
        return 10.0


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading model...")
    model = GPT2SASurrogate(
        gpt2_model_name=args.gpt2_model,
        udlm_embedding_path=args.udlm_model,
        freeze_backbone=args.freeze_backbone,
        dropout=args.dropout,
    ).to(device)

    print("Loading datasets...")
    train_dataset = QMD5Dataset(split='train')
    val_dataset = QMD5Dataset(split='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    output_dir = Path(args.output_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')

    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            smiles = batch['smiles']

            target_scores = torch.tensor([compute_sascore(s) for s in smiles], device=device)

            optimizer.zero_grad()

            embeddings = extract_embedding_from_udlm(args.udlm_model, device=device)[0]
            vocab_size = embeddings.shape[0]
            batch_size, seq_len = input_ids.shape

            flat_ids = input_ids.view(-1)
            input_embeddings = embeddings[flat_ids].view(batch_size, seq_len, -1).float()

            predictions = model(input_embeddings, attention_mask)

            loss = criterion(predictions, target_scores)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                smiles = batch['smiles']

                target_scores = torch.tensor([compute_sascore(s) for s in smiles], device=device)

                embeddings = extract_embedding_from_udlm(args.udlm_model, device=device)[0]
                vocab_size = embeddings.shape[0]
                batch_size, seq_len = input_ids.shape

                flat_ids = input_ids.view(-1)
                input_embeddings = embeddings[flat_ids].view(batch_size, seq_len, -1).float()

                predictions = model(input_embeddings, attention_mask)

                loss = criterion(predictions, target_scores)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, output_dir / 'best_model.pt')
            print(f"  -> Saved best model (val_loss = {val_loss:.4f})")

    print(f"Training complete. Best val_loss = {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pt'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpt2_model', type=str, default='gpt2')
    parser.add_argument('--udlm_model', type=str, default='kuleshov-group/udlm-qm9')
    parser.add_argument('--output_dir', type=str, default='/home/zyluo/code/constrained-dllm/cdd/sascore/checkpoints')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--freeze_backbone', action='store_true')
    args = parser.parse_args()

    train(args)