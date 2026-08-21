"""Training script for toxicity surrogate model."""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "discrete-diffusion-guidance"))

from cdd.toxicity.model import ToxicitySurrogate
from cdd.toxicity.data import JigsawToxicityDataset
from cdd.toxicity.utils import extract_embedding_from_text_model


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or f"toxicity_surrogate_{args.text_model.split('/')[-1]}",
        config=vars(args),
    )

    print("Loading embedding matrix...")
    emb_matrix, emb_info = extract_embedding_from_text_model(args.text_model, device=device)
    embedding_dim = emb_info['embedding_dim']
    vocab_size = emb_info['vocab_size']
    print(f"  Vocab size: {vocab_size}, Embedding dim: {embedding_dim}")
    emb_matrix = emb_matrix.float()

    print("Loading model...")
    model = ToxicitySurrogate(
        backbone_name=args.backbone_model,
        embedding_dim=embedding_dim,
        freeze_backbone=args.freeze_backbone,
        dropout=args.dropout,
    ).to(device)
    print(f"  Backbone: {args.backbone_model}, hidden size: {model.backbone.config.hidden_size}")

    print("Loading datasets...")
    tokenizer_name = args.tokenizer_name if args.tokenizer_name else args.text_model
    train_dataset = JigsawToxicityDataset(split='train', max_length=args.max_length, text_model_name=tokenizer_name)
    val_dataset = JigsawToxicityDataset(split='test', max_length=args.max_length, text_model_name=tokenizer_name)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    train_toxic_ratio = sum(train_dataset[i]['label'].item() for i in range(min(1000, len(train_dataset)))) / min(1000, len(train_dataset))
    print(f"Toxic ratio in train: {train_toxic_ratio:.3f}")
    wandb.log({'train_toxic_ratio': train_toxic_ratio})

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    output_dir = Path(args.output_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')

    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        step = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            batch_size, seq_len = input_ids.shape
            flat_ids = input_ids.view(-1)
            input_embeddings = emb_matrix[flat_ids].view(batch_size, seq_len, -1).float()

            optimizer.zero_grad()
            logits = model(input_embeddings, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            step += 1

            if step % args.log_interval == 0:
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/epoch': epoch + step / len(train_loader),
                })

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                batch_size, seq_len = input_ids.shape
                flat_ids = input_ids.view(-1)
                input_embeddings = emb_matrix[flat_ids].view(batch_size, seq_len, -1).float()

                logits = model(input_embeddings, attention_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        accuracy = correct / total

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {accuracy:.4f}")

        wandb.log({
            'epoch': epoch + 1,
            'train/loss': train_loss,
            'val/loss': val_loss,
            'val/accuracy': accuracy,
            'train/lr': optimizer.param_groups[0]['lr'],
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = output_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': accuracy,
                'config': vars(args),
            }, ckpt_path)
            wandb.log({'val/best_loss': best_val_loss, 'val/best_epoch': epoch + 1})
            wandb.save(str(ckpt_path))
            print(f"  -> Saved best model (val_loss = {val_loss:.4f})")

    print(f"Training complete. Best val_loss = {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pt'}")
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone_model', type=str, default='EleutherAI/gpt-neo-1.3B')
    parser.add_argument('--text_model', type=str, default='kuleshov-group/mdlm-owt')
    parser.add_argument('--tokenizer_name', type=str, default='gpt2')
    parser.add_argument('--output_dir', type=str, default='/home/zyluo/code/constrained-dllm/cdd/toxicity/checkpoints')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='cdd-toxicity')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--log_interval', type=int, default=50)
    args = parser.parse_args()

    train(args)
