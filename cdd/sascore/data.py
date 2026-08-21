"""Dataset utilities for SA score training."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "discrete-diffusion-guidance"))

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import datasets


class QMD5Dataset(Dataset):
    """QM9 dataset for SA score training."""

    def __init__(self, split='train', max_length=32):
        self.dataset = datasets.load_dataset('yairschiff/qm9', split=split)
        self.tokenizer = AutoTokenizer.from_pretrained('yairschiff/qm9-tokenizer', trust_remote_code=True)
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        smiles = item['canonical_smiles']

        encoding = self.tokenizer(
            smiles,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'smiles': smiles,
        }