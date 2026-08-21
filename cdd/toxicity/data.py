"""Dataset utilities for toxicity surrogate training."""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import datasets


class JigsawToxicityDataset(Dataset):
    """Jigsaw Toxic Comment Classification dataset for toxicity prediction.

    Loads from HuggingFace datasets, consolidates multiple toxicity labels
    into a single binary target (toxic vs. non-toxic).
    """

    def __init__(self, split='train', max_length=128, text_model_name='bert-base-uncased'):
        all_data = datasets.load_dataset('thesofakillers/jigsaw-toxic-comment-classification-challenge', split='train')
        if split == 'train':
            self.dataset = all_data.select(range(int(len(all_data) * 0.9)))
        elif split == 'test':
            self.dataset = all_data.select(range(int(len(all_data) * 0.9), len(all_data)))
        else:
            self.dataset = all_data
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.max_length = max_length

        self.toxicity_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        comment_text = item['comment_text']

        is_toxic = any(item[col] == 1 for col in self.toxicity_columns)
        label = 1.0 if is_toxic else 0.0

        encoding = self.tokenizer(
            comment_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float),
            'text': comment_text,
        }
