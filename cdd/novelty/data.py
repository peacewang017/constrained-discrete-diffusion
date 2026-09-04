"""Dataset utilities for the novelty constraint."""

from typing import Set

import datasets


def load_seen_smiles(split: str = 'train', dataset_name: str = 'yairschiff/qm9') -> Set[str]:
    """Load the set of canonical SMILES strings from a QM9 dataset split."""
    ds = datasets.load_dataset(dataset_name, split=split)
    return set(ds['canonical_smiles'])
