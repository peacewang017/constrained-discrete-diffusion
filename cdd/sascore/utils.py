"""Utilities for SA score model."""

import sys
from pathlib import Path
from typing import Tuple, Dict
import importlib.util

import torch

_discrete_diffusion_path = Path(__file__).parent.parent.parent / "discrete-diffusion-guidance"


def _install_flash_attn_mock():
    """Install a mock flash_attn module to bypass the import."""
    spec = importlib.util.find_spec('flash_attn')
    if spec is not None:
        return

    class MockFlashAttn:
        def __getattr__(self, name):
            return MockFlashAttn()
        def __call__(self, *args, **kwargs):
            return None

    sys.modules['flash_attn'] = MockFlashAttn()
    sys.modules['flash_attn.layers'] = MockFlashAttn()
    sys.modules['flash_attn.layers.rotary'] = MockFlashAttn()


def extract_embedding_from_udlm(
    model_name: str = "kuleshov-group/udlm-qm9",
    device: str = "cpu"
) -> Tuple[torch.Tensor, Dict[str, any]]:
    """Extract embedding matrix from UDLM-QM9 model.

    Directly loads from safetensors file, bypassing flash_attn dependency.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    _install_flash_attn_mock()

    model_path = hf_hub_download(model_name, "model.safetensors")
    state_dict = load_file(model_path)

    embedding_matrix = state_dict["backbone.vocab_embed.embedding"].detach().to(device)

    info = {
        "vocab_size": embedding_matrix.shape[0],
        "embedding_dim": embedding_matrix.shape[1],
        "model_name": model_name,
    }

    return embedding_matrix, info