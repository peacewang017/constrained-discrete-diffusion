"""Constraint functions for CDD."""

import torch
from typing import Callable, Optional
from pathlib import Path

UDLM_MODEL_NAME = "kuleshov-group/udlm-qm9"
SASCORE_DEFAULT_CKPT = Path(__file__).parent / "sascore" / "checkpoints"

TEXT_MODEL_NAME = "kuleshov-group/udlm-lm1b"
TOXICITY_DEFAULT_CKPT = Path(__file__).parent / "toxicity" / "checkpoints"

_sascore_model_cache: dict = {}
_toxicity_model_cache: dict = {}
_gumbel_embedder_cache: dict = {}
_gumbel_embedder_text_cache: dict = {}


class GumbelEmbedder:
    """Converts soft token distributions to embeddings via UDLM embedding matrix."""

    def __init__(self, udlm_model_name: str = UDLM_MODEL_NAME, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from sascore.utils import extract_embedding_from_udlm
        emb_matrix, info = extract_embedding_from_udlm(udlm_model_name, device=self.device)
        self.vocab_size = info["vocab_size"]
        self.embedding_dim = info["embedding_dim"]
        self.embedding_matrix = emb_matrix

    def soft_to_embedding(self, soft_probs: torch.Tensor) -> torch.Tensor:
        """Convert soft token distribution to embedding sequence."""
        batch_size, seq_len, vocab_size = soft_probs.shape
        flat = soft_probs.view(-1, vocab_size)
        emb_matrix = self.embedding_matrix.to(device=flat.device, dtype=flat.dtype)
        embedded = flat @ emb_matrix
        return embedded.view(batch_size, seq_len, self.embedding_dim)


class SASCOREWrapper:
    """Wraps trained sascore model for SA score prediction."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        freeze_backbone: bool = False,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from cdd.sascore.model import GPT2SASurrogate
        self.model = GPT2SASurrogate(
            gpt2_model_name="gpt2",
            udlm_embedding_path=UDLM_MODEL_NAME,
            freeze_backbone=freeze_backbone,
            dropout=0.1,
        ).to(self.device)

        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])

        self.model.eval()

    def __call__(self, embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict SA score for a batch of molecular embeddings."""
        if attention_mask is None:
            attention_mask = torch.ones(embeddings.shape[:2], device=self.device, dtype=torch.long)
        return self.model(embeddings, attention_mask)


def _get_latest_sascore_ckpt() -> Optional[str]:
    """Find the most recent sascore checkpoint."""
    if not SASCORE_DEFAULT_CKPT.exists():
        return None
    runs = sorted(SASCORE_DEFAULT_CKPT.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for run in runs:
        best = run / "best_model.pt"
        if best.exists():
            return str(best)
    return None


def delta_g_qm9_sa(
    y_soft: torch.Tensor,
    sascore_model: Optional[SASCOREWrapper] = None,
    embedder: Optional[GumbelEmbedder] = None,
    τ: float = 4.0,
) -> torch.Tensor:
    """SA constraint violation: max(0, sascore(y) - τ)."""
    if sascore_model is None:
        ckpt = _get_latest_sascore_ckpt()
        if ckpt is None:
            raise RuntimeError("No sascore checkpoint found. Train sascore first.")
        if ckpt not in _sascore_model_cache:
            _sascore_model_cache[ckpt] = SASCOREWrapper(checkpoint_path=ckpt)
        sascore_model = _sascore_model_cache[ckpt]

    if embedder is None:
        if UDLM_MODEL_NAME not in _gumbel_embedder_cache:
            _gumbel_embedder_cache[UDLM_MODEL_NAME] = GumbelEmbedder()
        embedder = _gumbel_embedder_cache[UDLM_MODEL_NAME]

    embeddings = embedder.soft_to_embedding(y_soft).float()
    sa_scores = sascore_model(embeddings)
    return torch.clamp(sa_scores - τ, min=0)


def make_delta_g_fn(
    sascore_ckpt: Optional[str] = None,
    τ: float = 4.0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Factory to create a delta_g function for ALM projection (SA constraint)."""
    sascore_model = None
    if sascore_ckpt:
        sascore_model = SASCOREWrapper(checkpoint_path=sascore_ckpt)
    embedder = GumbelEmbedder()

    def delta_g_fn(y_soft):
        return delta_g_qm9_sa(y_soft, sascore_model=sascore_model, embedder=embedder, τ=τ)

    return delta_g_fn


class GumbelEmbedderText:
    """Converts soft token distributions to embeddings via text model embedding matrix."""

    def __init__(self, text_model_name: str = TEXT_MODEL_NAME, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from cdd.toxicity.utils import extract_embedding_from_text_model
        emb_matrix, info = extract_embedding_from_text_model(text_model_name, device=self.device)
        self.vocab_size = info["vocab_size"]
        self.embedding_dim = info["embedding_dim"]
        self.embedding_matrix = emb_matrix

    def soft_to_embedding(self, soft_probs: torch.Tensor) -> torch.Tensor:
        """Convert soft token distribution to embedding sequence."""
        batch_size, seq_len, vocab_size = soft_probs.shape
        flat = soft_probs.view(-1, vocab_size)
        emb_matrix = self.embedding_matrix.to(device=flat.device, dtype=flat.dtype)
        embedded = flat @ emb_matrix
        return embedded.view(batch_size, seq_len, self.embedding_dim)


class ToxicityWrapper:
    """Wraps trained toxicity surrogate model for toxicity score prediction."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        backbone_name: str = "EleutherAI/gpt-neo-1.3B",
        embedding_dim: int = 768,
        freeze_backbone: bool = False,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from cdd.toxicity.model import ToxicitySurrogate

        # Infer backbone from checkpoint config if available.
        if checkpoint_path:
            ckpt_path = Path(checkpoint_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Toxicity checkpoint not found: {checkpoint_path}. "
                    "Train the toxicity surrogate first or provide a valid path."
                )
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            config = ckpt.get("config", {})
            # Support both old (gpt2_model) and new (backbone_model) config keys.
            if "backbone_model" in config:
                backbone_name = config["backbone_model"]
            elif "gpt2_model" in config:
                backbone_name = config["gpt2_model"]
            print(f"  Loading toxicity surrogate from {checkpoint_path}")
            print(f"  Backbone from checkpoint: {backbone_name}")

        self.model = ToxicitySurrogate(
            backbone_name=backbone_name,
            embedding_dim=embedding_dim,
            freeze_backbone=freeze_backbone,
            dropout=0.1,
        ).to(self.device)

        if checkpoint_path:
            state_dict = ckpt["model_state_dict"]
            # Map old GPT2ToxicitySurrogate keys (gpt2.*) to new ToxicitySurrogate keys (backbone.*).
            mapped_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("gpt2."):
                    k = "backbone." + k[5:]
                mapped_state_dict[k] = v
            missing, unexpected = self.model.load_state_dict(mapped_state_dict, strict=False)
            if missing:
                print(f"  Warning: missing keys in checkpoint: {missing}")
            if unexpected:
                print(f"  Warning: unexpected keys in checkpoint: {unexpected}")

        for param in self.model.backbone.parameters():
            param.requires_grad = False

        self.model.eval()

    def __call__(self, embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict toxicity score for a batch of text embeddings."""
        if attention_mask is None:
            attention_mask = torch.ones(embeddings.shape[:2], device=self.device, dtype=torch.long)
        logits = self.model(embeddings, attention_mask)
        return torch.sigmoid(logits)


def _get_latest_toxicity_ckpt() -> Optional[str]:
    """Find the most recent toxicity surrogate checkpoint."""
    if not TOXICITY_DEFAULT_CKPT.exists():
        return None
    runs = sorted(TOXICITY_DEFAULT_CKPT.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for run in runs:
        best = run / "best_model.pt"
        if best.exists():
            return str(best)
    return None


def delta_g_toxicity(
    y_soft: torch.Tensor,
    toxicity_model: Optional[ToxicityWrapper] = None,
    embedder: Optional[GumbelEmbedderText] = None,
    τ: float = 0.5,
) -> torch.Tensor:
    """Toxicity constraint violation: max(0, toxicity_score - τ)."""
    if toxicity_model is None:
        ckpt = _get_latest_toxicity_ckpt()
        if ckpt is None:
            raise RuntimeError("No toxicity checkpoint found. Train toxicity surrogate first.")
        if ckpt not in _toxicity_model_cache:
            _toxicity_model_cache[ckpt] = ToxicityWrapper(checkpoint_path=ckpt)
        toxicity_model = _toxicity_model_cache[ckpt]

    if embedder is None:
        if TEXT_MODEL_NAME not in _gumbel_embedder_text_cache:
            _gumbel_embedder_text_cache[TEXT_MODEL_NAME] = GumbelEmbedderText()
        embedder = _gumbel_embedder_text_cache[TEXT_MODEL_NAME]

    embeddings = embedder.soft_to_embedding(y_soft).float()
    toxicity_scores = toxicity_model(embeddings)
    return torch.clamp(toxicity_scores - τ, min=0)


def make_delta_g_fn_toxicity(
    toxicity_ckpt: Optional[str] = None,
    τ: float = 0.5,
    text_model_name: str = TEXT_MODEL_NAME,
    backbone_name: str = "EleutherAI/gpt-neo-1.3B",
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Factory to create a delta_g function for ALM projection (toxicity constraint).

    Args:
        toxicity_ckpt: path to toxicity surrogate checkpoint
        τ: toxicity threshold
        text_model_name: HF model name for embedding extraction (e.g. udlm-lm1b or mdlm-owt)
        backbone_name: backbone for toxicity surrogate (overridden by checkpoint config if available)
    """
    embedder = GumbelEmbedderText(text_model_name=text_model_name)
    toxicity_model = None
    if toxicity_ckpt:
        toxicity_model = ToxicityWrapper(
            checkpoint_path=toxicity_ckpt,
            backbone_name=backbone_name,
            embedding_dim=embedder.embedding_dim,
        )

    def delta_g_fn(y_soft):
        return delta_g_toxicity(y_soft, toxicity_model=toxicity_model, embedder=embedder, τ=τ)

    return delta_g_fn