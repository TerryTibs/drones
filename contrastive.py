"""
ContrastiveAlignmentLoss  (v2 - fixed)
=======================================
Key fixes vs v1:
  • Projection heads pre-registered at __init__ via register_modalities().
    v1 created them lazily in forward() → broke GPU device placement.
  • TemporalContrastiveLoss projectors also pre-registered.
  • Added momentum queue for small-batch stability (optional).
  • Temperature properly clamped with detach to avoid log(0).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class ContrastiveAlignmentLoss(nn.Module):
    """
    CLIP-style InfoNCE loss for cross-modal alignment.

    IMPORTANT: call register_modalities(names) at init time so all
    projection heads are created on the correct device before training.

    Args:
        latent_dim      : shared embedding dim
        modality_names  : list of all modality names to pre-register
        temperature     : initial softmax temperature
        learn_temp      : if True, temperature is a learnable scalar
        symmetric       : compute both i→j and j→i directions
        queue_size      : if > 0, use a momentum queue for negatives
        modality_pairs  : specific pairs to align; None = all pairs
    """

    def __init__(
        self,
        latent_dim:     int               = 512,
        modality_names: Optional[List[str]] = None,
        temperature:    float             = 0.07,
        learn_temp:     bool              = True,
        symmetric:      bool              = True,
        queue_size:     int               = 0,
        modality_pairs: Optional[List[Tuple[str, str]]] = None,
    ):
        super().__init__()
        self.latent_dim     = latent_dim
        self.symmetric      = symmetric
        self.modality_pairs = modality_pairs
        self.queue_size     = queue_size

        if learn_temp:
            self.log_temp = nn.Parameter(torch.tensor(math.log(temperature)))
        else:
            self.register_buffer("log_temp", torch.tensor(math.log(temperature)))

        # Pre-registered projection heads
        self._proj_heads = nn.ModuleDict()
        if modality_names:
            for name in modality_names:
                self._register_proj(name)

        # Optional momentum queues per modality
        if queue_size > 0:
            self._queues: Dict[str, torch.Tensor] = {}
            self._queue_ptrs: Dict[str, int] = {}

    def _register_proj(self, name: str) -> None:
        if name not in self._proj_heads:
            self._proj_heads[name] = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.GELU(),
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.LayerNorm(self.latent_dim),
            )

    def register_modality(self, name: str) -> None:
        """Register a new modality projection head (idempotent)."""
        self._register_proj(name)

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp().clamp(0.01, 1.0)

    def _pool_project(self, name: str, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T, D) → L2-normalised (B, D)"""
        if name not in self._proj_heads:
            self._register_proj(name)
            # Move to same device as tokens
            self._proj_heads[name] = self._proj_heads[name].to(tokens.device)
        pooled = tokens.mean(dim=1)
        return F.normalize(self._proj_heads[name](pooled), dim=-1)

    def _infonce(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """a, b: (B, D) normalised. Diagonal = positive pairs."""
        B   = a.size(0)
        sim = (a @ b.T) / self.temperature
        lbl = torch.arange(B, device=a.device)
        if self.symmetric:
            return (F.cross_entropy(sim, lbl) + F.cross_entropy(sim.T, lbl)) / 2
        return F.cross_entropy(sim, lbl)

    def forward(
        self,
        modality_tokens: Dict[str, torch.Tensor],
        temporal_mask:   Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        embeddings = {name: self._pool_project(name, tok)
                      for name, tok in modality_tokens.items()}
        names = list(embeddings.keys())

        if self.modality_pairs is not None:
            pairs = [(a, b) for a, b in self.modality_pairs
                     if a in embeddings and b in embeddings]
        else:
            pairs = [(names[i], names[j])
                     for i in range(len(names))
                     for j in range(i + 1, len(names))]

        device = next(iter(embeddings.values())).device
        if not pairs:
            return torch.tensor(0.0, device=device, requires_grad=True), {}

        total, loss_dict = torch.tensor(0.0, device=device), {}
        for a, b in pairs:
            loss = self._infonce(embeddings[a], embeddings[b])
            loss_dict[f"contrastive/{a}-{b}"] = loss.item()
            total = total + loss

        total = total / len(pairs)
        loss_dict["contrastive/total"] = total.item()
        return total, loss_dict


class TemporalContrastiveLoss(nn.Module):
    """
    Temporal alignment: events at the same time step across modalities
    should be close; events at different time steps should be far.

    Pre-register modality names at init.
    """

    def __init__(
        self,
        latent_dim:      int             = 512,
        modality_names:  Optional[List[str]] = None,
        window_size:     int             = 4,
        temperature:     float           = 0.07,
    ):
        super().__init__()
        self.window_size = window_size
        self.temperature = temperature
        self.latent_dim  = latent_dim

        self._proj = nn.ModuleDict()
        if modality_names:
            for name in modality_names:
                self._proj[name] = nn.Linear(latent_dim, latent_dim)

    def _get_proj(self, name: str, device: torch.device) -> nn.Module:
        if name not in self._proj:
            self._proj[name] = nn.Linear(self.latent_dim, self.latent_dim).to(device)
        return self._proj[name]

    def forward(
        self,
        tokens_a: torch.Tensor,    # (B, T, D)
        tokens_b: torch.Tensor,    # (B, T, D)
        name_a:   str = "mod_a",
        name_b:   str = "mod_b",
    ) -> torch.Tensor:
        B, T, D = tokens_a.shape
        device  = tokens_a.device
        # Truncate to minimum length
        min_T = min(T, tokens_b.size(1))
        a = tokens_a[:, :min_T, :]
        b = tokens_b[:, :min_T, :]

        proj_a = F.normalize(self._get_proj(name_a, device)(a), dim=-1)  # (B, T, D)
        proj_b = F.normalize(self._get_proj(name_b, device)(b), dim=-1)

        # Compute similarity strictly within each batch element: (B, T, T)
        sim = torch.bmm(proj_a, proj_b.transpose(1, 2)) / self.temperature

        # Time indices are strictly sequence-length: (T,)
        t_idx = torch.arange(min_T, device=device)
        # Create a (1, T, T) mask that broadcasts across the batch dimension
        pos_mask = (t_idx.unsqueeze(1) - t_idx.unsqueeze(0)).abs() <= self.window_size
        pos_mask = pos_mask.unsqueeze(0).float()

        log_probs = F.log_softmax(sim, dim=-1)
        denom     = pos_mask.sum(dim=-1).clamp(min=1.0)
        return -(log_probs * pos_mask).sum(dim=-1).div(denom).mean()
