"""
AudioMoE  (v2 - fixed)
=======================
Key fixes vs v1:
  • _init_weights applied throughout.
  • PitchSubExpert sin activation kept but scaled properly.
  • RhythmSubExpert depthwise conv initialised.
  • TimbreSubExpert global speaker vector broadcast is learnable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from ..core.expert_base import ExpertBase, SubExpert, _init_weights
from ..routing.local_router import LocalRouter


class PitchSubExpert(SubExpert):
    def __init__(self, latent_dim: int = 512):
        super().__init__("pitch", latent_dim)
        self.pitch_proj = nn.Linear(latent_dim, latent_dim)
        _init_weights(self)

    def forward(self, x, world_readout=None):
        x = super().forward(x, world_readout)
        return x + torch.sin(self.pitch_proj(x)) * 0.05


class RhythmSubExpert(SubExpert):
    def __init__(self, latent_dim: int = 512):
        super().__init__("rhythm", latent_dim)
        self.onset_conv = nn.Conv1d(latent_dim, latent_dim, kernel_size=3,
                                    padding=1, groups=latent_dim)
        # Kaiming normal is correct for depthwise conv (not Xavier which
        # had a fan_in/fan_out swap due to the reshape chain).
        nn.init.kaiming_normal_(self.onset_conv.weight, nonlinearity="relu")
        if self.onset_conv.bias is not None:
            nn.init.zeros_(self.onset_conv.bias)

    def forward(self, x, world_readout=None):
        x = super().forward(x, world_readout)
        onset = self.onset_conv(x.transpose(1, 2)).transpose(1, 2)
        return x + onset * 0.1


class TimbreSubExpert(SubExpert):
    def __init__(self, latent_dim: int = 512):
        super().__init__("timbre", latent_dim)
        self.spk_proj = nn.Linear(latent_dim, latent_dim)
        self.spk_norm = nn.LayerNorm(latent_dim)
        _init_weights(self)

    def forward(self, x, world_readout=None):
        x = super().forward(x, world_readout)
        spk = self.spk_norm(self.spk_proj(x.mean(dim=1, keepdim=True)))
        return x + spk


class PhonemeSubExpert(SubExpert):
    """Bridges audio and language — maps audio frames toward phoneme space."""
    def __init__(self, latent_dim: int = 512):
        super().__init__("phoneme", latent_dim)
        self.phoneme_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
        )
        _init_weights(self)

    def forward(self, x, world_readout=None):
        x = super().forward(x, world_readout)
        return self.phoneme_proj(x)


class EnvironmentSubExpert(SubExpert):
    def __init__(self, latent_dim: int = 512):
        super().__init__("environment", latent_dim)
        self.scene_proj = nn.Linear(latent_dim, latent_dim)
        _init_weights(self)

    def forward(self, x, world_readout=None):
        x = super().forward(x, world_readout)
        env = self.scene_proj(x.mean(dim=1, keepdim=True))
        return x + env * 0.1


class WaveformEncoder(nn.Module):
    def __init__(self, latent_dim: int = 512, in_channels: int = 1):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv1d(in_channels, 64,  kernel_size=11, stride=4,  padding=5),
            nn.GELU(),
            nn.Conv1d(64,         128,  kernel_size=7,  stride=2,  padding=3),
            nn.GELU(),
            nn.Conv1d(128,        256,  kernel_size=5,  stride=2,  padding=2),
            nn.GELU(),
            nn.Conv1d(256,        latent_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(latent_dim)
        for m in self.conv_stack.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.norm(self.conv_stack(x).transpose(1, 2))


class AudioMoE(ExpertBase):
    def __init__(self, latent_dim=512, in_channels=1, top_k=2, dropout=0.1):
        super().__init__("audio", latent_dim, top_k, dropout)
        self.waveform_encoder = WaveformEncoder(latent_dim, in_channels)

        for cls in [PitchSubExpert, RhythmSubExpert, TimbreSubExpert,
                    PhonemeSubExpert, EnvironmentSubExpert]:
            self.add_sub_expert(cls(latent_dim))

        self.local_router = LocalRouter(latent_dim, len(self.sub_experts), top_k)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.waveform_encoder(x)

    def forward(self, x, world_readout=None) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.encode(x)
        gates, idx, aux = self.local_router(tokens, self.training)
        out = self.local_router.dispatch(tokens, gates, idx, self.sub_experts, world_readout)
        return self.output_norm(self.dropout(out)), aux
