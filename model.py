"""Standalone BiLSTM: ordered reflectance patches -> geometry parameters.

The sole sequence encoder is nn.LSTM(bidirectional=True). There are no
Transformer, attention, or ResNet blocks. The patch projection and position
encoding match the earlier plain Metaformer input representation.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class BiLSTMConfig:
    spectrum_points: int = 1000
    patch_count: int = 100
    embedding_dim: int = 128
    hidden_size: int = 64  # Per direction; concatenated output has 128 features.
    recurrent_layers: int = 2
    mlp_dim: int = 256
    dropout: float = 0.1
    geometry_dim: int = 7
    angle_symmetry: str = "square_periodic"
    position_encoding: bool = True

    def __post_init__(self):
        for name in ("spectrum_points", "patch_count", "embedding_dim", "hidden_size",
                     "recurrent_layers", "mlp_dim", "geometry_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.spectrum_points % self.patch_count:
            raise ValueError("spectrum_points must be divisible by patch_count")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if not isinstance(self.position_encoding, bool):
            raise ValueError("position_encoding must be boolean")
        if self.angle_symmetry not in ("none", "square_mirror", "square_periodic"):
            raise ValueError("Unknown angle symmetry")
        if self.angle_symmetry == "square_mirror" and self.geometry_dim != 6:
            raise ValueError("Mirror angle loss requires six outputs")
        if self.angle_symmetry == "square_periodic" and self.geometry_dim != 7:
            raise ValueError("Quadrant periodic loss requires seven outputs")

    @classmethod
    def from_dict(cls, values):
        return cls(**dict(values))

    @property
    def patch_size(self):
        return self.spectrum_points // self.patch_count


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, length, dim):
        super().__init__()
        positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        indices = torch.arange(0, dim, 2, dtype=torch.float32)
        angles = positions * torch.exp(-indices * math.log(10000.0) / dim)
        encoding = torch.zeros(1, length, dim)
        encoding[0, :, 0::2] = angles.sin()
        encoding[0, :, 1::2] = angles[:, :dim // 2].cos()
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, tokens):
        return tokens + self.encoding[:, :tokens.shape[1]].to(tokens.dtype)


class InverseBiLSTM(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or BiLSTMConfig()
        c = self.config
        # This shared linear patch projection is implemented as a strided Conv1d.
        # It is not a convolutional feature frontend.
        self.patch_embedding = nn.Conv1d(1, c.embedding_dim,
                                        kernel_size=c.patch_size, stride=c.patch_size)
        self.position_encoding = (SinusoidalPositionEncoding(c.patch_count, c.embedding_dim)
                                  if c.position_encoding else nn.Identity())
        self.bilstm = nn.LSTM(c.embedding_dim, c.hidden_size,
                             num_layers=c.recurrent_layers, batch_first=True,
                             bidirectional=True,
                             dropout=c.dropout if c.recurrent_layers > 1 else 0.0)
        features = 2 * c.hidden_size
        self.final_norm = nn.LayerNorm(features)
        self.regressor = nn.Sequential(nn.Linear(features, c.mlp_dim), nn.GELU(),
                                       nn.Linear(c.mlp_dim, c.geometry_dim), nn.Sigmoid())

    def encode(self, spectrum):
        if spectrum.ndim == 3 and spectrum.shape[1] == 1:
            spectrum = spectrum[:, 0, :]
        if spectrum.ndim != 2 or spectrum.shape[1] != self.config.spectrum_points:
            raise ValueError(f"Expected [batch,{self.config.spectrum_points}] or [batch,1,{self.config.spectrum_points}]")
        tokens = self.patch_embedding(spectrum.unsqueeze(1)).transpose(1, 2)
        tokens = self.position_encoding(tokens)
        # No recurrent state is carried between spectra or batches.
        sequence, _ = self.bilstm(tokens)
        return self.final_norm(sequence)

    def forward(self, spectrum):
        return self.regressor(self.encode(spectrum).mean(dim=1))


def geometry_mse(prediction: Tensor, target: Tensor, angle_symmetry="none"):
    """Match the previous geometry loss, including 90-degree square symmetry."""
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes must match")
    if angle_symmetry == "square_periodic":
        if prediction.shape[-1] != 7:
            raise ValueError("Quadrant periodic loss requires seven outputs")
        difference = prediction - target
        angles = torch.remainder(difference[..., 5:] + .5, 1.0) - .5
        return torch.cat((difference[..., :5], angles), dim=-1).square().mean()
    if angle_symmetry == "square_mirror":
        if prediction.shape[-1] != 6:
            raise ValueError("Mirror angle loss requires six outputs")
        prediction = torch.cat((prediction[..., :5], (2*prediction[..., 5:]-1).abs()), dim=-1)
        target = torch.cat((target[..., :5], (2*target[..., 5:]-1).abs()), dim=-1)
    elif angle_symmetry != "none":
        raise ValueError("Unknown angle symmetry")
    return F.mse_loss(prediction, target)
