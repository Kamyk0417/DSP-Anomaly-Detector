"""Autoencoder, LSTM autoencoder and Transformer autoencoder models.

All models consume a flat feature vector per window (raw samples, an FFT
magnitude spectrum, or the concatenation of both) and implement the section 5.3
contract::

    forward(x) -> (x_hat, latent)
    reconstruction_error(x) -> Tensor[float]     # one score per window

Checkpoints are ``torch.save({"model": state_dict, "config": cfg}, path)`` so
that inference is self-contained.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)

ARCHITECTURES: tuple[str, ...] = ("autoencoder", "lstm_ae", "transformer_ae")
ARCH_DEFAULT = "autoencoder"


class BaseAutoencoder(nn.Module):
    """Shared interface for every reconstruction-based detector."""

    arch: str = "base"

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:  # pragma: no cover - abstract
        raise NotImplementedError

    def reconstruct(self, x: Tensor) -> Tensor:
        x_hat, _ = self(x)
        return x_hat

    @torch.no_grad()
    def reconstruction_error(self, x: Tensor) -> Tensor:
        """Mean squared reconstruction error per window."""
        x_hat = self.reconstruct(x)
        return torch.mean((x_hat - x) ** 2, dim=1)


class ConvAutoencoder(BaseAutoencoder):
    """1D convolutional autoencoder; the baseline detector.

    ``latent_dim`` is the width of the bottleneck *code*: the encoder compresses
    the window to ``latent_dim`` channels over ``sequence_length / 8`` positions,
    and ``forward`` also returns the pooled ``(batch, latent_dim)`` vector for
    callers that want a single embedding per window. A globally pooled latent
    (one 32-d vector for 1024 samples) collapses to the mean-prediction solution
    in practice and never learns to reconstruct, so the code stays spatial.
    """

    arch = "autoencoder"

    def __init__(self, sequence_length: int, hidden: int = 128, latent_dim: int = 32) -> None:
        super().__init__()
        if sequence_length < 8:
            raise ValueError(f"sequence_length must be at least 8, got {sequence_length}")
        if hidden < 4:
            raise ValueError(f"hidden must be at least 4, got {hidden}")
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")

        self.sequence_length = int(sequence_length)
        self.hidden = int(hidden)
        self.latent_dim = int(latent_dim)

        # Every strided convolution halves the length with kernel 4 / pad 1.
        self.encoder = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, hidden, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(hidden, hidden, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(hidden, 1, kernel_size=4, stride=2, padding=1),
        )

    def _check_length(self, x: Tensor) -> None:
        if x.dim() != 2 or x.shape[1] != self.sequence_length:
            raise ValueError(
                f"{self.arch} expects input of shape (batch, {self.sequence_length}), "
                f"got {tuple(x.shape)}"
            )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        self._check_length(x)
        code = self.encoder(x.unsqueeze(1))
        latent = torch.mean(code, dim=2)

        decoded = self.decoder(code).squeeze(1)
        if decoded.shape[1] > self.sequence_length:
            decoded = decoded[:, : self.sequence_length]
        elif decoded.shape[1] < self.sequence_length:
            decoded = torch.nn.functional.pad(
                decoded, (0, self.sequence_length - decoded.shape[1])
            )
        return decoded, latent


class LstmAutoencoder(BaseAutoencoder):
    """LSTM autoencoder: the encoder's last state seeds a positional decoder."""

    arch = "lstm_ae"

    def __init__(self, sequence_length: int, hidden: int = 128, latent_dim: int = 32) -> None:
        super().__init__()
        if sequence_length < 8:
            raise ValueError(f"sequence_length must be at least 8, got {sequence_length}")
        if hidden < 2:
            raise ValueError(f"hidden must be at least 2, got {hidden}")
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")

        self.sequence_length = int(sequence_length)
        self.hidden = int(hidden)
        self.latent_dim = int(latent_dim)

        self.encoder = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden)
        # A purely constant decoder input forces the LSTM to synthesise all time
        # structure from its own state; a learned positional signal makes the
        # reconstruction task well-conditioned and trainable.
        self.positions = nn.Parameter(torch.zeros(1, sequence_length, hidden))
        self.decoder = nn.LSTM(input_size=hidden, hidden_size=hidden, batch_first=True)
        self.to_output = nn.Linear(hidden, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.dim() != 2 or x.shape[1] != self.sequence_length:
            raise ValueError(
                f"{self.arch} expects input of shape (batch, {self.sequence_length}), "
                f"got {tuple(x.shape)}"
            )

        _, (hidden_state, _) = self.encoder(x.unsqueeze(2))
        latent = self.to_latent(hidden_state.squeeze(0))

        seed = torch.tanh(self.from_latent(latent)).unsqueeze(1)
        decoder_input = seed + self.positions
        decoded, _ = self.decoder(decoder_input)
        return self.to_output(decoded).squeeze(2), latent


class TransformerAutoencoder(BaseAutoencoder):
    """Transformer encoder with a reconstruction head."""

    arch = "transformer_ae"

    def __init__(self, sequence_length: int, hidden: int = 128, latent_dim: int = 32) -> None:
        super().__init__()
        if sequence_length < 8:
            raise ValueError(f"sequence_length must be at least 8, got {sequence_length}")
        if hidden < 8:
            raise ValueError(f"hidden must be at least 8, got {hidden}")
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")

        self.sequence_length = int(sequence_length)
        self.hidden = int(hidden)
        self.latent_dim = int(latent_dim)
        self.heads = next(count for count in (8, 4, 2, 1) if hidden % count == 0)

        self.embed = nn.Linear(1, hidden)
        self.register_buffer(
            "positional", self._positional_encoding(sequence_length, hidden), persistent=False
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=self.heads,
            dim_feedforward=hidden * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=2, enable_nested_tensor=False
        )
        # Spatial bottleneck: latent_dim channels per position, pooled for the API.
        self.to_code = nn.Linear(hidden, latent_dim)
        self.from_code = nn.Linear(latent_dim, hidden)
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=self.heads,
                dim_feedforward=hidden * 2,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=2,
            enable_nested_tensor=False,
        )
        self.to_output = nn.Linear(hidden, 1)

    @staticmethod
    def _positional_encoding(sequence_length: int, hidden: int) -> Tensor:
        position = torch.arange(sequence_length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, hidden, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden)
        )
        encoding = torch.zeros(sequence_length, hidden)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)[:, : encoding[:, 1::2].shape[1]]
        return encoding.unsqueeze(0)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.dim() != 2 or x.shape[1] != self.sequence_length:
            raise ValueError(
                f"{self.arch} expects input of shape (batch, {self.sequence_length}), "
                f"got {tuple(x.shape)}"
            )

        embedded = self.embed(x.unsqueeze(2)) + self.positional
        encoded = self.encoder(embedded)
        code = self.to_code(encoded)
        latent = torch.mean(code, dim=1)

        decoded_input = self.from_code(code) + self.positional
        decoded = self.to_output(self.decoder(decoded_input)).squeeze(2)
        return decoded, latent


def build_model(
    arch: str, sequence_length: int, *, hidden: int = 128, latent_dim: int = 32
) -> BaseAutoencoder:
    """Instantiate one of the three architectures."""
    if arch == "autoencoder":
        return ConvAutoencoder(sequence_length, hidden=hidden, latent_dim=latent_dim)
    if arch == "lstm_ae":
        return LstmAutoencoder(sequence_length, hidden=hidden, latent_dim=latent_dim)
    if arch == "transformer_ae":
        return TransformerAutoencoder(sequence_length, hidden=hidden, latent_dim=latent_dim)
    raise ValueError(f"unknown architecture {arch!r}; expected one of {ARCHITECTURES}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def save_checkpoint(path: Path, model: BaseAutoencoder, config: dict[str, Any]) -> None:
    """Write a self-contained checkpoint: weights plus everything inference needs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": dict(config)}, path)
    logger.debug("wrote checkpoint %s", path)


def load_checkpoint(
    path: Path, device: torch.device | str = "cpu"
) -> tuple[BaseAutoencoder, dict[str, Any]]:
    """Rebuild a model from a checkpoint written by :func:`save_checkpoint`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint {path} does not exist")

    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or "model" not in payload or "config" not in payload:
        raise ValueError(f"{path} is not a detector checkpoint (expected 'model' and 'config')")

    config = payload["config"]
    model = build_model(
        str(config["arch"]),
        int(config["sequence_length"]),
        hidden=int(config.get("hidden", 128)),
        latent_dim=int(config.get("latent_dim", 32)),
    )
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model, config

