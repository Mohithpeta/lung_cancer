"""Custom TabICL-inspired multimodal fusion model.

Input:
    A batch dictionary containing:
        behavioral:   [batch_size, behavioral_features]
        environmental:[batch_size, environmental_features]
        clinical:     [batch_size, clinical_features]

Output:
    Probability of lung cancer for each sample, shape [batch_size].
"""

from collections.abc import Mapping

import torch
from torch import nn


MODALITIES = ("behavioral", "environmental", "clinical")


class ModalityEncoder(nn.Module):
    """Encode feature values as tokens and learn within-modality interactions."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if input_dim < 1:
            raise ValueError("Each modality must contain at least one feature.")

        self.input_dim = input_dim
        self.feature_projection = nn.Linear(1, hidden_dim)
        self.position_embedding = nn.Parameter(
            torch.randn(1, input_dim, hidden_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input shape [batch, {self.input_dim}], "
                f"received {tuple(x.shape)}."
            )

        # Treat each preprocessed feature as an individual token.
        tokens = self.feature_projection(x.unsqueeze(-1))
        tokens = tokens + self.position_embedding

        encoded = self.transformer(tokens)

        # Pool feature tokens into one representation per modality.
        representation = encoded.mean(dim=1)
        return self.normalization(representation)


class PairwiseInteraction(nn.Module):
    """Learn an interaction between two modality representations."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(torch.cat((first, second), dim=-1))


class TabICLXG(nn.Module):
    """Custom multimodal model for binary lung-cancer classification."""

    def __init__(
        self,
        input_dims: Mapping[str, int],
        hidden_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if set(input_dims) != set(MODALITIES):
            raise ValueError(
                f"input_dims must contain exactly these keys: {MODALITIES}"
            )

        if hidden_dim < 1 or num_heads < 1 or hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be positive and divisible by num_heads."
            )

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        self.input_dims = dict(input_dims)

        # 1. Modality-specific representation learning.
        self.encoders = nn.ModuleDict(
            {
                name: ModalityEncoder(
                    input_dim=self.input_dims[name],
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    dropout=dropout,
                )
                for name in MODALITIES
            }
        )

        # 2. Explicit pairwise cross-modal interactions.
        self.interaction_be = PairwiseInteraction(hidden_dim, dropout)
        self.interaction_bc = PairwiseInteraction(hidden_dim, dropout)
        self.interaction_ec = PairwiseInteraction(hidden_dim, dropout)

        # 3. Interaction-aware modality gating.
        self.gate_networks = nn.ModuleDict(
            {
                name: nn.Linear(hidden_dim, 1)
                for name in MODALITIES
            }
        )

        # 4. One fusion operation: concatenate gated representations.
        # 5. Binary classifier: Dense -> ReLU -> Dropout -> Dense(1).
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _representations(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(batch, Mapping):
            raise TypeError("Input must be a batch dictionary.")

        missing = set(MODALITIES) - set(batch)
        if missing:
            raise KeyError(f"Missing modality inputs: {sorted(missing)}")

        inputs = {}

        for name in MODALITIES:
            x = batch[name]

            if not isinstance(x, torch.Tensor):
                raise TypeError(f"'{name}' must be a torch.Tensor.")

            if not x.is_floating_point():
                raise TypeError(f"'{name}' must contain floating-point values.")

            if not torch.isfinite(x).all():
                raise ValueError(f"'{name}' contains NaN or infinite values.")

            inputs[name] = x

        batch_sizes = {x.shape[0] for x in inputs.values()}
        if len(batch_sizes) != 1:
            raise ValueError("All modalities must have the same batch size.")

        # Encode each modality independently.
        behavioral = self.encoders["behavioral"](inputs["behavioral"])
        environmental = self.encoders["environmental"](
            inputs["environmental"]
        )
        clinical = self.encoders["clinical"](inputs["clinical"])

        # Compute pairwise interactions.
        interaction_be = self.interaction_be(
            behavioral, environmental
        )
        interaction_bc = self.interaction_bc(behavioral, clinical)
        interaction_ec = self.interaction_ec(
            environmental, clinical
        )

        # Inject pairwise information into the corresponding modalities.
        behavioral = behavioral + (interaction_be + interaction_bc) / 2
        environmental = environmental + (
            interaction_be + interaction_ec
        ) / 2
        clinical = clinical + (interaction_bc + interaction_ec) / 2

        return behavioral, environmental, clinical

    def forward_logits(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return raw logits for numerically stable BCEWithLogitsLoss training."""

        representations = self._representations(batch)

        # Learn relative importance of the three interaction-enhanced modalities.
        gate_logits = torch.cat(
            [
                self.gate_networks[name](representation)
                for name, representation in zip(
                    MODALITIES, representations
                )
            ],
            dim=1,
        )
        gate_weights = torch.softmax(gate_logits, dim=1)

        gated = [
            representation * gate_weights[:, i : i + 1]
            for i, representation in enumerate(representations)
        ]

        # Single multimodal fusion operation.
        fused = torch.cat(gated, dim=1)

        return self.classifier(fused).squeeze(-1)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return lung-cancer probabilities in the range [0, 1]."""
        return torch.sigmoid(self.forward_logits(batch))