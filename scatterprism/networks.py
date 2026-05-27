"""Neural-network backbones for ScatterPrism generative models.

Plain MLP and residual MLP backbones, time embeddings (sinusoidal, Fourier,
learned), and the (un)conditional variants wired into the Flow Matching and
Diffusion models.
"""

import logging
import math

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}

_NORMS: dict[str | None, type[nn.Module]] = {
    "layer": nn.LayerNorm,
    "batch": nn.BatchNorm1d,
    None: nn.Identity,
}


def _ensure_two_hidden_dims(hidden_dims: list[int] | int) -> list[int]:
    """ResidualNetwork needs at least two hidden dims to insert a ResBlock."""
    if isinstance(hidden_dims, int):
        return [hidden_dims, hidden_dims]
    if len(hidden_dims) == 1:
        return hidden_dims + hidden_dims
    return list(hidden_dims)


class BaseNetwork(nn.Module):
    """Marker base class so model code can ``isinstance`` against it."""


class MLP(BaseNetwork):
    """Bare MLP backbone (models wrap this and handle their own conditioning).

    Args:
        input_dim:   Input dimension.
        output_dim:  Output dimension.
        hidden_dims: Hidden-layer widths (``int`` is broadcast to one layer).
        activation:  One of ``"relu" | "silu" | "gelu" | "tanh"``.
        norm:        ``"layer"``, ``"batch"`` or ``None``.
        dropout:     Dropout probability applied after each activation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | int = 256,
        activation: str = "relu",
        norm: str | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        act_fn = _ACTIVATIONS[activation]

        layers: list[nn.Module] = []
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if norm == "layer":
                    layers.append(nn.LayerNorm(dims[i + 1]))
                elif norm == "batch":
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(act_fn())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualNetwork(BaseNetwork):
    """Stack of residual MLP blocks with optional norm/activation/dropout."""

    class ResBlock(nn.Module):
        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int = 512,
            activation: str = "relu",
            norm: str | None = None,
            dropout: float = 0.0,
        ):
            super().__init__()
            act_fn = _ACTIVATIONS[activation]
            norm_fn = _NORMS[norm]

            layers: list[nn.Module] = [
                nn.Linear(input_dim, hidden_dim),
                norm_fn(hidden_dim),
                act_fn(),
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers += [
                nn.Linear(hidden_dim, output_dim),
                norm_fn(output_dim),
                act_fn(),
            ]
            self.net = nn.Sequential(*layers)
            self.skip = (
                nn.Linear(input_dim, output_dim)
                if input_dim != output_dim else nn.Identity()
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x) + self.skip(x)

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: list[int] | None = None,
        activation: str = "relu",
        norm: str | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = [256]
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim[0])]
        c = hidden_dim[0]
        for s in hidden_dim[1:]:
            layers.append(self.ResBlock(
                input_dim=c, output_dim=s, hidden_dim=c,
                norm=norm, activation=activation, dropout=dropout,
            ))
            c = s
        layers.append(nn.Linear(c, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ── Time embeddings ──────────────────────────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding (used by DDPM)."""

    def __init__(self, dim: int, max_period: float = 1000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class LearnedEmbedding(nn.Module):
    """Two-layer learned embedding (legacy; kept for old CFM checkpoints)."""

    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.net(x)


class FourierEmbedding(nn.Module):
    """Fourier features for continuous time ``t ∈ [0, 1]`` (default for CFM).

    Fixed sinusoidal bases with geometrically-spaced frequencies (1 …
    ``max_freq`` cycles per unit interval) followed by a learned linear
    projection — supplies ready-made high-frequency features so the network
    doesn't have to rediscover them from a scalar input.

    Args:
        embed_dim: Output feature dimension (must be even).
        max_freq:  Highest frequency in cycles per unit interval. Default 64
                   resolves velocity changes at scale ~ 1/64.
    """

    def __init__(self, embed_dim: int, max_freq: float = 64.0):
        super().__init__()
        half = embed_dim // 2
        freqs = torch.exp(
            torch.linspace(0.0, math.log(max_freq), half)
        ) * 2.0 * math.pi
        self.register_buffer('freqs', freqs)  # fixed, not trained
        self.proj = nn.Linear(embed_dim, embed_dim)  # learned mixing
        self.dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)                       # [B, 1]
        args = t * self.freqs[None, :]                # [B, half]
        fourier = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.proj(fourier)                     # [B, embed_dim]


# ── Diffusion backbone ───────────────────────────────────────────────────────

class DiffusionMLP(BaseNetwork):
    """MLP wrapping a sinusoidal time embedding, used by DDPM."""

    def __init__(
        self,
        data_dim: int,
        time_embed_dim: int = 64,
        hidden_dims: list[int] | int = 256,
        **mlp_kwargs,
    ):
        super().__init__()
        self.time_embed = SinusoidalEmbedding(time_embed_dim)
        self.time_proj = nn.Linear(time_embed_dim, time_embed_dim)
        self.net = MLP(
            input_dim=data_dim + time_embed_dim,
            output_dim=data_dim,
            hidden_dims=hidden_dims,
            **mlp_kwargs,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``x``: noisy data ``[B, data_dim]``, ``t``: timestep ``[B]``."""
        t_embed = self.time_proj(self.time_embed(t))
        return self.net(torch.cat([x, t_embed], dim=-1))


# ── Flow-matching backbones ──────────────────────────────────────────────────

class FlowMatchingMLP(BaseNetwork):
    """MLP backbone for unconditional (Conditional) Flow Matching."""

    def __init__(
        self,
        data_dim: int,
        time_embed_dim: int = 64,
        hidden_dims: list[int] | int = 256,
        use_ot_cost: bool = False,
        time_embedding: str = "fourier",  # "fourier" (default) or "learned" (legacy)
        **mlp_kwargs,
    ):
        super().__init__()
        self.use_ot_cost = use_ot_cost
        self.time_embed = (
            LearnedEmbedding(1, time_embed_dim) if time_embedding == "learned"
            else FourierEmbedding(time_embed_dim)
        )
        extra_dim = 1 if use_ot_cost else 0
        self.net = MLP(
            input_dim=data_dim + time_embed_dim + extra_dim,
            output_dim=data_dim,
            hidden_dims=hidden_dims,
            **mlp_kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        ot_cost: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``x``: x_t ``[B, D]``, ``t``: ``[B]`` ∈ [0,1]. Returns velocity ``[B, D]``."""
        inputs = [x, self.time_embed(t)]
        if self.use_ot_cost and ot_cost is not None:
            inputs.append(ot_cost)
        return self.net(torch.cat(inputs, dim=-1))


class FlowMatchingResNet(BaseNetwork):
    """ResNet backbone for unconditional CFM (better gradient flow than MLP)."""

    def __init__(
        self,
        data_dim: int,
        time_embed_dim: int = 64,
        hidden_dims: list[int] | int = 256,
        use_ot_cost: bool = False,
        time_embedding: str = "fourier",
        **resnet_kwargs,
    ):
        super().__init__()
        self.use_ot_cost = use_ot_cost
        self.time_embed = (
            LearnedEmbedding(1, time_embed_dim) if time_embedding == "learned"
            else FourierEmbedding(time_embed_dim)
        )
        extra_dim = 1 if use_ot_cost else 0
        self.net = ResidualNetwork(
            input_dim=data_dim + time_embed_dim + extra_dim,
            output_dim=data_dim,
            hidden_dim=_ensure_two_hidden_dims(hidden_dims),
            **resnet_kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        ot_cost: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs = [x, self.time_embed(t)]
        if self.use_ot_cost and ot_cost is not None:
            inputs.append(ot_cost)
        return self.net(torch.cat(inputs, dim=-1))


# ── Generic conditional MLP (kept for back-compat / future use) ──────────────

class ConditionalMLP(BaseNetwork):
    """MLP with generic conditioning (class labels, context). Unused at present."""

    time_embed: nn.Module  # Sequential or LearnedEmbedding — keep loose typing.

    def __init__(
        self,
        data_dim: int,
        cond_dim: int,
        time_embed_dim: int = 64,
        cond_embed_dim: int = 64,
        hidden_dims: list[int] | int = 256,
        time_embedding: str = "sinusoidal",  # "sinusoidal" or "learned"
        **mlp_kwargs,
    ):
        super().__init__()
        if time_embedding == "sinusoidal":
            self.time_embed = nn.Sequential(
                SinusoidalEmbedding(time_embed_dim),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
        else:
            self.time_embed = LearnedEmbedding(1, time_embed_dim)
        self.cond_embed = nn.Linear(cond_dim, cond_embed_dim)
        self.net = MLP(
            input_dim=data_dim + time_embed_dim + cond_embed_dim,
            output_dim=data_dim,
            hidden_dims=hidden_dims,
            **mlp_kwargs,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat(
            [x, self.time_embed(t), self.cond_embed(cond)], dim=-1
        ))


# ── Conditional CFM backbones (detector unfolding) ───────────────────────────

class ConditionalFlowMatchingMLP(BaseNetwork):
    """MLP CFM backbone with explicit conditioning (detector data → particle level).

    Forward signature: ``(x_t, t, cond) -> velocity``.
    """

    def __init__(
        self,
        data_dim: int,
        cond_dim: int,
        time_embed_dim: int = 64,
        cond_embed_dim: int | None = None,
        hidden_dims: list[int] | int = 256,
        time_embedding: str = "fourier",
        **mlp_kwargs,
    ):
        super().__init__()
        if cond_embed_dim is None:
            cond_embed_dim = data_dim
        self.time_embed = (
            LearnedEmbedding(1, time_embed_dim) if time_embedding == "learned"
            else FourierEmbedding(time_embed_dim)
        )
        # Embed detector data through a small MLP so the main backbone sees a
        # learned representation rather than raw conditioning values.
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, cond_embed_dim),
            nn.SiLU(),
            nn.Linear(cond_embed_dim, cond_embed_dim),
        )
        self.net = MLP(
            input_dim=data_dim + time_embed_dim + cond_embed_dim,
            output_dim=data_dim,
            hidden_dims=hidden_dims,
            **mlp_kwargs,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat(
            [x, self.time_embed(t), self.cond_embed(cond)], dim=-1
        ))


class ConditionalFlowMatchingResNet(BaseNetwork):
    """ResNet CFM backbone with explicit conditioning. Forward: ``(x_t, t, cond)``."""

    def __init__(
        self,
        data_dim: int,
        cond_dim: int,
        time_embed_dim: int = 64,
        cond_embed_dim: int | None = None,
        hidden_dims: list[int] | int = 256,
        time_embedding: str = "fourier",
        **resnet_kwargs,
    ):
        super().__init__()
        if cond_embed_dim is None:
            cond_embed_dim = data_dim
        self.time_embed = (
            LearnedEmbedding(1, time_embed_dim) if time_embedding == "learned"
            else FourierEmbedding(time_embed_dim)
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, cond_embed_dim),
            nn.SiLU(),
            nn.Linear(cond_embed_dim, cond_embed_dim),
        )
        self.net = ResidualNetwork(
            input_dim=data_dim + time_embed_dim + cond_embed_dim,
            output_dim=data_dim,
            hidden_dim=_ensure_two_hidden_dims(hidden_dims),
            **resnet_kwargs,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat(
            [x, self.time_embed(t), self.cond_embed(cond)], dim=-1
        ))