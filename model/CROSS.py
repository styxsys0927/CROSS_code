"""CROSS full model used in the main comparison.

The CROSS_repro release keeps only the executable path needed for the main
experiment.  Historical research switches are deliberately omitted to keep the
implementation readable.

Model stages:

1. Neural temporal tokenization: create standard CROSS temporal tokens.
2. Spatial residual token branch: extract a DSAINet-inspired spatial token from
   a wider temporal window and add it through a learnable residual scale.
3. Scale-aware representation learning: refine multiple temporal scales with
   intra-scale and inter-scale attention before adaptive fusion.
4. SNSM: assign learned saliency weights to temporal tokens.
5. GNSI: prepend a learnable global prompt and classify the inferred state.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class TemporalTokenization(nn.Module):
    """Encode fixed-length EEG segments into CROSS temporal tokens."""

    def __init__(
        self,
        channels: int,
        segment_length: int,
        token_stride: int,
        d_model: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if channels <= 0 or segment_length <= 0 or token_stride <= 0:
            raise ValueError("channels, segment_length, and token_stride must be positive")
        self.channels = int(channels)
        self.segment_length = int(segment_length)
        self.token_stride = int(token_stride)
        segment_features = channels * segment_length
        self.encoder = nn.Sequential(
            nn.LayerNorm(segment_features),
            nn.Linear(segment_features, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            if x.shape[1] != 1:
                raise ValueError("Expected input shaped (B,1,C,T)")
            x = x.squeeze(1)
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise ValueError(
                f"Expected input shaped (B,{self.channels},T), got {tuple(x.shape)}"
            )
        if x.shape[2] < self.segment_length:
            raise ValueError(
                f"Input length {x.shape[2]} is shorter than segment length "
                f"{self.segment_length}"
            )

        # Symmetric reflection padding keeps token centers aligned when the
        # segment length is larger than the stride.
        total_padding = max(self.segment_length - self.token_stride, 0)
        if total_padding:
            left_padding = total_padding // 2
            right_padding = total_padding - left_padding
            x = F.pad(x, (left_padding, right_padding), mode="reflect")
        segments = x.unfold(2, self.segment_length, self.token_stride)
        segments = segments.permute(0, 2, 1, 3).contiguous()
        return self.encoder(segments.flatten(start_dim=2))


class MultiKernelTemporalFilterBank(nn.Module):
    """Optional duration-defined temporal filters for dataset-aware kernels."""

    def __init__(
        self,
        sampling_rate: float,
        kernel_durations: Sequence[float],
        scale_filters: Sequence[int],
    ) -> None:
        super().__init__()
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        if len(kernel_durations) != len(scale_filters):
            raise ValueError("kernel_durations and scale_filters must have the same length")
        self.kernel_sizes = tuple(
            max(3, 2 * int((float(duration) * sampling_rate) // 2) + 1)
            for duration in kernel_durations
        )
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(
                    1,
                    int(filters),
                    kernel_size=(1, kernel_size),
                    padding=(0, kernel_size // 2),
                    bias=False,
                )
                for kernel_size, filters in zip(self.kernel_sizes, scale_filters)
            ]
        )
        self.norm = nn.BatchNorm2d(sum(int(value) for value in scale_filters))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(torch.cat([branch(x) for branch in self.branches], dim=1))


class SpatialResidualTokenization(nn.Module):
    """DSAINet-inspired spatial token branch over a wider temporal window."""

    def __init__(
        self,
        channels: int,
        token_window: int,
        token_stride: int,
        d_model: int,
        temporal_filters: int,
        depth_multiplier: int,
        temporal_kernel: int,
        refinement_kernel: int,
        dropout: float,
        sampling_rate: float | None = None,
        temporal_kernel_durations: Sequence[float] | None = None,
        temporal_scale_filters: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if token_window <= 0 or token_stride <= 0:
            raise ValueError("token_window and token_stride must be positive")
        if temporal_kernel <= 0 or temporal_kernel % 2 == 0:
            raise ValueError("temporal_kernel must be a positive odd integer")
        if refinement_kernel <= 0 or refinement_kernel % 2 == 0:
            raise ValueError("refinement_kernel must be a positive odd integer")

        spatial_features = temporal_filters * depth_multiplier
        self.channels = int(channels)
        self.token_window = int(token_window)
        self.token_stride = int(token_stride)

        if temporal_kernel_durations is None:
            self.temporal = nn.Sequential(
                nn.Conv2d(
                    1,
                    temporal_filters,
                    kernel_size=(1, temporal_kernel),
                    padding=(0, temporal_kernel // 2),
                    bias=False,
                ),
                nn.BatchNorm2d(temporal_filters),
            )
        else:
            if sampling_rate is None or sampling_rate <= 0:
                raise ValueError("sampling_rate is required for duration-based kernels")
            if temporal_scale_filters is None:
                num_scales = len(temporal_kernel_durations)
                base, remainder = divmod(temporal_filters, num_scales)
                temporal_scale_filters = tuple(
                    base + (1 if index < remainder else 0)
                    for index in range(num_scales)
                )
            self.temporal = MultiKernelTemporalFilterBank(
                sampling_rate=sampling_rate,
                kernel_durations=temporal_kernel_durations,
                scale_filters=temporal_scale_filters,
            )

        # Depthwise channel-spanning convolution extracts spatial structure
        # inside each token window while keeping temporal filters separated.
        self.spatial = nn.Sequential(
            nn.Conv3d(
                temporal_filters,
                spatial_features,
                kernel_size=(1, channels, 1),
                groups=temporal_filters,
                bias=False,
            ),
            nn.BatchNorm3d(spatial_features),
            nn.ELU(),
        )
        self.refinement = nn.Sequential(
            nn.Conv3d(
                spatial_features,
                spatial_features,
                kernel_size=(1, 1, refinement_kernel),
                padding=(0, 0, refinement_kernel // 2),
                groups=spatial_features,
                bias=False,
            ),
            nn.Conv3d(spatial_features, spatial_features, kernel_size=1, bias=False),
            nn.BatchNorm3d(spatial_features),
            nn.ELU(),
        )
        self.refinement_scale = nn.Parameter(torch.tensor(0.0))
        self.projection = nn.Sequential(
            nn.Linear(spatial_features, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4 or x.shape[1] != 1 or x.shape[2] != self.channels:
            raise ValueError(
                f"Expected input shaped (B,1,{self.channels},T), got {tuple(x.shape)}"
            )

        features = self.temporal(x)
        total_padding = max(self.token_window - self.token_stride, 0)
        if total_padding:
            left_padding = total_padding // 2
            right_padding = total_padding - left_padding
            features = F.pad(features, (left_padding, right_padding, 0, 0), mode="reflect")

        # (B,F,C,T) -> (B,F,C,N,W) -> (B,F,N,C,W)
        token_windows = features.unfold(3, self.token_window, self.token_stride)
        token_windows = token_windows.permute(0, 1, 3, 2, 4).contiguous()
        tokens = self.spatial(token_windows)
        tokens = tokens + self.refinement_scale * self.refinement(tokens)
        tokens = tokens.mean(dim=(3, 4)).transpose(1, 2)
        return self.projection(tokens)


class HybridTokenization(nn.Module):
    """Fuse standard CROSS tokens with the spatial residual branch."""

    def __init__(
        self,
        channels: int,
        segment_length: int,
        token_stride: int,
        spatial_token_window: int,
        d_model: int,
        temporal_filters: int,
        depth_multiplier: int,
        temporal_kernel: int,
        refinement_kernel: int,
        dropout: float,
        sampling_rate: float | None = None,
        temporal_kernel_durations: Sequence[float] | None = None,
        temporal_scale_filters: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.cross_ntt = TemporalTokenization(
            channels=channels,
            segment_length=segment_length,
            token_stride=token_stride,
            d_model=d_model,
            dropout=dropout,
        )
        # The original full-model implementation restores the RNG after
        # constructing the spatial branch so that downstream modules keep the
        # same initialization stream.  We retain that detail for reproducibility.
        rng_state = torch.get_rng_state()
        self.spatial_ntt = SpatialResidualTokenization(
            channels=channels,
            token_window=spatial_token_window,
            token_stride=token_stride,
            d_model=d_model,
            temporal_filters=temporal_filters,
            depth_multiplier=depth_multiplier,
            temporal_kernel=temporal_kernel,
            refinement_kernel=refinement_kernel,
            dropout=dropout,
            sampling_rate=sampling_rate,
            temporal_kernel_durations=temporal_kernel_durations,
            temporal_scale_filters=temporal_scale_filters,
        )
        torch.set_rng_state(rng_state)
        self.spatial_norm = nn.LayerNorm(d_model)
        self.spatial_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cross_tokens = self.cross_ntt(x)
        spatial_tokens = self.spatial_norm(self.spatial_ntt(x))
        if cross_tokens.shape != spatial_tokens.shape:
            raise RuntimeError(
                "CROSS and spatial token shapes differ: "
                f"{tuple(cross_tokens.shape)} versus {tuple(spatial_tokens.shape)}"
            )
        return cross_tokens + self.spatial_scale * spatial_tokens


class BottleneckAttentionBlock(nn.Module):
    """Low-rank attention block used before scale fusion."""

    def __init__(
        self,
        d_model: int,
        attention_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if attention_dim % num_heads:
            raise ValueError("attention_dim must be divisible by num_heads")
        self.query_norm = nn.LayerNorm(d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self.query_projection = nn.Linear(d_model, attention_dim, bias=False)
        self.context_projection = nn.Linear(d_model, attention_dim, bias=False)
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_projection = nn.Linear(attention_dim, d_model, bias=False)
        self.attention_scale = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        normalized_query = self.query_norm(query)
        normalized_context = self.context_norm(context)
        projected_query = self.query_projection(normalized_query)
        projected_context = self.context_projection(normalized_context)
        attended, _ = self.attention(
            projected_query,
            projected_context,
            projected_context,
            need_weights=False,
        )
        attended = self.output_projection(attended)
        query = self.norm1(query + self.dropout(self.attention_scale * attended))
        return self.norm2(query + self.dropout(self.ffn(query)))


class ScaleAwareRepresentationLearning(nn.Module):
    """Multi-scale temporal operators with intra/inter-scale attention."""

    def __init__(
        self,
        d_model: int,
        kernel_sizes: Sequence[int],
        scale_gate_hidden: int,
        attention_dim: int,
        attention_heads: int,
        attention_ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if not kernel_sizes:
            raise ValueError("kernel_sizes must contain at least one scale")
        if any(kernel <= 0 or kernel % 2 == 0 for kernel in kernel_sizes):
            raise ValueError("scale kernels must be positive odd integers")
        self.num_scales = len(kernel_sizes)
        self.operators = nn.ModuleList(
            [
                nn.Conv1d(d_model, d_model, kernel_size=kernel, padding=kernel // 2)
                for kernel in kernel_sizes
            ]
        )
        self.norms = nn.ModuleList([nn.BatchNorm1d(d_model) for _ in kernel_sizes])
        self.operator_activation = nn.ReLU(inplace=True)
        self.intra_scale_attention = BottleneckAttentionBlock(
            d_model=d_model,
            attention_dim=attention_dim,
            num_heads=attention_heads,
            ffn_dim=attention_ffn_dim,
            dropout=dropout,
        )
        self.inter_scale_attention = BottleneckAttentionBlock(
            d_model=d_model,
            attention_dim=attention_dim,
            num_heads=attention_heads,
            ffn_dim=attention_ffn_dim,
            dropout=dropout,
        )
        self.fusion_activation = nn.ReLU(inplace=True)
        self.scale_gate = nn.Sequential(
            nn.Linear(self.num_scales, scale_gate_hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(scale_gate_hidden, self.num_scales, bias=False),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_tokens, d_model = x.shape
        x_channels_first = x.transpose(1, 2)
        scale_features = torch.stack(
            [
                self.operator_activation(norm(operator(x_channels_first))).transpose(1, 2)
                for operator, norm in zip(self.operators, self.norms)
            ],
            dim=1,
        )

        flattened = scale_features.reshape(batch_size * self.num_scales, num_tokens, d_model)
        flattened = self.intra_scale_attention(flattened, flattened)
        scale_features = flattened.reshape(batch_size, self.num_scales, num_tokens, d_model)

        other_scales = (scale_features.sum(dim=1, keepdim=True) - scale_features) / (
            self.num_scales - 1
        )
        interacted = self.inter_scale_attention(
            scale_features.reshape(batch_size * self.num_scales, num_tokens, d_model),
            other_scales.reshape(batch_size * self.num_scales, num_tokens, d_model),
        ).reshape(batch_size, self.num_scales, num_tokens, d_model)

        interacted = self.fusion_activation(interacted)
        descriptors = interacted.mean(dim=(2, 3))
        scale_weights = torch.softmax(self.scale_gate(descriptors), dim=1)
        fused = (interacted * scale_weights[:, :, None, None]).sum(dim=1)
        return fused, scale_weights


class SegmentwiseNeuralSaliencyModeling(nn.Module):
    """Learn normalized saliency weights over temporal tokens."""

    def __init__(self, num_segments: int) -> None:
        super().__init__()
        self.num_segments = int(num_segments)
        self.context_encoder = nn.Sequential(
            nn.Linear(num_segments, num_segments, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_segments, num_segments, bias=False),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[1] != self.num_segments:
            raise ValueError(
                f"Expected {self.num_segments} temporal tokens, received {x.shape[1]}"
            )
        descriptors = x.mean(dim=-1)
        weights = torch.softmax(self.context_encoder(descriptors), dim=1)
        return x * weights.unsqueeze(-1), weights


class GlobalNeuralStateInference(nn.Module):
    """Infer a global neural-state prompt from saliency-weighted tokens."""

    def __init__(
        self,
        num_segments: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.global_state = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position_embedding = nn.Parameter(torch.zeros(1, num_segments + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        prompt = self.global_state.expand(batch_size, -1, -1)
        states = torch.cat((prompt, x), dim=1)
        states = states + self.position_embedding
        states = self.transformer(states)
        return self.output_norm(states[:, 0])


class CROSS(nn.Module):
    """Full CROSS classifier."""

    def __init__(
        self,
        n_classes: int,
        Chans: int,
        Samples: int,
        segment_length: int = 50,
        token_stride: int = 50,
        spatial_token_window: int = 100,
        context_window_size: Sequence[int] | None = None,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        scale_gate_hidden: int = 4,
        prefusion_attention_dim: int = 64,
        prefusion_attention_heads: int = 4,
        prefusion_ffn_dim: int = 128,
        token_temporal_filters: int = 16,
        token_depth_multiplier: int = 2,
        token_temporal_kernel: int = 31,
        token_refinement_kernel: int = 7,
        sampling_rate: float | None = None,
        token_temporal_kernel_durations: Sequence[float] | None = None,
        token_temporal_scale_filters: Sequence[int] | None = None,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        if Samples < segment_length:
            raise ValueError("Samples must be at least segment_length")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if context_window_size is None:
            context_window_size = (3, 5, 7)

        token_padding = max(segment_length - token_stride, 0)
        self.num_segments = (Samples + token_padding - segment_length) // token_stride + 1
        self.ntt = HybridTokenization(
            channels=Chans,
            segment_length=segment_length,
            token_stride=token_stride,
            spatial_token_window=spatial_token_window,
            d_model=d_model,
            temporal_filters=token_temporal_filters,
            depth_multiplier=token_depth_multiplier,
            temporal_kernel=token_temporal_kernel,
            refinement_kernel=token_refinement_kernel,
            dropout=dropout_rate,
            sampling_rate=sampling_rate,
            temporal_kernel_durations=token_temporal_kernel_durations,
            temporal_scale_filters=token_temporal_scale_filters,
        )
        self.sarl = ScaleAwareRepresentationLearning(
            d_model=d_model,
            kernel_sizes=context_window_size,
            scale_gate_hidden=scale_gate_hidden,
            attention_dim=prefusion_attention_dim,
            attention_heads=prefusion_attention_heads,
            attention_ffn_dim=prefusion_ffn_dim,
            dropout=dropout_rate,
        )
        self.snsm = SegmentwiseNeuralSaliencyModeling(self.num_segments)
        self.gnsi = GlobalNeuralStateInference(
            num_segments=self.num_segments,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, n_classes),
        )

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        tokens = self.ntt(x)
        fused_tokens, scale_weights = self.sarl(tokens)
        salient_tokens, saliency_weights = self.snsm(fused_tokens)
        global_state = self.gnsi(salient_tokens)
        return global_state, {
            "scale_weights": scale_weights,
            "saliency_weights": saliency_weights,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global_state, _ = self.forward_features(x)
        return self.classifier(global_state)
