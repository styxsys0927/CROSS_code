"""Model factory used by the CROSS reproducibility scripts.

Only the main-comparison models are exposed here: CROSS, DSAINet, and the
selected baseline models.
"""

from __future__ import annotations

import torch

from model.ADFCNN import ADFCNN
from model.CROSS import CROSS
from model.CTNet import CTNet
from model.DSAINet import DSAINet
from model.DeepConvNet import DeepConvNet
from model.Deformer import Deformer
from model.EEGNet import EEGNet
from model.LMDANet import LMDANet
from model.MSCFormer import MSCFormer
from model.ShallowConvNet import ShallowConvNet
from model.TMSANet import TMSANet


def _adam(model, config):
    return torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"].get("weight_decay", 0.0),
    )


def _adam_betas(model, config):
    return torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["lr"],
        betas=(config["train"].get("b1", 0.5), config["train"].get("b2", 0.999)),
        weight_decay=config["train"].get("weight_decay", 0.0),
    )


def load_model(model_params, config, n_class, n_channels, n_times, logger, device):
    """Build a model, criterion, and optimizer from a YAML config."""

    name = model_params["name"]
    criterion = torch.nn.CrossEntropyLoss()

    if name == "CROSS":
        # CROSS full model: temporal tokens + spatial residual branch +
        # scale-aware attention fusion + SNSM + prompt-based GNSI.
        model = CROSS(
            n_classes=n_class,
            Chans=n_channels,
            Samples=n_times,
            segment_length=model_params["segment_length"],
            token_stride=model_params["token_stride"],
            spatial_token_window=model_params["spatial_token_window"],
            context_window_size=model_params["context_window_size"],
            d_model=model_params["d_model"],
            nhead=model_params["nhead"],
            num_layers=model_params["num_layers"],
            dim_feedforward=model_params["dim_feedforward"],
            scale_gate_hidden=model_params["scale_gate_hidden"],
            prefusion_attention_dim=model_params["prefusion_attention_dim"],
            prefusion_attention_heads=model_params["prefusion_attention_heads"],
            prefusion_ffn_dim=model_params["prefusion_ffn_dim"],
            token_temporal_filters=model_params["token_temporal_filters"],
            token_depth_multiplier=model_params["token_depth_multiplier"],
            token_temporal_kernel=model_params["token_temporal_kernel"],
            sampling_rate=model_params.get("sampling_rate"),
            token_temporal_kernel_durations=model_params.get(
                "token_temporal_kernel_durations"
            ),
            token_temporal_scale_filters=model_params.get(
                "token_temporal_scale_filters"
            ),
            token_refinement_kernel=model_params["token_refinement_kernel"],
            dropout_rate=model_params["dropout_rate"],
        ).to(device)
        optimizer = _adam(model, config)

    elif name == "DSAINet":
        model = DSAINet(
            n_classes=n_class,
            Chans=n_channels,
            Samples=n_times,
            emb_size=model_params["emb_size"],
            heads=model_params["heads"],
            attn_depth=model_params["attn_depth"],
            attn_dropout=model_params["attn_dropout"],
            eeg1_f1=model_params["eeg1_f1"],
            eeg1_kernel_size=model_params["eeg1_kernel_size"],
            eeg1_D=model_params["eeg1_D"],
            eeg1_pooling_size1=model_params["eeg1_pooling_size1"],
            eeg1_pooling_size2=model_params["eeg1_pooling_size2"],
            eeg1_dropout_rate=model_params["eeg1_dropout_rate"],
            branch_1_kernels=model_params["branch_1_kernels"],
            branch_2_kernels=model_params["branch_2_kernels"],
            conv_expansion=model_params["conv_expansion"],
            conv_dropout=model_params["conv_dropout"],
            intra_ffn_expansion=model_params["intra_ffn_expansion"],
            inter_ffn_expansion=model_params["inter_ffn_expansion"],
            big_residual=model_params["big_residual"],
            big_residual_learnable=model_params["big_residual_learnable"],
            cls_dropout=model_params["cls_dropout"],
        ).to(device)
        optimizer = _adam(model, config)

    elif name == "EEGNet":
        model = EEGNet(
            n_classes=n_class,
            Chans=n_channels,
            Samples=n_times,
            kernLenght=model_params["kernLenght"],
            F1=model_params["F1"],
            D=model_params["D"],
            F2=model_params["F2"],
            dropoutRate=model_params["dropoutRate"],
            norm_rate=model_params["norm_rate"],
        ).to(device)
        optimizer = _adam(model, config)

    elif name == "ShallowConvNet":
        model = ShallowConvNet(
            nChan=n_channels,
            nTime=n_times,
            nClass=n_class,
            dropoutP=model_params["dropoutRate"],
        ).to(device)
        optimizer = _adam(model, config)

    elif name == "DeepConvNet":
        model = DeepConvNet(
            nChan=n_channels,
            nTime=n_times,
            nClass=n_class,
            dropoutP=model_params["dropoutRate"],
        ).to(device)
        optimizer = _adam(model, config)

    elif name == "ADFCNN":
        model = ADFCNN(num_classes=n_class, num_channels=n_channels).to(device)
        optimizer = _adam(model, config)

    elif name == "CTNet":
        model = CTNet(
            heads=model_params["heads"],
            emb_size=model_params["emb_size"],
            depth=model_params["depth"],
            eeg1_f1=model_params["eeg1_f1"],
            eeg1_kernel_size=model_params["eeg1_kernel_size"],
            eeg1_D=model_params["eeg1_D"],
            eeg1_pooling_size1=model_params["eeg1_pooling_size1"],
            eeg1_pooling_size2=model_params["eeg1_pooling_size2"],
            eeg1_dropout_rate=model_params["eeg1_dropout_rate"],
            flatten_eeg1=model_params["flatten_eeg1"],
            number_class=n_class,
            eeg1_number_channel=n_channels,
        ).to(device)
        optimizer = _adam_betas(model, config)

    elif name == "LMDANet":
        model = LMDANet(
            num_classes=n_class,
            chans=n_channels,
            samples=n_times,
            depth=model_params["depth"],
            kernel=model_params["kernel"],
            channel_depth1=model_params["channel_depth1"],
            channel_depth2=model_params["channel_depth2"],
            ave_depth=model_params["ave_depth"],
            avepool=model_params["avepool"],
        ).to(device)
        optimizer = _adam(model, config)

    elif name == "MSCFormer":
        model = MSCFormer(
            class_num=n_class,
            dropout_rate=model_params["dropout_rate"],
            chn=n_channels,
        ).to(device)
        optimizer = _adam_betas(model, config)

    elif name == "TMSANet":
        model = TMSANet(
            in_planes=n_channels,
            radix=model_params["radix"],
            time_points=n_times,
            num_classes=n_class,
            embed_dim=model_params["embed_dim"],
            pool_size=model_params["pool_size"],
            pool_stride=model_params["pool_stride"],
            num_heads=model_params["num_heads"],
            fc_ratio=model_params["fc_ratio"],
            depth=model_params["depth"],
            attn_drop=model_params["attn_drop"],
            fc_drop=model_params["fc_drop"],
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["train"]["lr"],
            weight_decay=config["train"].get("weight_decay", 0.0),
        )

    elif name == "Deformer":
        model = Deformer(
            num_chan=n_channels,
            num_time=n_times,
            temporal_kernel=model_params["temporal_kernel"],
            num_kernel=model_params["num_kernel"],
            num_classes=n_class,
            depth=model_params["depth"],
            heads=model_params["heads"],
            mlp_dim=model_params["mlp_dim"],
            dim_head=model_params["dim_head"],
            dropout=model_params["dropoutRate"],
        ).to(device)
        optimizer = _adam(model, config)

    else:
        raise ValueError(f"Unsupported model in CROSS_repro: {name}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Parameters: {n_params}")
    return model, criterion, optimizer
