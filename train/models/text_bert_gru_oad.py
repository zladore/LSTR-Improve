#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Any

import torch
import torch.nn as nn


class TextBertGRUOAD(nn.Module):
    """
    Text-only OAD model:
        [B, K, D_text] -> proj -> GRU -> classifier

    Default:
        D_text = 768
        proj_dim = 512
        hidden_dim = 512
        num_layers = 1
        num_classes = 22
    """

    def __init__(
        self,
        text_dim: int = 768,
        proj_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.3,
        num_classes: int = 22,
    ) -> None:
        super().__init__()

        self.text_dim = text_dim
        self.proj_dim = proj_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.num_classes = num_classes

        self.input_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=False,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)

    def forward(self, text_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_seq: [B, K, D]
        Returns:
            logits: [B, num_classes]
        """
        if text_seq.ndim != 3:
            raise ValueError(f"text_seq must be [B, K, D], got shape={tuple(text_seq.shape)}")

        x = self.input_proj(text_seq)      # [B, K, proj_dim]
        out, h_n = self.gru(x)             # out: [B, K, H], h_n: [num_layers, B, H]
        last_hidden = h_n[-1]              # [B, H]
        logits = self.head(last_hidden)    # [B, C]
        return logits

    @torch.no_grad()
    def infer(self, text_seq: torch.Tensor) -> Dict[str, Any]:
        logits = self.forward(text_seq)
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1)
        conf = torch.max(probs, dim=-1).values
        return {
            "logits": logits,
            "probs": probs,
            "pred": pred,
            "conf": conf,
        }