import os
import json
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


class TextCalibrationBank:
    def __init__(
        self,
        class_feature_path: str,
        class_order_path: str,
        class_text_json_path: str,
        frame_feature_root: str,
        alpha: float,
        text_temp: float,
        strict_length: bool,
        device: torch.device,
    ):
        self.device = device
        self.alpha = float(alpha)
        self.text_temp = float(text_temp)
        self.strict_length = bool(strict_length)
        self.frame_feature_root = frame_feature_root

        class_feats = np.load(class_feature_path)
        with open(class_order_path, "r", encoding="utf-8") as f:
            self.class_order = json.load(f)

        self.class_features = torch.as_tensor(class_feats, dtype=torch.float32, device=device)
        self.class_features = F.normalize(self.class_features, dim=-1)

        self.valid_text_mask = torch.ones(len(self.class_order), dtype=torch.bool, device=device)

        if class_text_json_path and os.path.exists(class_text_json_path):
            with open(class_text_json_path, "r", encoding="utf-8") as f:
                items = json.load(f)
            if len(items) == len(self.class_order):
                mask = []
                for item in items:
                    txt = str(item.get("text", "")).strip()
                    mask.append(len(txt) > 0)
                self.valid_text_mask = torch.as_tensor(mask, dtype=torch.bool, device=device)

        self._frame_cache = {}

    def _load_frame_features(self, session: str) -> torch.Tensor:
        if session in self._frame_cache:
            return self._frame_cache[session]

        path = os.path.join(self.frame_feature_root, f"{session}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Frame text feature file not found: {path}")

        feats = np.load(path)
        feats = torch.as_tensor(feats, dtype=torch.float32, device=self.device)
        feats = F.normalize(feats, dim=-1)

        self._frame_cache[session] = feats
        return feats

    def _to_index_tensor(self, query_indices):
        if isinstance(query_indices, torch.Tensor):
            return query_indices.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(np.asarray(query_indices), dtype=torch.long, device=self.device)

    def calibrate_probs(self, session: str, vis_probs: torch.Tensor, query_indices=None) -> torch.Tensor:
        """
        vis_probs: [T, C], already softmax probabilities for the current query window
        query_indices: indices of this query window within the full video timeline
        returns: [T, C]
        """
        frame_feats_full = self._load_frame_features(session)

        if query_indices is not None:
            idx = self._to_index_tensor(query_indices)

            if idx.numel() != vis_probs.shape[0]:
                raise ValueError(
                    f"query_indices/visual length mismatch for {session}: "
                    f"indices={idx.numel()} vs vis_probs={vis_probs.shape[0]}"
                )

            if idx.max().item() >= frame_feats_full.shape[0]:
                raise ValueError(
                    f"query index out of range for {session}: "
                    f"max_idx={idx.max().item()} vs frame_text_len={frame_feats_full.shape[0]}"
                )

            frame_feats = frame_feats_full.index_select(0, idx)
        else:
            frame_feats = frame_feats_full
            if frame_feats.shape[0] != vis_probs.shape[0]:
                if self.strict_length:
                    raise ValueError(
                        f"text/visual length mismatch for {session}: "
                        f"frame_text={frame_feats.shape[0]} vs vis_probs={vis_probs.shape[0]}"
                    )
                else:
                    t = min(frame_feats.shape[0], vis_probs.shape[0])
                    frame_feats = frame_feats[:t]
                    vis_probs = vis_probs[:t]

        text_logits = frame_feats @ self.class_features.t()   # [T, C]
        text_logits = text_logits / self.text_temp

        invalid_mask = ~self.valid_text_mask
        if invalid_mask.any():
            text_logits[:, invalid_mask] = -1e4

        text_probs = torch.softmax(text_logits, dim=-1)
        mixed = (1.0 - self.alpha) * vis_probs + self.alpha * text_probs
        mixed = mixed / mixed.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return mixed


def build_text_calibrator(cfg, device: torch.device) -> Optional[TextCalibrationBank]:
    if not cfg.MODEL.TEXT_CALIBRATION.ENABLED:
        return None

    return TextCalibrationBank(
        class_feature_path=cfg.MODEL.TEXT_CALIBRATION.CLASS_FEATURE_PATH,
        class_order_path=cfg.MODEL.TEXT_CALIBRATION.CLASS_ORDER_PATH,
        class_text_json_path=cfg.MODEL.TEXT_CALIBRATION.CLASS_TEXT_JSON_PATH,
        frame_feature_root=cfg.MODEL.TEXT_CALIBRATION.FRAME_FEATURE_ROOT,
        alpha=cfg.MODEL.TEXT_CALIBRATION.ALPHA,
        text_temp=cfg.MODEL.TEXT_CALIBRATION.TEXT_TEMP,
        strict_length=cfg.MODEL.TEXT_CALIBRATION.STRICT_LENGTH,
        device=device,
    )