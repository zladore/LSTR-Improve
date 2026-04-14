#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------
# import local modules
# ---------------------------------------------------------------------
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_DATASET_DIR = os.path.join(_CUR_DIR, "datasets")
_MODEL_DIR = os.path.join(_CUR_DIR, "models")
sys.path.append(_DATASET_DIR)
sys.path.append(_MODEL_DIR)

from datasets.temporal_text_dataset_lstr import TemporalTextDatasetLSTR
from models.text_bert_gru_oad import TextBertGRUOAD


# ---------------------------------------------------------------------
# THUMOS official-style eval (aligned with your uploaded official files)
# ---------------------------------------------------------------------
THUMOS_CLASS_NAMES = [
    "Background",
    "BaseballPitch",
    "BasketballDunk",
    "Billiards",
    "CleanAndJerk",
    "CliffDiving",
    "CricketBowling",
    "CricketShot",
    "Diving",
    "FrisbeeCatch",
    "GolfSwing",
    "HammerThrow",
    "HighJump",
    "JavelinThrow",
    "LongJump",
    "PoleVault",
    "Shotput",
    "SoccerPenalty",
    "TennisSwing",
    "ThrowDiscus",
    "VolleyballSpiking",
    "Ambiguous",
]


def calibrated_average_precision_score(y_true, y_score):
    y_true_sorted = y_true[np.argsort(-y_score)]
    tp = y_true_sorted.astype(float)
    fp = np.abs(y_true_sorted.astype(float) - 1)
    tps = np.cumsum(tp)
    fps = np.cumsum(fp)
    ratio = np.sum(tp == 0) / np.sum(tp)
    cprec = tps / (tps + fps / (ratio + np.finfo(float).eps) + np.finfo(float).eps)
    cap = np.sum(cprec[tp == 1]) / np.sum(tp)
    return cap


def thumos_postprocessing(ground_truth, prediction, smooth=False, switch=False):
    """
    Official-style THUMOS postprocessing.
    - optional temporal smoothing
    - optional CliffDiving->Diving switch
    - remove ambiguous (class 21)
    """
    if smooth:
        prob = np.copy(prediction)
        prob1 = prob.reshape(1, prob.shape[0], prob.shape[1])
        prob2 = np.append(prob[0, :].reshape(1, -1), prob[0:-1, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
        prob3 = np.append(prob[1:, :], prob[-1, :].reshape(1, -1), axis=0).reshape(1, prob.shape[0], prob.shape[1])
        prob4 = np.append(prob[0:2, :], prob[0:-2, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
        prob5 = np.append(prob[2:, :], prob[-2:, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
        probsmooth = np.squeeze(np.max(np.concatenate((prob1, prob2, prob3, prob4, prob5), axis=0), axis=0))
        prediction = np.copy(probsmooth)

    if switch:
        switch_index = np.where(prediction[:, 5] > prediction[:, 8])[0]
        prediction[switch_index, 8] = prediction[switch_index, 5]

    valid_index = np.where(ground_truth[:, 21] != 1)[0]
    return ground_truth[valid_index], prediction[valid_index]


def perframe_average_precision(
    ground_truth,
    prediction,
    class_names,
    ignore_index,
    metrics="AP",
    smooth=False,
    switch=False,
):
    result = {}
    ground_truth = np.array(ground_truth)
    prediction = np.array(prediction)

    ground_truth, prediction = thumos_postprocessing(
        ground_truth, prediction, smooth=smooth, switch=switch
    )

    if metrics == "AP":
        compute_score = average_precision_score
    elif metrics == "cAP":
        compute_score = calibrated_average_precision_score
    else:
        raise RuntimeError(f"Unknown metrics: {metrics}")

    ignore_index = set([0, ignore_index])  # official logic: ignore background=0 and ignore_index
    result["per_class_AP"] = {}

    for idx, class_name in enumerate(class_names):
        if idx not in ignore_index:
            if np.any(ground_truth[:, idx]):
                result["per_class_AP"][class_name] = compute_score(
                    ground_truth[:, idx], prediction[:, idx]
                )

    result["mean_AP"] = float(np.mean(list(result["per_class_AP"].values())))
    return result


def compute_official_thumos_map(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ignore_index: int = 21,
    metrics: str = "AP",
    smooth: bool = False,
    switch: bool = False,
) -> Tuple[float, Dict[str, float]]:
    """
    y_true: [N] integer labels in [0, 21]
    y_prob: [N, 22] probabilities or scores
    """
    ground_truth = np.zeros((len(y_true), 22), dtype=np.int32)
    valid = (y_true >= 0) & (y_true < 22)
    ground_truth[np.arange(len(y_true))[valid], y_true[valid]] = 1

    result = perframe_average_precision(
        ground_truth=ground_truth,
        prediction=y_prob,
        class_names=THUMOS_CLASS_NAMES,
        ignore_index=ignore_index,
        metrics=metrics,
        smooth=smooth,
        switch=switch,
    )
    return result["mean_AP"], result["per_class_AP"]


# ---------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_warmup_cosine_lambda(
    total_epochs: int,
    warmup_epochs: int,
):
    def lr_lambda(epoch: int) -> float:
        if total_epochs <= 0:
            return 1.0
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    return x


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    test_loss: float
    test_acc: float
    test_macro_f1: float
    official_map: float


# ---------------------------------------------------------------------
# train / eval
# ---------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    criterion: nn.Module,
    use_amp: bool,
    grad_clip: float,
    ignore_index: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        text_seq, label = batch[:2]
        text_seq = to_device(text_seq, device)
        label = to_device(label, device)

        valid_mask = label != ignore_index
        if valid_mask.sum().item() == 0:
            continue

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                logits = model(text_seq)
                loss = criterion(logits[valid_mask], label[valid_mask])
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(text_seq)
            loss = criterion(logits[valid_mask], label[valid_mask])
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = int(valid_mask.sum().item())
        total_loss += loss.item() * bs
        total_count += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(1, total_count)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    ignore_index: int,
    num_classes: int,
) -> Tuple[float, float, float, float, Dict[str, float], List[Dict], np.ndarray, np.ndarray]:
    model.eval()

    total_loss = 0.0
    total_count = 0

    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []
    all_rows: List[Dict] = []

    pbar = tqdm(loader, desc="Eval", leave=False)
    for batch in pbar:
        text_seq, label, meta = batch
        text_seq = to_device(text_seq, device)
        label = to_device(label, device)

        logits = model(text_seq)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        valid_mask = label != ignore_index
        if valid_mask.sum().item() > 0:
            loss = criterion(logits[valid_mask], label[valid_mask])
            bs = int(valid_mask.sum().item())
            total_loss += loss.item() * bs
            total_count += bs

        probs_np = probs.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()
        labels_np = label.detach().cpu().numpy()

        all_labels.extend(labels_np.tolist())
        all_preds.extend(preds_np.tolist())
        all_probs.append(probs_np)

        batch_size = len(labels_np)
        for i in range(batch_size):
            row = {
                "sample_id": int(meta["sample_id"][i]),
                "video_id": meta["video_id"][i],
                "frame_index_in_video": int(meta["frame_index_in_video"][i]),
                "seq_idx": int(meta["seq_idx"][i]),
                "split": meta["split"][i],
                "gt_label": meta["gt_label"][i],
                "target": int(meta["gt_label_id"][i]),        # official eval csv expects target
                "pred_label_id": int(preds_np[i]),
                "is_background": int(meta["is_background"][i]),
                "current_feature_index": int(meta["current_feature_index"][i]),
            }
            for c in range(num_classes):
                row[f"score_cls_{c}"] = float(probs_np[i, c])  # official eval csv expects score_cls_i
            all_rows.append(row)

    y_true = np.asarray(all_labels, dtype=np.int64)
    y_pred = np.asarray(all_preds, dtype=np.int64)
    y_prob = np.concatenate(all_probs, axis=0) if len(all_probs) > 0 else np.zeros((0, num_classes), dtype=np.float32)

    valid_mask = y_true != ignore_index
    if valid_mask.sum() > 0:
        acc = accuracy_score(y_true[valid_mask], y_pred[valid_mask])
        macro_f1 = f1_score(y_true[valid_mask], y_pred[valid_mask], average="macro")
    else:
        acc = float("nan")
        macro_f1 = float("nan")

    official_map, per_class_ap = compute_official_thumos_map(
        y_true=y_true,
        y_prob=y_prob,
        ignore_index=ignore_index,
        metrics="AP",
        smooth=False,
        switch=False,
    )

    mean_loss = total_loss / max(1, total_count)
    return mean_loss, acc, macro_f1, official_map, per_class_ap, all_rows, y_true, y_prob


def save_checkpoint(path: str, model: nn.Module, optimizer, scheduler, epoch: int, args_dict: Dict):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "args": args_dict,
    }
    torch.save(ckpt, path)


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--temporal_index_csv",
        type=str,
        default="/home/hbxz_lzl/pro1_baseline/valText/data/output/bert_text_strict/temporal_text_index_k8.csv",
    )
    parser.add_argument(
        "--text_feature_npy",
        type=str,
        default="/home/hbxz_lzl/pro1_baseline/valText/data/output/bert_text_strict/bert_text_features.npy",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/hbxz_lzl/pro1_baseline/valText/train/outputs/text_bert_gru_k8",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--use_amp", action="store_true")

    parser.add_argument("--text_dim", type=int, default=768)
    parser.add_argument("--proj_dim", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--num_classes", type=int, default=22)
    parser.add_argument("--background_idx", type=int, default=0)   # fixed to official THUMOS order
    parser.add_argument("--ignore_index", type=int, default=21)    # Ambiguous

    args = parser.parse_args()
    ensure_dir(args.output_dir)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("[INFO] preparing datasets...")
    train_ds = TemporalTextDatasetLSTR(
        temporal_index_csv=args.temporal_index_csv,
        text_feature_npy=args.text_feature_npy,
        split="train",
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        return_meta=False,
    )
    test_ds = TemporalTextDatasetLSTR(
        temporal_index_csv=args.temporal_index_csv,
        text_feature_npy=args.text_feature_npy,
        split="test",
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        return_meta=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    print("[INFO] building model...")
    model = TextBertGRUOAD(
        text_dim=args.text_dim,
        proj_dim=args.proj_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_classes=args.num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=build_warmup_cosine_lambda(
            total_epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
        ),
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(args.use_amp and device.type == "cuda"))

    print("=" * 80)
    print("[INFO] training config")
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))
    print(f"device = {device}")
    print("=" * 80)

    train_log_csv = os.path.join(args.output_dir, "train_log.csv")
    results: List[EpochResult] = []

    best_official_map = -1.0
    best_ckpt_path = os.path.join(args.output_dir, "best_model.pt")
    final_ckpt_path = os.path.join(args.output_dir, "final_model.pt")

    for epoch in range(args.epochs):
        print(f"\n[Epoch {epoch + 1}/{args.epochs}]")

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            criterion=criterion,
            use_amp=args.use_amp,
            grad_clip=args.grad_clip,
            ignore_index=args.ignore_index,
        )

        test_loss, test_acc, test_macro_f1, official_map, _, _, _, _ = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            criterion=criterion,
            ignore_index=args.ignore_index,
            num_classes=args.num_classes,
        )

        scheduler.step()

        res = EpochResult(
            epoch=epoch + 1,
            train_loss=float(train_loss),
            test_loss=float(test_loss),
            test_acc=float(test_acc),
            test_macro_f1=float(test_macro_f1),
            official_map=float(official_map),
        )
        results.append(res)

        print(
            f"[Epoch {epoch + 1}] "
            f"train_loss={train_loss:.6f} | "
            f"test_loss={test_loss:.6f} | "
            f"acc={test_acc:.6f} | "
            f"macro_f1={test_macro_f1:.6f} | "
            f"official_mAP={official_map:.6f}"
        )

        if np.isfinite(official_map) and official_map > best_official_map:
            best_official_map = official_map
            save_checkpoint(
                path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                args_dict=vars(args),
            )
            print(f"[INFO] saved best checkpoint -> {best_ckpt_path}")

        with open(train_log_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["epoch", "train_loss", "test_loss", "test_acc", "test_macro_f1", "official_map"],
            )
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))

    save_checkpoint(
        path=final_ckpt_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=args.epochs,
        args_dict=vars(args),
    )
    print(f"[INFO] saved final checkpoint -> {final_ckpt_path}")

        # ------------------------------------------------------------
    # export FINAL checkpoint predictions
    # ------------------------------------------------------------
    print("\n[INFO] running final-checkpoint evaluation/export...")
    final_test_loss, final_test_acc, final_test_macro_f1, final_official_map, final_per_class_ap, final_rows, final_y_true, final_y_prob = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        criterion=criterion,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    final_pred_csv = os.path.join(args.output_dir, "final_test_predictions.csv")
    final_pred_df = pd.DataFrame(final_rows)
    final_pred_df.to_csv(final_pred_csv, index=False, encoding="utf-8")
    print(f"[INFO] saved FINAL test predictions to: {final_pred_csv}")

    # ------------------------------------------------------------
    # load BEST checkpoint and export BEST predictions
    # ------------------------------------------------------------
    print("\n[INFO] loading best checkpoint for official export...")
    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model"], strict=True)

    best_test_loss, best_test_acc, best_test_macro_f1, best_official_map, best_per_class_ap, best_rows, best_y_true, best_y_prob = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        criterion=criterion,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    best_pred_csv = os.path.join(args.output_dir, "best_test_predictions.csv")
    best_pred_df = pd.DataFrame(best_rows)
    best_pred_df.to_csv(best_pred_csv, index=False, encoding="utf-8")
    print(f"[INFO] saved BEST test predictions to: {best_pred_csv}")

    result_json = os.path.join(args.output_dir, "text_bert_gru_result.json")
    result = {
        "train_log_csv": train_log_csv,
        "best_ckpt_path": best_ckpt_path,
        "final_ckpt_path": final_ckpt_path,

        "final_test_pred_csv": final_pred_csv,
        "best_test_pred_csv": best_pred_csv,

        "final_test_loss": float(final_test_loss),
        "final_test_acc": float(final_test_acc),
        "final_test_macro_f1": float(final_test_macro_f1),
        "final_official_map": float(final_official_map),
        "final_per_class_ap": final_per_class_ap,

        "best_test_loss": float(best_test_loss),
        "best_test_acc": float(best_test_acc),
        "best_test_macro_f1": float(best_test_macro_f1),
        "best_official_map": float(best_official_map),
        "best_per_class_ap": best_per_class_ap,

        "num_test_samples": int(len(best_y_true)),
        "eval_note": "THUMOS official-style frame mAP with background=0 ignored, ambiguous=21 removed by postprocessing",
    }
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[INFO] saved result json to: {result_json}")

    print("\n[ALL DONE]")
    print(json.dumps({
        "best_official_map": result["best_official_map"],
        "final_official_map": result["final_official_map"],
        "best_test_pred_csv": result["best_test_pred_csv"],
        "final_test_pred_csv": result["final_test_pred_csv"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
