"""
Train InsightFace Siamese head on AT&T faces.

Usage:
    python -m face_recognition.train_att_siamese
    python -m face_recognition.train_att_siamese --config configs/insightface_siamese_att.yaml
    python -m face_recognition.train_att_siamese --att_root data/raw/faces/att_faces --epochs 100

Checkpoint is saved to <results_dir>/checkpoints/best.pth
Only the SiameseHead weights are saved (backbone is the frozen ONNX file).
"""

import os
import sys
import csv
import time
import argparse
import yaml

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Add project root to path when run as a module
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from face_recognition.insightface_siamese import InsightFaceSiamese, _BUFFALO_ONNX
from face_recognition.att_dataset import (
    load_att_subjects, extract_features, split_subjects, ATTSiamesePairDataset,
)
from losses.losses import ContrastiveLoss


# ── Defaults (overridable via YAML or CLI) ────────────────────────────────

DEFAULTS = {
    "att_root":       "data/raw/faces/att_faces",
    "embedding_dim":  128,
    "dropout":        0.3,
    "margin":         1.0,
    "epochs":         60,
    "lr":             1e-3,
    "weight_decay":   1e-4,
    "batch_size":     64,
    "patience":       15,
    "val_ratio":      0.2,
    "neg_ratio":      1.0,
    "cache_features": True,
    "results_dir":    "results/insightface_siamese_att",
}


# ── Helpers ───────────────────────────────────────────────────────────────

def accuracy_at_threshold(distances: torch.Tensor, labels: torch.Tensor, thr: float = 0.5) -> float:
    """Binary accuracy: predicted_match = (dist < thr)."""
    preds = (distances < thr).float()
    return (preds == labels).float().mean().item()


def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train(training)
    total_loss = 0.0
    total_acc  = 0.0
    n = 0

    for feat1, feat2, labels in loader:
        feat1  = feat1.to(device)
        feat2  = feat2.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(training):
            out  = model(feat1, feat2)
            loss = criterion(out["distance"], labels)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy_at_threshold(out["distance"].detach(), labels) * bs
        n += bs

    return total_loss / n, total_acc / n


# ── Main ──────────────────────────────────────────────────────────────────

def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    # ── Directories ───────────────────────────────────────────────────
    ckpt_dir = os.path.join(cfg["results_dir"], "checkpoints")
    log_dir  = os.path.join(cfg["results_dir"], "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    # ── Load AT&T ──────────────────────────────────────────────────────
    att_root = cfg["att_root"]
    subjects = load_att_subjects(att_root)
    if not subjects:
        raise FileNotFoundError(
            f"No subjects found at '{att_root}'.\n"
            f"Expected folder structure:  att_faces/s1/1.pgm  att_faces/s2/1.pgm  ..."
        )
    print(f"[ATT] {len(subjects)} subjects, "
          f"{sum(len(v) for v in subjects.values())} images")

    # ── Pre-extract backbone features ──────────────────────────────────
    cache_file = (
        os.path.join(att_root, ".feature_cache", "buffalo_l.npz")
        if cfg.get("cache_features", True)
        else None
    )
    features = extract_features(
        subjects, onnx_path=_BUFFALO_ONNX, cache_file=cache_file
    )

    # ── Subject split ──────────────────────────────────────────────────
    train_subj, val_subj = split_subjects(subjects, val_ratio=cfg["val_ratio"])
    print(f"[Split] train={len(train_subj)} subjects, val={len(val_subj)} subjects")

    # ── Datasets & loaders ─────────────────────────────────────────────
    train_ds = ATTSiamesePairDataset(train_subj, features, neg_ratio=cfg["neg_ratio"], seed=42)
    val_ds   = ATTSiamesePairDataset(val_subj,   features, neg_ratio=cfg["neg_ratio"], seed=0)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    # ── Model ──────────────────────────────────────────────────────────
    model = InsightFaceSiamese(
        embedding_dim=cfg["embedding_dim"],
        dropout=cfg["dropout"],
    ).to(device)

    # ── Loss & optimiser ───────────────────────────────────────────────
    criterion = ContrastiveLoss(margin=cfg["margin"])
    optimizer = optim.Adam(
        model.head.parameters(),        # only head params
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    # ── CSV log ────────────────────────────────────────────────────────
    log_path = os.path.join(log_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "is_best"]
        )

    # ── Training loop ──────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_ctr  = 0
    best_ckpt_path = os.path.join(ckpt_dir, "best.pth")

    print(f"\n{'='*60}")
    print(f"  InsightFace Siamese — AT&T Fine-tuning")
    print(f"  Backbone : buffalo_l / w600k_r50 (frozen)")
    print(f"  Head     : 512 → 256 → {cfg['embedding_dim']}-d  (trainable)")
    print(f"  Loss     : ContrastiveLoss (margin={cfg['margin']})")
    print(f"  Epochs   : {cfg['epochs']}  |  Patience: {cfg['patience']}")
    print(f"{'='*60}\n")

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, optimizer, device, training=False)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            patience_ctr  = 0
            torch.save(
                {
                    "epoch":              epoch,
                    "head_state_dict":    model.head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss":      best_val_loss,
                    "config": {
                        "embedding_dim": cfg["embedding_dim"],
                        "dropout":       cfg["dropout"],
                    },
                },
                best_ckpt_path,
            )
        else:
            patience_ctr += 1

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                f"{train_loss:.6f}", f"{train_acc:.4f}",
                f"{val_loss:.6f}",   f"{val_acc:.4f}",
                f"{lr:.6f}",         int(is_best),
            ])

        marker = " ← best" if is_best else ""
        print(
            f"[{epoch:3d}/{cfg['epochs']}]  "
            f"loss {train_loss:.4f}/{val_loss:.4f}  "
            f"acc {train_acc:.3f}/{val_acc:.3f}  "
            f"lr {lr:.5f}  {elapsed:.1f}s{marker}"
        )

        if patience_ctr >= cfg["patience"]:
            print(f"\n[EarlyStopping] No improvement for {cfg['patience']} epochs. Stopping.")
            break

    print(f"\n[Done] Best val_loss={best_val_loss:.6f}  checkpoint → {best_ckpt_path}")
    return best_val_loss


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train InsightFace Siamese on AT&T")
    parser.add_argument("--config",        type=str,   default=None,
                        help="YAML config file (overrides defaults)")
    parser.add_argument("--att_root",      type=str,   default=None)
    parser.add_argument("--embedding_dim", type=int,   default=None)
    parser.add_argument("--epochs",        type=int,   default=None)
    parser.add_argument("--lr",            type=float, default=None)
    parser.add_argument("--batch_size",    type=int,   default=None)
    parser.add_argument("--margin",        type=float, default=None)
    parser.add_argument("--patience",      type=int,   default=None)
    parser.add_argument("--results_dir",   type=str,   default=None)
    args = parser.parse_args()

    # Start from defaults
    cfg = dict(DEFAULTS)

    # Layer YAML on top
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg.update({k: v for k, v in yaml.safe_load(f).items() if v is not None})

    # Layer CLI on top
    for key in ["att_root", "embedding_dim", "epochs", "lr", "batch_size",
                "margin", "patience", "results_dir"]:
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val

    print("[Config]")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print()

    train(cfg)


if __name__ == "__main__":
    main()
