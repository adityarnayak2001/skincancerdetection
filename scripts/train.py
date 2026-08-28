#!/usr/bin/env python3
"""Train on data-dir/train and data-dir/valid ImageFolder directories."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from skin_cancer.data import evaluation_transform, imagefolder, make_eval_loader, make_train_loader, training_transform
from skin_cancer.models import build_model
from skin_cancer.training import run_epoch, save_checkpoint, set_seed


def parse_args():
    parser = argparse.ArgumentParser(); parser.add_argument("--data-dir", type=Path, required=True); parser.add_argument("--architecture", choices=["mobilenet_v2", "resnet18"], default="mobilenet_v2"); parser.add_argument("--epochs", type=int, default=25); parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--image-size", type=int, default=224); parser.add_argument("--lr", type=float, default=3e-4); parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--patience", type=int, default=6); parser.add_argument("--workers", type=int, default=2); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--output", type=Path, default=Path("models/best_model.pth")); parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args(); set_seed(args.seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = imagefolder(args.data_dir / "train", training_transform(args.image_size)); valid_ds = imagefolder(args.data_dir / "valid", evaluation_transform(args.image_size))
    if train_ds.classes != valid_ds.classes: raise ValueError("train and valid class folders must match")
    model = build_model(args.architecture, len(train_ds.classes), not args.no_pretrained).to(device); criterion = nn.CrossEntropyLoss(label_smoothing=0.05); optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay); scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=2); scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    train_loader, valid_loader = make_train_loader(train_ds, args.batch_size, args.workers), make_eval_loader(valid_ds, args.batch_size, args.workers); best, stale = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics, _, _ = run_epoch(model, train_loader, criterion, device, optimizer, scaler); val_loss, val_metrics, _, _ = run_epoch(model, valid_loader, criterion, device); scheduler.step(val_metrics["macro_f1"])
        print(json.dumps({"epoch": epoch, "train_loss": round(train_loss, 4), "valid_loss": round(val_loss, 4), **{f"train_{k}": round(v, 4) for k, v in train_metrics.items()}, **{f"valid_{k}": round(v, 4) for k, v in val_metrics.items()}}))
        if val_metrics["macro_f1"] > best: best, stale = val_metrics["macro_f1"], 0; save_checkpoint(args.output, model, args.architecture, train_ds.classes, args.image_size, val_metrics)
        else: stale += 1
        if stale >= args.patience: print("Early stopping."); break
    print(f"Best checkpoint: {args.output} (validation macro F1: {best:.4f})")


if __name__ == "__main__": main()
