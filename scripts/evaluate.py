#!/usr/bin/env python3
"""Evaluate a saved classifier on a held-out ImageFolder test set."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from skin_cancer.data import evaluation_transform, imagefolder, make_eval_loader
from skin_cancer.models import build_model
from skin_cancer.training import report, run_epoch


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--test-dir", type=Path, required=True); parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--workers", type=int, default=2); args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False); classes = checkpoint["class_names"]
    dataset = imagefolder(args.test_dir, evaluation_transform(checkpoint["image_size"]))
    if dataset.classes != classes: raise ValueError(f"Checkpoint classes {classes} do not match test classes {dataset.classes}")
    model = build_model(checkpoint["architecture"], len(classes), pretrained=False).to(device); model.load_state_dict(checkpoint["state_dict"])
    loss, metrics, labels, probabilities = run_epoch(model, make_eval_loader(dataset, args.batch_size, args.workers), nn.CrossEntropyLoss(), device)
    print(f"loss: {loss:.4f}\n{metrics}\n{report(labels, probabilities, classes)}")


if __name__ == "__main__": main()
