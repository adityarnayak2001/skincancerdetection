from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score, roc_auc_score


def set_seed(seed: int) -> None:
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.backends.cudnn.benchmark = True


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None; model.train(training)
    losses, predictions, labels, probabilities = [], [], [], []
    for images, target in loader:
        images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            logits, loss = model(images), None
            loss = criterion(logits, target)
        if training: scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        losses.append(loss.detach().item() * len(target)); predictions.extend(logits.argmax(1).detach().cpu().tolist()); labels.extend(target.detach().cpu().tolist()); probabilities.extend(logits.softmax(1).detach().cpu().tolist())
    y_true, y_pred, y_prob = np.array(labels), np.array(predictions), np.array(probabilities)
    metrics = {"accuracy": accuracy_score(y_true, y_pred), "balanced_accuracy": balanced_accuracy_score(y_true, y_pred), "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0)}
    if len(np.unique(y_true)) == 2: metrics["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
    return sum(losses) / len(y_true), metrics, y_true, y_prob


def save_checkpoint(path: Path, model, architecture: str, class_names: list[str], image_size: int, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"architecture": architecture, "class_names": class_names, "image_size": image_size, "metrics": metrics, "state_dict": model.state_dict()}, path)


def report(y_true: np.ndarray, probabilities: np.ndarray, class_names: list[str]) -> str:
    return classification_report(y_true, probabilities.argmax(axis=1), target_names=class_names, digits=4, zero_division=0)
