from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model, self.activations, self.gradients = model, None, None
        self.handle = target_layer.register_forward_hook(self._save_activations)

    def _save_activations(self, _module, _inputs, output):
        self.activations = output; output.register_hook(self._save_gradients)

    def _save_gradients(self, gradients): self.gradients = gradients

    def generate(self, image: torch.Tensor, class_index: int | None = None) -> tuple[np.ndarray, int, float]:
        self.model.eval(); logits = self.model(image); probabilities = logits.softmax(dim=1)
        predicted = int(probabilities.argmax(dim=1).item()) if class_index is None else class_index
        self.model.zero_grad(set_to_none=True); logits[:, predicted].sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = F.interpolate((weights * self.activations).sum(dim=1, keepdim=True).relu(), size=image.shape[-2:], mode="bilinear", align_corners=False).squeeze().detach().cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap, predicted, float(probabilities[0, predicted].item())

    def close(self): self.handle.remove()


def save_overlay(image: Image.Image, heatmap: np.ndarray, output_path: str, alpha: float = 0.42) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5)); image_array = np.asarray(image.convert("RGB"))
    axes[0].imshow(image_array); axes[0].set_title("Input")
    axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Grad-CAM")
    axes[2].imshow(image_array); axes[2].imshow(heatmap, cmap="jet", alpha=alpha); axes[2].set_title("Localization overlay")
    for axis in axes: axis.axis("off")
    fig.tight_layout(); fig.savefig(output_path, dpi=180, bbox_inches="tight"); plt.close(fig)
