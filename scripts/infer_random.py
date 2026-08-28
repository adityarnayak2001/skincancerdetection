#!/usr/bin/env python3
"""Choose a random image, classify it, and save a Grad-CAM localization overlay."""
from __future__ import annotations
import argparse, random, sys
from pathlib import Path
import torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from skin_cancer.data import evaluation_display_transform, evaluation_transform
from skin_cancer.gradcam import GradCAM, save_overlay
from skin_cancer.models import build_model, gradcam_layer


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--image-dir", type=Path, required=True); parser.add_argument("--image", type=Path); parser.add_argument("--output", type=Path, default=Path("outputs/random_gradcam.png")); parser.add_argument("--seed", type=int, default=42); args = parser.parse_args()
    random.seed(args.seed); candidates = [p for p in args.image_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not candidates and args.image is None: raise FileNotFoundError(f"No JPG or PNG files under {args.image_dir}")
    image_path = args.image or random.choice(candidates); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False); model = build_model(checkpoint["architecture"], len(checkpoint["class_names"]), pretrained=False).to(device); model.load_state_dict(checkpoint["state_dict"])
    original = Image.open(image_path).convert("RGB"); size = checkpoint["image_size"]; tensor = evaluation_transform(size)(original).unsqueeze(0).to(device)
    cam = GradCAM(model, gradcam_layer(model, checkpoint["architecture"])); heatmap, prediction, confidence = cam.generate(tensor); cam.close()
    args.output.parent.mkdir(parents=True, exist_ok=True); save_overlay(evaluation_display_transform(size)(original), heatmap, str(args.output))
    print(f"Image: {image_path}\nPrediction: {checkpoint['class_names'][prediction]} ({confidence:.1%})\nHeatmap: {args.output}")


if __name__ == "__main__": main()
