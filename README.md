# Skin Cancer Detection

Research notebooks for skin-lesion classification and melanoma detection using the HAM10000/HMNIST datasets. The workspace contains TensorFlow/Keras and PyTorch experiments, including MobileNet-based training.

## Layout

```text
notebooks/
  pytorch_backbone_search_bat_pso.ipynb        # PyTorch backbones with Bat/PSO search
  efficientnet_metadata_attention.ipynb        # EfficientNet with metadata and attention
  utilities/                                                                 # data preparation notebooks
  archive/                                                                   # preserved historical notebook variants
data/
  raw/                                                                       # local CSV datasets (not committed)
models/                                                                      # local trained checkpoints (not committed)
```

## Getting started

1. Create and activate a Python virtual environment.
2. Install the notebook dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Keep the source CSV files under `data/raw/` and checkpoints under `models/`.
4. Open the relevant notebook with JupyterLab or VS Code and update any old Colab/Kaggle file paths before running it.

## Data

The original local data files are intentionally ignored by Git:

- `HAM10000_metadata.csv` — lesion metadata.
- `hmnist_8_8_L.csv`, `hmnist_8_8_RGB.csv` — 8x8 HMNIST pixel datasets.
- `hmnist_28_28_L.csv`, `hmnist_28_28_RGB.csv` — 28x28 HMNIST pixel datasets.

## Notes

Several historical notebooks were created in Google Colab and may import Colab/Kaggle helpers or require environment-specific packages. The versions in `notebooks/archive/` are retained as reference material; no notebook content was deleted during the cleanup.

## Improved training and testing

The legacy notebooks are retained for reference. New reusable scripts in `scripts/` replace their hard-coded paths and fixed two-class assumptions.

The expected input layout is an `ImageFolder` directory:

```text
data/images/
  train/<class-name>/*.jpg
  valid/<class-name>/*.jpg
  test/<class-name>/*.jpg
```

Train with normalized augmentation, weighted sampling for class imbalance, label smoothing, validation macro-F1 checkpoint selection, learning-rate reduction, and early stopping:

```bash
python scripts/train.py --data-dir data/images --architecture mobilenet_v2 --output models/skin_lesion_mobilenet.pth
```

Evaluate strictly on the held-out test images:

```bash
python scripts/evaluate.py --checkpoint models/skin_lesion_mobilenet.pth --test-dir data/images/test
```

Classify one random image and create a Grad-CAM localization panel. The heatmap marks image regions that most influenced the predicted class; it is an explanation aid, not a clinical lesion segmentation or diagnosis.

```bash
python scripts/infer_random.py --checkpoint models/skin_lesion_mobilenet.pth --image-dir data/images/test --output outputs/random_gradcam.png
```

## Notebook guide

- `pytorch_backbone_search_bat_pso.ipynb` — compares PyTorch MobileNetV2, InceptionV3, and ResNet18, with Bat Algorithm and Particle Swarm Optimization experiments.
- `efficientnet_metadata_attention.ipynb` — TensorFlow EfficientNet ensemble workflow that combines image features, patient metadata, attention, and progressive sprinkle augmentation.
- `utilities/image_flip_rotation_augmentation.ipynb` — creates rotated and flipped image variants.
- `utilities/keras_tensor_batch_export.ipynb` — converts dataset images to batched NumPy tensors for Keras training.
- `utilities/ham10000_binary_dataset_split.ipynb` — prepares non-melanoma examples and creates train/validation splits.
- `utilities/keras_mobilenet_pso_tuning.ipynb` — MobileNet training with PSO hyperparameter tuning.
- `archive/keras_mobilenet_256px_lr_0_001.ipynb` — 256px TensorFlow/Keras MobileNet variant using learning rate 0.001.
- `archive/keras_mobilenet_256px_lr_0_01.ipynb` — 256px TensorFlow/Keras MobileNet variant using learning rate 0.01.
- `archive/keras_mobilenet_256px_colab_cam.ipynb` — Colab-oriented 256px MobileNet variant with class-activation-map steps.
- `archive/keras_mobilenet_512px_cam_tensorboard.ipynb` — 512px Keras MobileNet experiment with CAM and TensorBoard code.
