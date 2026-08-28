# Skin Cancer Detection

Research notebooks for skin-lesion classification and melanoma detection using the HAM10000/HMNIST datasets. The workspace contains TensorFlow/Keras and PyTorch experiments, including MobileNet-based training.

## Layout

```text
notebooks/
  utilities/                                                                 # data preparation notebooks
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
4. Open the relevant notebook with JupyterLab or VS Code.

## Data

The original local data files are intentionally ignored by Git:

- `HAM10000_metadata.csv` — lesion metadata.
- `hmnist_8_8_L.csv`, `hmnist_8_8_RGB.csv` — 8x8 HMNIST pixel datasets.
- `hmnist_28_28_L.csv`, `hmnist_28_28_RGB.csv` — 28x28 HMNIST pixel datasets.

## Improved training and testing

Reusable scripts in `scripts/` provide the supported training, evaluation, and inference workflow.

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

- `utilities/image_flip_rotation_augmentation.ipynb` — creates rotated and flipped image variants.
- `utilities/keras_tensor_batch_export.ipynb` — converts dataset images to batched NumPy tensors for Keras training.
- `utilities/ham10000_binary_dataset_split.ipynb` — prepares non-melanoma examples and creates train/validation splits.
