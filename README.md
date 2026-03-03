# XRClip - LPCVC 2026 Track 1

## Efficient CLIP for Low-Power Computer Vision

This repository implements **XRClip**, a highly optimized dual-encoder model designed for the **2026 Low-Power Computer Vision Challenge (LPCVC)**. The goal is to achieve high-performance zero-shot image retrieval while maintaining extreme efficiency for deployment on constrained edge devices.

### 🚀 Key Features

- **Lightweight Backbone**: Uses `MobileNetV3-Large` (pretrained on ImageNet-1k) for image encoding -> projected to 256 dimensions.
- **Efficient Text Encoder**: Uses a distilled `ViT-B/32` (OpenCLIP) text transformer, optimized with manual forward pass logic for ONNX compatibility.
- **Quantization Aware**: Includes `FakeQuant` modules (STE-enabled) to simulate INT8 quantization effects during training.
- **Distillation Training**: Leverages a frozen `ViT-B/32` teacher model to guide the lightweight student model.
- **ONNX Export**: Fully compatible with ONNX Opset 17 for deployment.

### 📂 Project Structure

```
├── track1.py           # Core model definition (XRClip) & ONNX export logic
├── train.py            # Main training loop with distillation & gradient accumulation
├── dataset_loader.py   # Data loading utilities (COCO, Flickr30k)
├── debug_sanity.py     # Micro-training script to verify model logic
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

### 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/2026lpcvc-track1.git
    cd 2026lpcvc-track1
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### 🏋️ Training

The training script supports both full training and quick debugging modes.

**Basic Usage:**
```bash
python train.py --epochs 20 --batch_size 64 --use_distill
```

**Key Arguments:**
- `--epochs`: Number of training epochs (default: 20).
- `--batch_size`: Batch size (default: 32).
- `--lr`: Learning rate (default: 3e-4).
- `--use_distill`: Enable knowledge distillation from a frozen CLIP teacher.
- `--accum_steps`: Gradient accumulation steps for large batch simulation.
- `--save_path`: Path to save the final model weights (default: `xr_clip_s256.pth`).

### 🧪 Debugging & Sanity Checks

To verify the model architecture and training pipeline without waiting for a full epoch, run the sanity check script:

```bash
python debug_sanity.py
```
*This trains the model on 4 synthetic samples (Red, Blue, Green, Black) to 100% accuracy, confirming that gradients flow correctly through all components.*

### 📦 ONNX Export

To export the trained model for submission/deployment:

```python
from track1 import XRClip, export_onnx
import torch

model = XRClip()
model.load_state_dict(torch.load("xr_clip_s256.pth"))
export_onnx(model, "submission_model.onnx")
```

### 📊 Performance

| Metric | Value |
| :--- | :--- |
| **Backbone** | MobileNetV3-Large |
| **Embedding Dim** | 256 |
| **Input Resolution** | 224x224 |
| **Text Context Length** | 77 tokens |
| **Params (Image)** | ~4.2M |
| **Params (Text)** | ~63M (Transformer) |

---
**License**: MIT
