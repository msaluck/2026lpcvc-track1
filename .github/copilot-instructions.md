# Copilot Instructions for 2026 LPCVC Track 1

This repository contains the implementation of an efficient CLIP-like model (`XRClip`) for the 2026 Low-Power Computer Vision Challenge (LPCVC) - Track 1.

## Architecture Overview

The solution `track1.py` implements a dual-encoder architecture optimized for low-power inference:

- **Image Encoder**: Uses `mobilenet_v3_large` as the backbone with a projection head to 256 dimensions.
- **Text Encoder**: A lightweight custom Transformer with a fixed vocabulary (49408) and max length (77), projecting to 256 dimensions.
- **Core Model**: `XRClip` combines both encoders. The model is designed to be exported to ONNX.

## Critical Workflows

- **Training**:
  - The training loop uses mixed precision (`torch.cuda.amp`).
  - Optimizer: `AdamW` with weight decay (default lr=3e-4).
  - Loss: Symmetric `ClipLoss` with temperature scaling (0.07).
  - Data loading relies on `RetrievalDataset` which requires image paths and text captions.

- **Export**:
  - The model **MUST** be exportable to ONNX for submission.
  - Use the `export_onnx` function in `track1.py`.
  - ONNX Opset: 17.
  - Inputs: `image` (1, 3, 224, 224), `text_input` (1, 77).
  - Outputs: `image_emb`, `text_emb` (both 256-dim).

## Coding Conventions

- **Monolithic File**: Currently, logic resides in `track1.py`. Preserve this structure unless refactoring is explicitly requested.
- **Type Shapes**: Comment tensor shapes in `forward` methods (e.g., `# (B, 960, 7, 7)`).
- **Efficiency**: Prioritize lightweight operations. The `mobilenet_v3_large` backbone is chosen for this reason. Avoid introducing heavy dependencies or operations that break ONNX compatibility.

## Specific Patterns

- **Normalization**: The model uses `F.normalize(x, dim=-1)` at the end of both encoders. Ensure this is maintained for cosine similarity.
- **Text Processing**: The `TextEncoder` uses a simple token embedding lookup + positional embedding. Ensure `input_ids` are padded/truncated to 77 tokens.
- **Device Management**: Code checks for `cuda` availability but defaults to `cpu`. Ensure tensor device movements are explicit.

## Common Tasks

- **Modify Hyperparameters**: Look for `embed_dim`, `num_heads`, `depth` in the `__init__` methods or global variables like `lr`.
- **Debug ONNX Export**: If export fails, check for dynamic control flow or unsupported operators in `forward` methods.
