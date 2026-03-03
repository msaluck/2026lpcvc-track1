import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer
import open_clip
import argparse
import os
from dataset_loader import load_coco_captions, load_flickr30k

torch.backends.cudnn.benchmark = True

from tqdm import tqdm

from track1 import (
    XRClip,
    RetrievalDataset,
    ClipLoss,
    distillation_loss
)

# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--use_distill", action="store_true")
parser.add_argument("--save_path", type=str, default="xr_clip_s256.pth")
parser.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps")

args = parser.parse_args()

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# TOKENIZER
# ============================================================

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
tokenizer.add_special_tokens({'cls_token': tokenizer.eos_token})

# ============================================================
# DATASET (REPLACE WITH YOUR REAL DATA)
# ============================================================

# TODO: Replace with your actual dataset paths
# image_paths = ["image1.jpg", "image2.jpg"]
# texts = ["a red bus", "a blue bus"]

# --------------------------
# LOAD COCO
# --------------------------
coco_images, coco_texts = load_coco_captions(
    image_root="datasets/train2017",
    annotation_file="datasets/annotations/captions_train2017.json"
)

coco_images_all, coco_texts_all = load_coco_captions(
    image_root="datasets/train2017",
    annotation_file="datasets/annotations/captions_train2017.json"
)

# Keep only first caption per image
unique = {}
coco_images = []
coco_texts = []

for img, txt in zip(coco_images_all, coco_texts_all):
    if img not in unique:
        unique[img] = True
        coco_images.append(img)
        coco_texts.append(txt)

# --------------------------
# LOAD FLICKR
# --------------------------
flickr_images, flickr_texts = load_flickr30k()

# --------------------------
# MERGE
# --------------------------
image_paths = coco_images + flickr_images
texts = coco_texts + flickr_texts
# image_paths = coco_images
# texts = coco_texts

print("Total samples:", len(image_paths))

# temporary subset for faster training during development (remove for full training)
OVERFIT_SIZE = 120
image_paths = image_paths[:OVERFIT_SIZE]
texts = texts[:OVERFIT_SIZE]

# image_paths, texts = load_coco_captions(
#     image_root="datasets/train2017",
#     annotation_file="datasets/annotations/captions_train2017.json"
# )

dataset = RetrievalDataset(image_paths, texts, tokenizer)

# Semantic batch sampler groups similar samples together to improve convergence
# sampler = SemanticBatchSampler(dataset, batch_size=args.batch_size)

# dataloader = DataLoader(
#     dataset,
#     batch_sampler=sampler,
#     num_workers=8,     # Increase for 20-core CPU
#     pin_memory=True,
#     persistent_workers=True, # Prevent worker respawn overhead
#     prefetch_factor=2  # Buffer batches ahead of time
# )

# normal random batching (no semantic grouping)
dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,       # Shuffle ON for training
    num_workers=0,      # Simple
    pin_memory=False
)

# --------------------------
# LOAD VALIDATION (COCO VAL)
# --------------------------

# IMPORTANT: For overfitting test, validation MUST be the SAME as training
val_images, val_texts = image_paths, texts
print(f"Validation set (OVERFIT): {len(val_images)}")

val_dataset = RetrievalDataset(val_images, val_texts, tokenizer)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

# ============================================================
# MODEL
# ============================================================

# model = XRClip(embed_dim=256).to(device)
# criterion = ClipLoss()

# optimizer = torch.optim.AdamW(
#     model.parameters(),
#     lr=args.lr,
#     weight_decay=1e-4
# )

# ============================================================
# MODEL
# ============================================================

model = XRClip(embed_dim=256).to(device)
criterion = ClipLoss()

# # 🔒 Freeze text encoder (Phase 1 debugging)
# for p in model.text_encoder.parameters():
#     p.requires_grad = False

# print("Text encoder frozen (Phase 1)")

# optimizer = torch.optim.AdamW(
#     filter(lambda p: p.requires_grad, model.parameters()),
#     lr=args.lr,
#     weight_decay=1e-4
# )

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    weight_decay=1e-4
)

# ============================================================
# OPTIONAL DISTILLATION
# ============================================================

if args.use_distill:
    print("Distillation enabled")

    teacher_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
        force_quick_gelu=True
    )

    teacher_model.to(device)
    teacher_model.eval()
    teacher_model.requires_grad_(False)

    teacher_proj = nn.Linear(512, 256).to(device)
else:
    # Use distillation by default if not specified? Or just remove this else
    # Actually, let's DISABLE distillation for now to debug convergence
    print("Distillation DISABLED for debugging (Standard CLIP Loss only)")
    args.use_distill = False
    
    # teacher_model, _, _ = open_clip.create_model_and_transforms(
    #     "ViT-B-32",
    #     pretrained="openai"
    # )
    # ...
# else:
#     # Use distillation by default if not specified? Or just remove this else
#     # Actually, let's FORCE distillation for now to debug convergence
#     print("Distillation FORCED (Recommended for training from scratch)")
#     args.use_distill = True
    
#     teacher_model, _, _ = open_clip.create_model_and_transforms(
#         "ViT-B-32",
#         pretrained="openai"
#     )

#     teacher_model.to(device)
#     teacher_model.eval()
#     teacher_model.requires_grad_(False)

#     teacher_proj = nn.Linear(512, 256).to(device)

# ============================================================
# RECALL@10 EVALUATION FUNCTION
# ============================================================

def evaluate_recall(model, dataloader, device):

    model.eval()

    all_image_emb = []
    all_text_emb = []

    with torch.no_grad():
        for images, text_inputs in dataloader:

            images = images.to(device)
            text_inputs = text_inputs.to(device)

            with torch.amp.autocast('cuda'):
                image_emb, text_emb = model(images, text_inputs)

            all_image_emb.append(image_emb)
            all_text_emb.append(text_emb)

    image_emb = torch.cat(all_image_emb)
    text_emb = torch.cat(all_text_emb)

    # Similarity matrix
    sims = image_emb @ text_emb.T

    ranks = sims.argsort(dim=-1, descending=True)

    correct = torch.arange(len(image_emb), device=image_emb.device).unsqueeze(1)

    recall = (ranks[:, :10] == correct).any(dim=1).float().mean()

    return recall.item()

# ============================================================
# TRAINING LOOP
# ============================================================

scaler = torch.amp.GradScaler('cuda')
best_recall = 0

for epoch in range(args.epochs):

    model.train()
    total_loss = 0

    optimizer.zero_grad(set_to_none=True)

    loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{args.epochs}]")
    for step, (images, text_inputs) in enumerate(loop):

        images = images.to(device, non_blocking=True)
        text_inputs = text_inputs.to(device, non_blocking=True)

        # Do NOT zero_grad here, do it after step

        with torch.amp.autocast('cuda'):

            image_emb, text_emb = model(images, text_inputs)
            # loss = criterion(image_emb, text_emb)
            loss = criterion(image_emb, text_emb, model.logit_scale)

            # Distillation (optional)
            if args.use_distill:
                with torch.no_grad():
                    teacher_img = teacher_model.encode_image(images)
                    teacher_txt = teacher_model.encode_text(text_inputs)

                teacher_img = teacher_proj(teacher_img)
                teacher_txt = teacher_proj(teacher_txt)

                img_distill = distillation_loss(image_emb, teacher_img)
                txt_distill = distillation_loss(text_emb, teacher_txt)

                # Heavy distillation (90%) for faster convergence
                loss = 0.1 * loss + 0.45 * img_distill + 0.45 * txt_distill

        # Gradient Accumulation
        loss = loss / args.accum_steps
        scaler.scale(loss).backward()

        if (step + 1) % args.accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * args.accum_steps # Scale back for logging
        loop.set_postfix(loss=loss.item() * args.accum_steps)

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {avg_loss:.4f}")

    # 🔍 Quick Sanity Check
    print("Current logit_scale (exp):", model.logit_scale.exp().item())

    # Evaluate recall on validation set
    val_recall = evaluate_recall(model, val_loader, device)
    print(f"Validation Recall@10: {val_recall:.4f}")

    if val_recall > best_recall:
        best_recall = val_recall
        torch.save(model.state_dict(), "best_model.pth")
        print(f"New best recall: {best_recall:.4f} (model saved)")

# ============================================================
# SAVE MODEL
# ============================================================

torch.save(model.state_dict(), args.save_path)
print("Model saved to:", args.save_path)