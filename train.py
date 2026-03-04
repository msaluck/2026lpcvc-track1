import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer
import open_clip
import argparse
import os
import logging
from dataset_loader import load_coco_captions, load_flickr30k

# Suppress annoying inductor/triton warnings
logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)
torch.backends.cudnn.benchmark = True
# TF32 on Ampere GPUs (A100/A10/3090)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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
# Resume training from checkpoint
parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
parser.add_argument("--val_every", type=int, default=1, help="Validation frequency (epochs)")
parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (validation checks)")

# Dataset paths
parser.add_argument("--coco_img_path", type=str, default="datasets/train2017", help="Path to COCO images")
parser.add_argument("--coco_ann_path", type=str, default="datasets/annotations/captions_train2017.json", help="Path to COCO annotations")
parser.add_argument("--flickr_root", type=str, default="datasets", help="Root directory for Flickr images/cache")

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
coco_images_all, coco_texts_all = load_coco_captions(
    image_root=args.coco_img_path,
    annotation_file=args.coco_ann_path
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
flickr_images, flickr_texts = load_flickr30k(root_dir=args.flickr_root)

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
# image_paths = image_paths[:OVERFIT_SIZE]
# texts = texts[:OVERFIT_SIZE]

# ============================================================
# TRAIN & VALIDATION SPLIT
# ============================================================

import random
random.seed(42)

# Shuffle and split 90/10
combined = list(zip(image_paths, texts))
random.shuffle(combined)
image_paths, texts = zip(*combined)

split_idx = int(0.9 * len(image_paths))

train_images = image_paths[:split_idx]
train_texts = texts[:split_idx]

val_images = image_paths[split_idx:]
val_texts = texts[split_idx:]

print(f"Training samples: {len(train_images)}")
print(f"Validation samples: {len(val_images)}")

dataset = RetrievalDataset(train_images, train_texts, tokenizer)

# normal random batching (no semantic grouping)
dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,       # Shuffle ON for training
    num_workers=2,      # 2 workers is safe on Colab
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True
)

# --------------------------
# LOAD VALIDATION (COCO VAL)
# --------------------------

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

model = XRClip(embed_dim=256).to(device)

if torch.cuda.get_device_capability()[0] >= 7:
    print("Compiling model with torch.compile...")
    # model = torch.compile(model) # DISABLED: Compilation overhead may be too high for frequent small batches or debug runs.
    # Re-enable for long production runs if memory allows.

criterion = ClipLoss()

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
    
    # Compile teacher for faster inference
    if torch.cuda.get_device_capability()[0] >= 7:
        teacher_model = torch.compile(teacher_model)

    teacher_proj = nn.Linear(512, 256).to(device)
else:
    # Use distillation by default if not specified? Or just remove this else
    # Actually, let's DISABLE distillation for now to debug convergence
    print("Distillation DISABLED for debugging (Standard CLIP Loss only)")
    args.use_distill = False

# ============================================================
# RECALL@10 EVALUATION FUNCTION
# ============================================================

def evaluate_recall(model, dataloader, device):

    model.eval()

    all_image_emb = []
    all_text_emb = []

    print("Collecting validation embeddings...")
    with torch.no_grad():
        for images, text_inputs in tqdm(dataloader, desc="Inference"):

            images = images.to(device)
            text_inputs = text_inputs.to(device)

            with torch.amp.autocast('cuda'):
                image_emb, text_emb = model(images, text_inputs)

            all_image_emb.append(image_emb.cpu())
            all_text_emb.append(text_emb.cpu())

    image_emb = torch.cat(all_image_emb)
    text_emb = torch.cat(all_text_emb)
    
    del all_image_emb, all_text_emb
    torch.cuda.empty_cache()

    # Move to device for matrix ops
    image_emb = image_emb.to(device)
    text_emb = text_emb.to(device)

    num_samples = len(image_emb)
    
    # Batch size for similarity calculation to avoid OOM
    # Matrix size: batch_size * num_samples
    # 1000 * 27721 * 4 bytes approx 110 MB memory per batch
    batch_size = 1000 
    
    correct = 0
    k = 10
    
    print(f"Computing Recall@{k} for {num_samples} samples (Batched)...")

    for i in range(0, num_samples, batch_size):
        end = min(i + batch_size, num_samples)
        
        # Batch of query images [B, D]
        img_batch = image_emb[i:end]
        
        # Compute similarity against ALL texts [B, N]
        sims = img_batch @ text_emb.T
        
        # Get top k indices [B, K]
        # topk is much more memory efficient than argsort for small k
        _, topk_indices = sims.topk(k, dim=1)
        
        # Ground truth indices for this batch
        # The correct match for image[x] is text[x] (global index)
        targets = torch.arange(i, end, device=device).unsqueeze(1) 
        
        # Check matches
        hits = topk_indices.eq(targets).any(dim=1).sum().item()
        correct += hits

    recall = correct / num_samples
    print(f"Recall@{k}: {recall:.4f}")

    return recall

# ============================================================
# TRAINING LOOP
# ============================================================

scaler = torch.amp.GradScaler('cuda')

# ============================================================
# RESUME TRAINING
# ============================================================
best_recall = 0.0
start_epoch = 0
patience_counter = 0  # Track early stopping

if args.resume:
    if os.path.isfile(args.resume):
        print(f"=> loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=device)
        
        # If it's just a state dict (old format), we can't resume epoch/optimizer
        if 'epoch' not in checkpoint:
            print("Warning: Checkpoint appears to be model weights only. Starting from epoch 0.")
            model_to_load = model._orig_mod if hasattr(model, "_orig_mod") else model
            model_to_load.load_state_dict(checkpoint)
        else:
            start_epoch = checkpoint['epoch']
            best_recall = checkpoint.get('best_recall', 0.0)
            
            model_to_load = model._orig_mod if hasattr(model, "_orig_mod") else model
            model_to_load.load_state_dict(checkpoint['state_dict'])
            
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
    else:
        print(f"=> no checkpoint found at '{args.resume}'")

for epoch in range(start_epoch, args.epochs):

    model.train()
    total_loss = 0

    optimizer.zero_grad(set_to_none=True)

    loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{args.epochs}]", mininterval=10.0)
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

    # Evaluate recall on validation set (only every N epochs)
    if (epoch + 1) % args.val_every == 0:
        print(f"Running Validation (Epoch {epoch+1})...")
        val_recall = evaluate_recall(model, val_loader, device)
        print(f"Validation Recall@10: {val_recall:.4f}")

        if val_recall > best_recall:
            best_recall = val_recall
            # Save raw model state dict (uncompile if necessary, usually handled)
            # Use model._orig_mod if compiled to save clean weights
            save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            state = {
                'epoch': epoch + 1,
                'state_dict': save_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'best_recall': best_recall,
            }
            # Save best_model.pth adjacent to the final save_path
            best_path = os.path.join(os.path.dirname(args.save_path), "best_model.pth")
            torch.save(state, best_path)
            print(f"New best recall: {best_recall:.4f} (checkpoint saved to {best_path})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement in recall. Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f"Early stopping triggered! Model has not improved for {args.patience} validation checks.")
                break
    else:
        print(f"Skipping validation (Next: Epoch {epoch + 1 + args.val_every - (epoch+1)%args.val_every})")

# ============================================================
# SAVE FINAL MODEL
# ============================================================

save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
state = {
    'epoch': args.epochs,
    'state_dict': save_model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scaler': scaler.state_dict(),
    'best_recall': best_recall,
}
torch.save(state, args.save_path)
print("Final model saved to:", args.save_path)