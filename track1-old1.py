import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torch.utils.data import Dataset, Sampler
from PIL import Image
import torchvision.transforms as transforms
import random
import re
from collections import defaultdict

# ============================================================
# FAKE QUANT (Lightweight QAT)
# ============================================================

class FakeQuant(nn.Module):
    def __init__(self, scale=127.0):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        if self.training:
            return torch.round(x * self.scale) / self.scale
        return x


# ============================================================
# IMAGE ENCODER
# ============================================================

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256, pretrained=True):
        super().__init__()

        if pretrained:
            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        else:
            weights = None

        backbone = mobilenet_v3_large(weights=weights)
        
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.proj = nn.Linear(960, embed_dim)
        self.fake_quant = FakeQuant()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.proj(x)
        x = self.fake_quant(x)
        x = self.norm(x)
        x = F.normalize(x, dim=-1)
        return x


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, ffn_ratio=2.0):
        super().__init__()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)

        hidden_dim = int(embed_dim * ffn_ratio)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            FakeQuant(),
            nn.Linear(hidden_dim, embed_dim),
            FakeQuant()
        )

        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.norm1(x)

        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.norm2(x)

        return x


# ============================================================
# TEXT ENCODER
# ============================================================

class TextEncoder(nn.Module):
    def __init__(self,
                 vocab_size=49408,
                 max_len=77,
                 embed_dim=256,
                 depth=3,
                 num_heads=4):

        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, embed_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.fake_quant = FakeQuant()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids):
        x = self.token_embedding(input_ids)
        x = x + self.pos_embedding

        for blk in self.blocks:
            x = blk(x)

        eos_positions = (input_ids != 0).sum(dim=1) - 1
        x = x[torch.arange(x.size(0), device=x.device), eos_positions]

        x = self.proj(x)
        x = self.fake_quant(x)
        x = self.norm(x)
        x = F.normalize(x, dim=-1)

        return x


# ============================================================
# FULL MODEL
# ============================================================

# class XRClip(nn.Module):
#     def __init__(self, embed_dim=256, pretrained_image=True):
#         super().__init__()
#         self.image_encoder = ImageEncoder(embed_dim, pretrained=pretrained_image)
#         self.text_encoder = TextEncoder(embed_dim=embed_dim)

#         # Learnable logit scale (like real CLIP)
#         self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1/0.07)))

#     def forward(self, image, text_input):
#         image_emb = self.image_encoder(image)
#         text_emb = self.text_encoder(text_input)
#         return image_emb, text_emb

import open_clip

class XRClip(nn.Module):
    def __init__(self, embed_dim=256, pretrained_image=True):
        super().__init__()

        self.image_encoder = ImageEncoder(embed_dim, pretrained=pretrained_image)

        # Load OpenCLIP
        clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai"
        )

        self.clip_text_encoder = clip_model

        # Projection to 256
        self.text_proj = nn.Linear(512, embed_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1/0.07)))

    def forward(self, image, text_input):
        image_emb = self.image_encoder(image)

        text_features = self.clip_text_encoder.encode_text(text_input)
        text_emb = self.text_proj(text_features)

        text_emb = F.normalize(text_emb, dim=-1)

        return image_emb, text_emb

# ============================================================
# CLIP LOSS
# ============================================================

# class ClipLoss(nn.Module):
#     def __init__(self, temperature=0.07):
#         super().__init__()
#         self.temperature = temperature

#     def forward(self, image_emb, text_emb):
#         logits = image_emb @ text_emb.T
#         logits = logits / self.temperature

#         labels = torch.arange(len(logits), device=logits.device)

#         loss_i2t = F.cross_entropy(logits, labels)
#         loss_t2i = F.cross_entropy(logits.T, labels)

#         return (loss_i2t + loss_t2i) / 2

class ClipLoss(nn.Module):
    def forward(self, image_emb, text_emb, logit_scale):
        logits = logit_scale.exp() * (image_emb @ text_emb.T)

        labels = torch.arange(len(logits), device=logits.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2

# ============================================================
# HARD NEGATIVE MINING (Semantic Batching)
# ============================================================

def extract_semantic_key(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    words = text.split()

    stopwords = {"the", "a", "an", "this", "that"}

    for w in words:
        if w not in stopwords:
            return w

    return "unknown"


class SemanticBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.groups = list(dataset.semantic_groups.values())

    def __iter__(self):
        random.shuffle(self.groups)

        for group in self.groups:
            if len(group) < self.batch_size:
                continue

            random.shuffle(group)

            for i in range(0, len(group), self.batch_size):
                batch = group[i:i+self.batch_size]
                if len(batch) == self.batch_size:
                    yield batch

    def __len__(self):
        return sum(len(g) // self.batch_size for g in self.groups)


# ============================================================
# DATASET
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_and_preprocess(path):
    image = Image.open(path).convert("RGB")
    image = transform(image)
    return image


class RetrievalDataset(Dataset):
    def __init__(self, image_paths, texts, tokenizer):
        self.image_paths = image_paths
        self.texts = texts
        self.tokenizer = tokenizer

        # Pre-tokenize all texts to save CPU time during training
        # (600k samples * 77 ints is very small compared to images)
        print("Pre-tokenizing texts...")
        self.tokenized_texts = self.tokenizer(
            self.texts,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )["input_ids"]
        print("Pre-tokenization complete.")

        self.semantic_groups = defaultdict(list)

        # Optimize: avoid checking stopwords 600k times
        stopwords = {"the", "a", "an", "this", "that"}
        
        # Optimize loop with simple tokenization
        for idx, text in enumerate(self.texts):
            # Faster simplified key extraction
            text_lower = text.lower()
            words = text_lower.split()
            key = "unknown"
            for w in words:
                if w not in stopwords:
                    key = w
                    break
            self.semantic_groups[key].append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = load_and_preprocess(self.image_paths[idx])
        text_input = self.tokenized_texts[idx]
        return image, text_input


# ============================================================
# DISTILLATION SUPPORT
# ============================================================

def distillation_loss(student, teacher):
    student = F.normalize(student, dim=-1)
    teacher = F.normalize(teacher, dim=-1)
    return F.mse_loss(student, teacher)


# ============================================================
# ONNX EXPORT (AIHub Ready)
# ============================================================

def export_onnx(model, filename="xr_clip_s256.onnx"):
    model.eval()

    dummy_image = torch.randn(1, 3, 224, 224)
    dummy_text = torch.randint(0, 49408, (1, 77))

    torch.onnx.export(
        model,
        (dummy_image, dummy_text),
        filename,
        input_names=["image", "text_input"],
        output_names=["image_emb", "text_emb"],
        opset_version=17,
        dynamic_axes=None
    )


# ============================================================
# AIHUB CHECKLIST
# ============================================================

"""
AIHub Checklist:

1. model.eval() before export
2. Export static shapes only
3. Verify in Netron:
   - Inputs: image, text_input
   - Outputs: image_emb, text_emb
4. Compile on:
   device="XR2 Gen 2 (Proxy)"
5. Share compile job:
   compile_job.modify_sharing(add_emails=['lowpowervision@gmail.com'])
6. Confirm latency < 35ms
7. Submit compile job ID in form
"""


# ============================================================
# SAFE ENTRY
# ============================================================

if __name__ == "__main__":
    model = XRClip(embed_dim=256)
    dummy_image = torch.randn(1, 3, 224, 224)
    dummy_text = torch.randint(0, 49408, (1, 77))

    img_emb, txt_emb = model(dummy_image, dummy_text)

    print("Image embedding shape:", img_emb.shape)
    print("Text embedding shape:", txt_emb.shape)