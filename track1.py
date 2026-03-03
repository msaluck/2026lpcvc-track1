import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import open_clip

# ============================================================
# FAKE QUANT (Lightweight QAT Ready)
# ============================================================

class FakeQuant(nn.Module):
    def __init__(self, scale=127.0):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        if self.training:
            # STE (Straight-Through Estimator)
            xq = torch.round(x * self.scale) / self.scale
            return x + (xq - x).detach()
        return torch.round(x * self.scale) / self.scale


# ============================================================
# IMAGE ENCODER (MobileNetV3)
# ============================================================

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()

        backbone = mobilenet_v3_large(
            weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1
        )

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.proj = nn.Sequential(
        nn.Linear(960, 512),
        nn.GELU(),
        nn.Linear(512, embed_dim)
)
        self.norm = nn.LayerNorm(embed_dim)
        self.fake_quant = FakeQuant()

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
# FULL MODEL (MobileNet + OpenCLIP Text)
# ============================================================

class XRClip(nn.Module):
    def __init__(self, embed_dim=256, freeze_text=True):
        super().__init__()

        self.image_encoder = ImageEncoder(embed_dim)

        # Load OpenCLIP with explicit QuickGELU
        clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai",
            force_quick_gelu=True
        )

        self.clip_text = clip_model.token_embedding
        self.clip_pos = clip_model.positional_embedding
        self.clip_trans = clip_model.transformer
        self.clip_ln = clip_model.ln_final
        self.clip_proj = clip_model.text_projection

        # Register buffer for attention mask (critical for correct self-attention)
        self.register_buffer('attn_mask', clip_model.attn_mask)

        # Freezing logic
        if freeze_text:
            for p in self.clip_text.parameters(): p.requires_grad = False
            self.clip_pos.requires_grad = False
            for p in self.clip_trans.parameters(): p.requires_grad = False
            for p in self.clip_ln.parameters(): p.requires_grad = False
            self.clip_proj.requires_grad = False

        # Project CLIP 512-d to 256-d (This part IS trainable)
        self.text_proj = nn.Linear(512, embed_dim)
        
        # Norm and Quant (Trainable)
        self.text_norm = nn.LayerNorm(embed_dim)
        self.text_fake_quant = FakeQuant()

        # Learnable logit scale
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1 / 0.07))
        )

    def forward_text(self, text_input):
        # Explicit implementation matching OpenCLIP to ensure correct behavior
        x = self.clip_text(text_input)
        x = x + self.clip_pos   # (B, 77, 512) + (77, 512) -> (B, 77, 512)
        
        # Transformer expects (Batch, Seq, Dim) because batch_first=True in OpenCLIP
        x = self.clip_trans(x, attn_mask=self.attn_mask)  # (B, 77, 512)
        
        x = self.clip_ln(x)     # (B, 77, 512)
        
        # Take features from the eot embedding (eot_token is the highest number in each sequence)
        # Use gather for robust indexing. Note: in CLIP vocab, EOT (49407) is the largest ID,
        # so argmax(dim=-1) on the token IDs correctly finds the EOT position even with padding (0).
        idx = text_input.argmax(dim=-1, keepdim=True).unsqueeze(-1).expand(-1, -1, x.shape[-1])
        x = x.gather(1, idx).squeeze(1) # (B, 512)
        
        # Pass through original projection (512-dim)
        x = x @ self.clip_proj # (B, 512) @ (512, 512) = (B, 512)

        return x

    def forward(self, image, text_input):

        image_emb = self.image_encoder(image)

        with torch.no_grad():
            text_features = self.forward_text(text_input)

        text_emb = self.text_proj(text_features)
        text_emb = self.text_fake_quant(text_emb)
        text_emb = self.text_norm(text_emb)
        text_emb = F.normalize(text_emb, dim=-1)

        return image_emb, text_emb


def distillation_loss(student, teacher):
    # Ensure teacher is detached
    teacher = teacher.detach()
    return F.mse_loss(student, teacher)

class ClipLoss(nn.Module):
    def forward(self, image_emb, text_emb, logit_scale):
        logits = logit_scale.exp() * (image_emb @ text_emb.T)
        labels = torch.arange(len(logits), device=logits.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2


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
    return transform(image)


class RetrievalDataset(Dataset):
    def __init__(self, image_paths, texts, tokenizer):
        self.image_paths = image_paths
        self.texts = texts
        self.tokenizer = tokenizer

        print("Pre-tokenizing texts...")
        self.tokenized = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )["input_ids"]
        print("Pre-tokenization complete.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = load_and_preprocess(self.image_paths[idx])
        text = self.tokenized[idx]
        return image, text


# ============================================================
# ONNX EXPORT (AIHub Compatible)
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