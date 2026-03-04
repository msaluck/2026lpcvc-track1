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
    """
    Simulates quantization effects during training (Quantization Aware Training - QAT).

    Mathematically:
        1. Scaling: x_scaled = x * scale
        2. Rounding: x_int = round(x_scaled)
        3. De-scaling: x_quant = x_int / scale

    Backpropagation (Straight-Through Estimator - STE):
        The round() function has 0 gradient essentially everywhere.
        To allow gradient flow, we use the identity gradient:
        dL/dx ≈ dL/dx_quant
        
        Implementation:
            x_quant = x + (x_quant - x).detach()
        
        Forward pass: returns x_quant
        Backward pass: returns gradient of x (as if rounding didn't exist)
    """
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
    """
    Image Encoder based on MobileNetV3-Large.
    
    Architecture:
        1. Backbone: MobileNetV3-Large (IMAGENET1K_V1 weights)
           Output: (B, 960, 7, 7)
        2. Pooling: AdaptiveAvgPool2d((1, 1))
           Output: (B, 960, 1, 1) -> Flatten -> (B, 960)
        3. Projection Head:
           Linear(960 -> 512) -> GELU -> Linear(512 -> embed_dim)
        4. Normalization and Quantization:
           LayerNorm -> FakeQuant -> L2 Normalize (features on unit hypersphere)

    This architecture is chosen for low-power inference latency.
    """
    def __init__(self, embed_dim=256):
        super().__init__()

        backbone = mobilenet_v3_large(
            weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1
        )

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Projection Head: Projects 960-dim backbone features into shared embedding space
        self.proj = nn.Sequential(
            nn.Linear(960, 512),
            nn.GELU(),
            nn.Linear(512, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.fake_quant = FakeQuant()

    def forward(self, x):
        """
        Forward pass for image encoder.
        
        Args:
            x (torch.Tensor): Input images of shape (B, 3, 224, 224)
            
        Returns:
            torch.Tensor: Normalized image embeddings (B, embed_dim)
        """
        # Backbone Feature Extraction
        x = self.features(x)         # (B, 960, 7, 7)
        
        # Global Average Pooling
        x = self.pool(x)             # (B, 960, 1, 1)
        x = torch.flatten(x, 1)      # (B, 960)

        # Projection to Joint Embedding Space
        x = self.proj(x)             # (B, embed_dim)
        
        # Quantization Aware Layer
        x = self.fake_quant(x)
        
        # Normalization
        x = self.norm(x)
        
        # L2 Normalize: Required for Cosine Similarity
        # ||x||_2 = 1
        x = F.normalize(x, dim=-1)

        return x


# ============================================================
# FULL MODEL (MobileNet + OpenCLIP Text)
# ============================================================

class XRClip(nn.Module):
    """
    XRClip: A cross-modal retrieval model (Image-Text).
    
    Structure:
        - Image Encoder: Custom MobileNetV3 (Learnable)
        - Text Encoder: Distilled/Frozen CLIP ViT-B-32 (Fixed Weights)
    
    Why this design?
        The competition requires low-power inference on images.
        We freeze the heavy text encoder and only train a lightweight image encoder + projection.
        This allows high-quality text embeddings (from CLIP) without the training cost.
        A small projection layer (text_proj) adapts the 512-dim CLIP space to our 256-dim space.
    """
    def __init__(self, embed_dim=256, freeze_text=True):
        super().__init__()

        self.image_encoder = ImageEncoder(embed_dim)

        # -------------------------------------------------------------
        # Load Pre-trained OpenCLIP (ViT-B-32)
        # We manually extract components to support ONNX Export customization
        # -------------------------------------------------------------
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

        # Register buffer for attention mask (critical for correct self-attention in Transformer)
        # Without this, future tokens could leak into current prediction
        self.register_buffer('attn_mask', clip_model.attn_mask)

        # Freezing Logic: Lock the heavy transformer layers
        if freeze_text:
            for p in self.clip_text.parameters(): p.requires_grad = False
            self.clip_pos.requires_grad = False
            for p in self.clip_trans.parameters(): p.requires_grad = False
            for p in self.clip_ln.parameters(): p.requires_grad = False
            self.clip_proj.requires_grad = False

        # -------------------------------------------------------------
        # Adaptation Layers (Trainable)
        # -------------------------------------------------------------
        
        # Project CLIP 512-d to 256-d Target Space
        self.text_proj = nn.Linear(512, embed_dim)
        
        # Norm and Quant (Trainable) - same structure as Image Encoder
        self.text_norm = nn.LayerNorm(embed_dim)
        self.text_fake_quant = FakeQuant()

        # Learnable logit scale (Temperature parameter for Softmax)
        # Initialized to ln(1/0.07) ≈ 2.65 -> temperature = 0.07
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1 / 0.07))
        )

    def forward_text(self, text_input):
        """
        Custom forward pass for Text Encoder to ensure ONNX compatibility.
        
        Standard CLIP text encoding flow:
        1. Token Embedding: (B, 77) -> (B, 77, 512)
        2. Positional Embedding: Add learnable position vectors
        3. Transformer Encoder: 12 layers of Self-Attention + MLP
        4. Layer Norm: Final normalization
        5. EOT Extraction: Take the vector at the [EOT] token position
        
        Args:
            text_input (torch.Tensor): Tokenized text (B, 77)
            
        Returns:
            torch.Tensor: Text features (B, 512)
        """
        # 1. & 2. Embeddings
        x = self.clip_text(text_input)
        x = x + self.clip_pos   # (B, 77, 512) + (77, 512) -> (B, 77, 512) (Broadcasting)
        
        # 3. Transformer Encoder
        # attn_mask ensures causal masking (tokens can't attend to future)
        x = self.clip_trans(x, attn_mask=self.attn_mask)  # (B, 77, 512)
        
        # 4. Final Norm
        x = self.clip_ln(x)     # (B, 77, 512)
        
        # 5. Extract EOT (End-Of-Text) Features
        # The EOT token index varies per sequence length.
        # We find the index of the highest token ID (EOT ID is max in vocab).
        # idx shape: (B, 1, 1) -> expanded to (B, 1, 512)
        idx = text_input.argmax(dim=-1, keepdim=True).unsqueeze(-1).expand(-1, -1, x.shape[-1])
        
        # Gather the vector at the EOT position
        x = x.gather(dim=1, index=idx).squeeze(1) # (B, 512)
        
        # 6. CLIP Projection (512 -> 512)
        x = x @ self.clip_proj 

        return x

    def forward(self, image, text_input):
        """
        Joint Forward Pass.
        
        Args:
            image (torch.Tensor): (B, 3, 224, 224)
            text_input (torch.Tensor): (B, 77)
            
        Returns:
            image_emb (torch.Tensor): (B, 256)
            text_emb (torch.Tensor): (B, 256)
        """
        # Encode Image (Trainable)
        image_emb = self.image_encoder(image)

        # Encode Text (Frozen CLIP + Trainable Projection)
        with torch.no_grad():
            text_features = self.forward_text(text_input)

        # Project Text to Joint Space
        text_emb = self.text_proj(text_features)  # 512 -> 256
        text_emb = self.text_fake_quant(text_emb)
        text_emb = self.text_norm(text_emb)
        text_emb = F.normalize(text_emb, dim=-1)

        return image_emb, text_emb


def distillation_loss(student, teacher):
    """
    Computes the Mean Squared Error (MSE) loss during distillation.
    
    Args:
        student (torch.Tensor): Embeddings from the student model (B, D)
        teacher (torch.Tensor): Embeddings from the teacher model (B, D)
        
    Returns:
        torch.Tensor: Scalar loss value
    """
    # Ensure teacher gradients are detached (we don't train the teacher)
    teacher = teacher.detach()
    return F.mse_loss(student, teacher)

class ClipLoss(nn.Module):
    """
    Symmetric Cross Entropy Loss (InfoNCE).
    
    Standard contrastive loss used in CLIP training.
    Maximizes the cosine similarity of N positive (image, text) pairs
    while minimizing the similarity of N^2 - N negative pairs.
    
    Equation:
        L_i2t = -log( exp(s_ii / T) / sum_j(exp(s_ij / T)) )
        L_t2i = -log( exp(s_ii / T) / sum_j(exp(s_ji / T)) )
        L = (L_i2t + L_t2i) / 2
        
        Where:
        - s_ij = image_i @ text_j (Cosine Similarity)
        - T = temperature parameter (logit_scale.exp())
    """
    def forward(self, image_emb, text_emb, logit_scale):
        # 1. Compute Similarity Matrix
        #    sims: (B, B) matrix where sim[i, j] matches image[i] to text[j]
        logits = logit_scale.exp() * (image_emb @ text_emb.T)
        
        # 2. Assign ground truth labels
        #    Correct matches are on the diagonal (i == j)
        labels = torch.arange(len(logits), device=logits.device)

        # 3. Calculate Loss in Both Directions
        #    Image-to-Text: For each image, which text is correct? (Row-wise Softmax)
        loss_i2t = F.cross_entropy(logits, labels)
        
        #    Text-to-Image: For each text, which image is correct? (Column-wise Softmax)
        loss_t2i = F.cross_entropy(logits.T, labels)

        # 4. Average Loss
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