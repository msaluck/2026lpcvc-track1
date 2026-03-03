import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# ============================================================
# IMAGE ENCODER
# ============================================================

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()

        backbone = mobilenet_v3_large(weights=None)

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.proj = nn.Linear(960, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.proj(x)
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
            nn.Linear(hidden_dim, embed_dim),
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
                 depth=4,
                 num_heads=4):

        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, embed_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids):
        # (B, 77)

        x = self.token_embedding(input_ids)
        x = x + self.pos_embedding

        for blk in self.blocks:
            x = blk(x)

        # Proper EOS extraction (last non-zero token)
        eos_positions = (input_ids != 0).sum(dim=1) - 1
        x = x[torch.arange(x.size(0), device=x.device), eos_positions]

        x = self.proj(x)
        x = self.norm(x)
        x = F.normalize(x, dim=-1)

        return x


# ============================================================
# FULL MODEL
# ============================================================

class XRClip(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)

    def forward(self, image, text_input):
        image_emb = self.image_encoder(image)
        text_emb = self.text_encoder(text_input)
        return image_emb, text_emb


# ============================================================
# CLIP CONTRASTIVE LOSS
# ============================================================

class ClipLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_emb, text_emb):
        logits = image_emb @ text_emb.T
        logits = logits / self.temperature

        labels = torch.arange(len(logits), device=logits.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2


# ============================================================
# DATASET
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # scales to 0-1
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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = load_and_preprocess(self.image_paths[idx])

        tokens = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )

        text_input = tokens["input_ids"].squeeze(0)

        return image, text_input


# ============================================================
# METRICS
# ============================================================

def compute_recall_at_k(image_emb, text_emb, k=10):
    sims = image_emb @ text_emb.T
    ranks = sims.argsort(dim=-1, descending=True)

    correct = torch.arange(len(image_emb), device=image_emb.device).unsqueeze(1)
    recall = (ranks[:, :k] == correct).any(dim=1).float().mean()

    return recall.item()


# ============================================================
# ONNX EXPORT
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
# TRAINING ENTRY POINT
# ============================================================

def train(model, dataloader, epochs=10, lr=3e-4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = ClipLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for images, text_inputs in dataloader:

            images = images.to(device)
            text_inputs = text_inputs.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                image_emb, text_emb = model(images, text_inputs)
                loss = criterion(image_emb, text_emb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss/len(dataloader):.4f}")


# ============================================================
# MAIN (SAFE EXECUTION)
# ============================================================

if __name__ == "__main__":

    model = XRClip(embed_dim=256)

    dummy_image = torch.randn(1, 3, 224, 224)
    dummy_text = torch.randint(0, 49408, (1, 77))

    img_emb, txt_emb = model(dummy_image, dummy_text)

    print("Image embedding shape:", img_emb.shape)
    print("Text embedding shape:", txt_emb.shape)

    # Uncomment after training
    # export_onnx(model)