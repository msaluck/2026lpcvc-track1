import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()

        backbone = mobilenet_v3_large(weights=None)

        # Remove classifier
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.proj = nn.Linear(960, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.features(x)             # (B, 960, 7, 7)
        x = self.pool(x)                 # (B, 960, 1, 1)
        x = torch.flatten(x, 1)          # (B, 960)
        x = self.proj(x)                 # (B, 256)
        x = self.norm(x)
        x = F.normalize(x, dim=-1)
        return x

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
        # Self-attention
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.norm1(x)

        # Feedforward
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.norm2(x)

        return x

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
        # input_ids: (B, 77)

        x = self.token_embedding(input_ids)   # (B, 77, 256)
        x = x + self.pos_embedding            # add positional

        for blk in self.blocks:
            x = blk(x)

        # Use EOS token representation (assume last token)
        # x = x[:, -1, :]                       # (B, 256)

        # find last non-zero token (EOS)
        eos_mask = (input_ids != 0).sum(dim=1) - 1
        x = x[torch.arange(x.size(0)), eos_mask]

        x = self.proj(x)
        x = self.norm(x)
        x = F.normalize(x, dim=-1)

        return x

class XRClip(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()

        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)

    def forward(self, image, text_input):
        image_emb = self.image_encoder(image)
        text_emb = self.text_encoder(text_input)

        return image_emb, text_emb

model = XRClip()

class ClipLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_emb, text_emb):
        # Normalize (extra safety)
        # image_emb = F.normalize(image_emb, dim=-1)
        # text_emb = F.normalize(text_emb, dim=-1)

        # Similarity
        logits = image_emb @ text_emb.T   # (N, N)
        logits = logits / self.temperature

        labels = torch.arange(len(logits)).to(logits.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        loss = (loss_i2t + loss_t2i) / 2
        return loss
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = XRClip(embed_dim=256).to(device)
criterion = ClipLoss(temperature=0.07)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-4
)

scaler = torch.cuda.amp.GradScaler()

def train_one_epoch(dataloader):
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

    return total_loss / len(dataloader)

from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # scales to 0–1
])

def load_and_preprocess(path):
    image = Image.open(path).convert("RGB")
    image = transform(image)
    return image

from torch.utils.data import Dataset

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
    
def compute_recall_at_k(image_emb, text_emb, k=10):
    sims = image_emb @ text_emb.T
    ranks = sims.argsort(dim=-1, descending=True)

    correct = torch.arange(len(image_emb), device=image_emb.device).unsqueeze(1)

    recall = (ranks[:, :k] == correct).any(dim=1).float().mean()
    return recall.item()

def export_onnx(model):
    model.eval()

    dummy_image = torch.randn(1, 3, 224, 224)
    dummy_text = torch.randint(0, 49408, (1, 77))

    img_emb, txt_emb = model(dummy_image, dummy_text)

    print(img_emb.shape)  # (1, 256)
    print(txt_emb.shape)  # (1, 256)

    torch.onnx.export(
        model,
        (dummy_image, dummy_text),
        "xr_clip_s256.onnx",
        input_names=["image", "text_input"],
        output_names=["image_emb", "text_emb"],
        opset_version=17,
        dynamic_axes=None
    )