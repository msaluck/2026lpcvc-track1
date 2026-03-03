
import torch
import torch.nn as nn
import torch.optim as optim
from track1 import XRClip
from transformers import CLIPTokenizer
from PIL import Image
import torchvision.transforms as transforms

# 1. Setup minimal environment
device = "cuda" if torch.cuda.is_available() else "cpu"
model = XRClip(embed_dim=256, freeze_text=True).to(device)  # Revert to frozen text
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# 2. Create synthetic data (ONE BATCH ONLY)
# We use totally different colors to make it easy for the visual encoder
print("Generating synthetic data...")
img1 = Image.new('RGB', (224, 224), color='red')
img2 = Image.new('RGB', (224, 224), color='blue')
img3 = Image.new('RGB', (224, 224), color='green')
img4 = Image.new('RGB', (224, 224), color='black')

texts = ["a red image", "a blue image", "a green image", "a black image"]

# Preprocess
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

images_tensor = torch.stack([transform(img) for img in [img1, img2, img3, img4]]).to(device)
text_inputs = tokenizer(texts, padding="max_length", max_length=77, return_tensors="pt")["input_ids"].to(device)

print(f"Images shape: {images_tensor.shape}")
print(f"Text shape: {text_inputs.shape}")

# 3. Training Loop (Single Batch Overfit)
print("\nStarting micro-training on 4 samples...")
model.train()

for i in range(201):
    optimizer.zero_grad()
    
    img_emb, txt_emb = model(images_tensor, text_inputs)
    
    # Simple contrastive loss manually
    logits = (img_emb @ txt_emb.T) * model.logit_scale.exp()
    labels = torch.arange(4, device=device)
    
    loss = (torch.nn.functional.cross_entropy(logits, labels) + 
            torch.nn.functional.cross_entropy(logits.T, labels)) / 2
    
    loss.backward()
    optimizer.step()
    
    # Calculate accuracy
    preds = logits.argmax(dim=1)
    acc = (preds == labels).float().mean()
    
    if i % 20 == 0:
        print(f"Iter {i}: Loss={loss.item():.4f}, Acc={acc.item():.2f}")
        print(f"    Logits diagonal: {logits.diag().detach().cpu().numpy()}")
        print(f"    Logits off-diag mean: {logits.fill_diagonal_(0).mean().item():.4f}")

# 4. Final verification
print("\nFinal Check:")
with torch.no_grad():
    img_emb, txt_emb = model(images_tensor, text_inputs)
    sims = img_emb @ txt_emb.T
    print("Similarity Matrix:\n", sims.cpu().numpy().round(2))
    print("\nExpected diagonal to be high (close to 1.0), off-diagonal low.")
    
    if sims.diag().min() > 0.9:
        print("\nSUCCESS: Model can memorize simple colors.")
    else:
        print("\nFAILURE: Model cannot even distinguish colors.")
