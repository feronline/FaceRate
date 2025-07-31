import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from torchvision.models import resnet50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Model Sınıfı (Aynısı)
class ArcFaceBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet50(weights="DEFAULT")
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.output = nn.Linear(2048, 1)

    def forward(self, x):
        features = self.backbone(x)
        features_flat = features.view(features.size(0), -1)
        output = self.output(features_flat).squeeze()
        return output, features_flat

# === Görüntü İşleme
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(image_path, model_path):
    model = ArcFaceBackbone().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output, _ = model(input_tensor)
        score = round(output.item() * 10, 2)
        score = max(0, min(score, 10))
        # 0–1 → 0–10 arası
        return round(score, 2)

# === Örnek kullanım
image_path = "1 (6).jpg"  # buraya test etmek istediğin yüzü koy
model_path = "arcface_ranknet_final.pth"

score = predict(image_path, model_path)
print(f"💯 Tahmini Güzellik Skoru: {score}/10")
