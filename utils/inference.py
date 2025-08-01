import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from torchvision.models import resnet50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

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

def predict(image_path, model):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output, _ = model(input_tensor)
        score = torch.sigmoid(output).item() * 10
        return round(score, 2)
