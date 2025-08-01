import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os

# === Model Sınıfı (Aynı)
class ArcFaceBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet50(weights="DEFAULT")
        self.backbone = torch.nn.Sequential(*list(base.children())[:-1])
        self.output = torch.nn.Linear(2048, 1)

    def forward(self, x):
        features = self.backbone(x)
        features_flat = features.view(features.size(0), -1)
        output = self.output(features_flat).squeeze()
        return output, features

# === Grad-CAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor):
        output, _ = self.model(input_tensor)
        self.model.zero_grad()
        output.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = gradients.mean(dim=[1, 2], keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        return cam.cpu().numpy()

# === Görüntü İşleme
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    return image, tensor

# === CAM'i Görsel Olarak Bindir
def visualize_cam(image_pil, cam, output_path="gradcam.jpg"):
    image_np = np.array(image_pil.resize((224, 224)))
    cam = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    image_np = np.float32(image_np) / 255
    cam_image = heatmap + image_np
    cam_image = cam_image / cam_image.max()
    cv2.imwrite(output_path, np.uint8(255 * cam_image[:, :, ::-1]))
    print(f"✅ Grad-CAM görseli kaydedildi: {output_path}")

# === Ana Test
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ArcFaceBackbone().to(DEVICE)
model.load_state_dict(torch.load("Models/arcface_ranknet_final1.pth", map_location=DEVICE))
model.eval()

image_path = "Faces/ben.jpeg"
image_pil, input_tensor = load_image(image_path)

# Grad-CAM
target_layer = model.backbone[-1]  # Son CNN katmanı
grad_cam = GradCAM(model, target_layer)
cam = grad_cam.generate(input_tensor)
visualize_cam(image_pil, cam)
