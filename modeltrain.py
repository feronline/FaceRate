import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50
import torch.nn.functional as F

# === Ayarlar ===
DATASET_FOLDER = "/content/drive/MyDrive/Humans"
CSV_FILE = "/content/drive/MyDrive/merged_dataset.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 10
NUM_FOLDS = 5


# === Dataset Sınıfı ===
class FaceDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'path']
        score = self.df.loc[idx, 'puan'] / 10.0
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(score, dtype=torch.float32)


# === Transformlar ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# === ArcFace benzeri backbone (ResNet50 + feature output)
class ArcFaceBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # ResNet50 modelini yükle
        base = resnet50(weights="DEFAULT")

        # Orijinal ResNet50'nin son katmanını çıkar
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # avgpool'a kadar
        self.output = nn.Linear(2048, 1)

        # Grad-CAM için gerekli hook kayıtları
        self.gradients = None
        self.activations = None

        # Layer4'ün çıktısını kaydetmek için hook
        def save_activation(module, input, output):
            self.activations = output

        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # ResNet50'nin layer4'üne hook ekle
        for name, module in self.backbone.named_modules():
            if 'layer4' in name and len(name.split('.')) == 2:  # layer4'ün kendisi
                module.register_forward_hook(save_activation)
                module.register_backward_hook(save_gradient)
                break

    def forward(self, x):
        features = self.backbone(x)  # [B, 2048, 1, 1]
        features_flat = features.view(features.size(0), -1)  # [B, 2048]
        output = self.output(features_flat).squeeze()
        output = torch.sigmoid(output)
        return output, features_flat


# === RankNet Loss
class RankNetLoss(nn.Module):
    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin

    def forward(self, preds, targets):
        pair_loss = 0.0
        count = 0
        for i in range(len(preds)):
            for j in range(len(preds)):
                if i != j:
                    diff_pred = preds[i] - preds[j]
                    diff_true = targets[i] - targets[j]
                    prob = torch.sigmoid(diff_pred)
                    label = 1 if diff_true > 0 else 0
                    pair_loss += F.binary_cross_entropy(prob, torch.tensor(label, dtype=torch.float32).to(preds.device))
                    count += 1
        return pair_loss / count if count > 0 else torch.tensor(0.0).to(preds.device)

# === Hybrid Loss (RankNet + MSE)
class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.ranknet = RankNetLoss()
        self.mse = nn.MSELoss()
        self.alpha = alpha  # MSE katkısı

    def forward(self, preds, targets):
        return self.ranknet(preds, targets) + self.alpha * self.mse(preds, targets)

# === Grad-CAM (Düzeltilmiş)
def generate_gradcam(model, image_tensor, save_path, label=""):
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    image_tensor.requires_grad_()

    # Forward pass
    output, _ = model(image_tensor)

    # Backward pass
    model.zero_grad()
    output.backward()

    # Gradients ve activations'ları al
    gradients = model.gradients
    activations = model.activations

    if gradients is None or activations is None:
        print("Gradients veya activations None, Grad-CAM oluşturulamadı")
        return

    # Global average pooling
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # Activations'ı ağırlıklandır
    for i in range(activations.size(1)):
        activations[:, i, :, :] *= pooled_gradients[i]

    # Heatmap oluştur
    heatmap = torch.mean(activations, dim=1).squeeze().detach().cpu()
    heatmap = torch.clamp(heatmap, min=0)  # ReLU

    # Heatmap'i normalize et
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # NumPy'a çevir ve resize et
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)

    # Colormap uygula
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Orijinal görüntüyü al
    image_np = image_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # OpenCV BGR formatı

    # Blend
    blended = cv2.addWeighted(image_np, 0.6, heatmap_colored, 0.4, 0)

    # Kaydet
    cv2.imwrite(save_path, blended)
    print(f"Grad-CAM kaydedildi: {save_path}")


# === Veriyi oku
df = pd.read_csv(CSV_FILE)
df['path'] = df['dosya_adi'].apply(lambda x: os.path.join(DATASET_FOLDER, x))

kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
fold = 0

for train_index, val_index in kf.split(df):
    fold += 1
    print(f"\n===== FOLD {fold} =====")
    train_df = df.iloc[train_index]
    val_df = df.iloc[val_index]

    train_loader = DataLoader(FaceDataset(train_df, transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(FaceDataset(val_df, transform), batch_size=BATCH_SIZE)

    model = ArcFaceBackbone().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = HybridLoss(alpha=0.5)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for images, scores in tqdm(train_loader, desc=f"Fold {fold} - Epoch {epoch + 1}"):
            images, scores = images.to(DEVICE), scores.to(DEVICE)
            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, scores)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch + 1}] Loss: {total_loss:.4f}")

        # === GradCAM örneği (Her epoch'ta değil, sadece son epoch'ta)
        if epoch == EPOCHS - 1:
            try:
                example_img, _ = next(iter(val_loader))
                generate_gradcam(model, example_img[0], f"gradcam_fold{fold}_epoch{epoch + 1}.jpg")
            except Exception as e:
                print(f"Grad-CAM oluşturulurken hata: {e}")

    # === Değerlendirme
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, scores in val_loader:
            images = images.to(DEVICE)
            outputs, _ = model(images)
            preds = outputs.cpu().numpy() * 10
            actuals = scores.numpy() * 10
            y_pred.extend(preds)
            y_true.extend(actuals)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    spearman_corr = spearmanr(y_true, y_pred).correlation

    print(f"\n📊 Fold {fold} Metrikleri:")
    print(f"MAE: {mae:.2f} | MSE: {mse:.2f} | R²: {r2:.4f} | Spearman: {spearman_corr:.4f}")


# === Modeli Kaydet ve Drive'a Kopyala
MODEL_PATH = "arcface_ranknet_final.pth"
torch.save(model.state_dict(), MODEL_PATH)
print("✅ Model kaydedildi:", MODEL_PATH)

# Drive'a yedekle
DRIVE_SAVE_PATH = "/content/drive/MyDrive/face_model_yedek/arcface_ranknet_final1.pth"
os.makedirs(os.path.dirname(DRIVE_SAVE_PATH), exist_ok=True)
print(f"📂 Drive'a kopyalandı: {DRIVE_SAVE_PATH}")
