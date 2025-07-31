import os
import random
import csv
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm

# === Ayarlar ===
DATASET_FOLDER = "Humans"
CSV_FILE = "etiketler.csv"
AUTO_CSV_FILE = "auto_etiketler.csv"
SEED = 42
BATCH_SIZE = 32
EPOCHS = 10

# === Etiketli Veriyi Yükle ===
df = pd.read_csv(CSV_FILE)
df['path'] = df['dosya_adi'].apply(lambda x: os.path.join(DATASET_FOLDER, x))

# === Train/Val Böl ===
train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)

# === Dataset Sınıfı ===
class FaceScoreDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'path']
        score = self.df.loc[idx, 'puan'] / 10.0  # Normalized
        image = Image.open(img_path)
        if image.mode == 'P' or image.mode == 'RGBA':
            image = image.convert("RGBA").convert("RGB")
        else:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(score, dtype=torch.float32)


# === Transform ve Dataloader ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = FaceScoreDataset(train_df, transform=transform)
val_dataset = FaceScoreDataset(val_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# === Model Tanımı ===
class ResNetRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights="DEFAULT")
        base.fc = nn.Linear(base.fc.in_features, 1)
        self.model = base

    def forward(self, x):
        return self.model(x).squeeze(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetRegressor().to(device)

# === Eğitim ===
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train_model(epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, scores in train_loader:
            images, scores = images.to(device), scores.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, scores)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

train_model(EPOCHS)

# === Etiketsiz Görselleri Bul ===
etiketli_set = set(df['dosya_adi'])
tum_gorseller = [f for f in os.listdir(DATASET_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
kalanlar = sorted(list(set(tum_gorseller) - etiketli_set))

# === Unlabeled Dataset ===
class UnlabeledDataset(Dataset):
    def __init__(self, filenames, transform=None):
        self.files = filenames
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(DATASET_FOLDER, self.files[idx])
        image = Image.open(path)
        if image.mode == 'P' or image.mode == 'RGBA':
            image = image.convert("RGBA").convert("RGB")
        else:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.files[idx]


unlabeled_ds = UnlabeledDataset(kalanlar, transform=transform)
unlabeled_dl = DataLoader(unlabeled_ds, batch_size=BATCH_SIZE)

# === Tahmin Et ve Kaydet ===


# Tahmin ve kayıt
model.eval()
tahminler = []

with torch.no_grad():
    for images, filenames in tqdm(unlabeled_dl, desc="Auto tagging", unit="batch"):
        images = images.to(device)
        preds = model(images).cpu().numpy()
        preds = (preds * 10).clip(1, 10)
        for file, puan in zip(filenames, preds):
            tahminler.append((file, round(float(puan), 2)))

# CSV'ye yaz
auto_df = pd.DataFrame(tahminler, columns=["dosya_adi", "puan"])
auto_df.to_csv(AUTO_CSV_FILE, index=False)
print(f"{AUTO_CSV_FILE} kaydedildi.")