import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch

# Transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Custom Dataset
class BoneCancerDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        # Διαβάζουμε το CSV
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Καθαρίζουμε τυχόν whitespace στα headers
        self.data.columns = self.data.columns.str.strip()

        # Δημιουργούμε τη στήλη label: cancer=1, normal=0
        # Αν για κάποιο λόγο υπάρχει μόνο μια από τις δύο, παίρνει την τιμή της
        if 'cancer' in self.data.columns:
            self.data['label'] = self.data['cancer'].astype(int)
        elif 'normal' in self.data.columns:
            self.data['label'] = 1 - self.data['normal'].astype(int)  # normal=1 -> label=0
        else:
            raise ValueError("Το CSV πρέπει να έχει τη στήλη 'cancer' ή 'normal'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx]['filename'])
        image = Image.open(img_name).convert("RGB")
        label = torch.tensor(self.data.iloc[idx]['label'], dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label

# CSVs και φάκελοι εικόνων
train_csv = '/content/drive/MyDrive/xray-ai-project/dataset/train/_classes.csv'
valid_csv = '/content/drive/MyDrive/xray-ai-project/dataset/valid/_classes.csv'
test_csv  = '/content/drive/MyDrive/xray-ai-project/dataset/test/_classes.csv'

train_img_dir = '/content/drive/MyDrive/xray-ai-project/dataset/train'
valid_img_dir = '/content/drive/MyDrive/xray-ai-project/dataset/valid'
test_img_dir  = '/content/drive/MyDrive/xray-ai-project/dataset/test'

# Dataset objects
train_dataset = BoneCancerDataset(train_csv, img_dir=train_img_dir, transform=train_transform)
valid_dataset = BoneCancerDataset(valid_csv, img_dir=valid_img_dir, transform=val_transform)
test_dataset  = BoneCancerDataset(test_csv, img_dir=test_img_dir, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Έλεγχος
for imgs, labels in train_loader:
    print(imgs.shape, labels.shape)
    break

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Υποθέτουμε ότι τα train_loader και valid_loader είναι έτοιμα

# 1. Ορισμός του CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),  # αν input image=224x224
            nn.ReLU(),
            nn.Linear(128, 2)  # 2 κλάσεις: cancer / normal
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 2. Αρχικοποίηση μοντέλου, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels).item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total

    # Validation
    model.eval()
    val_corrects = 0
    val_total = 0
    with torch.no_grad():
        for imgs, labels in valid_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels).item()
            val_total += labels.size(0)
    val_acc = val_corrects / val_total

    print(f"Epoch {epoch+1}/{num_epochs} - "
          f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
          f"Val Acc: {val_acc:.4f}")
    
    # Test loop
model.eval()
test_corrects = 0
test_total = 0

all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)

        test_corrects += torch.sum(preds == labels).item()
        test_total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = test_corrects / test_total
print(f"Test Accuracy: {test_acc:.4f}")

# Προβολή προβλέψεων για μερικές εικόνες
for i in range(10):
    print(f"Image {i+1}: True Label = {all_labels[i]}, Predicted = {all_preds[i]}")