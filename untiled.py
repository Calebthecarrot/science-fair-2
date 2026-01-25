import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# ======================
# CONFIG
# ======================
data_dir = "data"
batch_size = 16
num_epochs = 10
learning_rate = 0.001
threshold = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

labels = ["blocked", "unblocked", "yellow"]
num_classes = len(labels)

# ======================
# CUSTOM DATASET FOR MULTI-LABEL
# ======================
class MultiLabelDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.targets = []

        # Expect subfolders per label
        for i, label in enumerate(labels):
            label_dir = os.path.join(root_dir, label)
            for fname in os.listdir(label_dir):
                if fname.endswith((".jpg", ".png")):
                    path = os.path.join(label_dir, fname)
                    self.image_paths.append(path)

                    # multi-hot vector
                    target = torch.zeros(num_classes, dtype=torch.float32)
                    target[i] = 1.0
                    self.targets.append(target)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        target = self.targets[idx]
        return img, target

# ======================
# TRANSFORMS
# ======================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# ======================
# DATALOADERS
# ======================
train_dataset = MultiLabelDataset(os.path.join(data_dir, "train"), transform=transform)
val_dataset = MultiLabelDataset(os.path.join(data_dir, "val"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ======================
# MODEL
# ======================
class MultiLabelPathClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, num_classes)  # multi-label logits

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        return self.fc2(x)

model = MultiLabelPathClassifier().to(device)

# ======================
# LOSS & OPTIMIZER
# ======================
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ======================
# TRAINING LOOP
# ======================
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/len(train_loader):.4f}")

    # ======================
    # VALIDATION
    # ======================
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            correct += (preds == targets).sum().item()
            total += targets.numel()

    print(f"Validation Accuracy: {100 * correct / total:.2f}%")

torch.save(model.state_dict(), "multi_label_path_classifier.pth")
print("Model saved as multi_label_path_classifier.pth")
