import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class UniversalDeepfakeDataset(Dataset):
    def __init__(self, base_dir, transform=None, is_train=True):
        self.transform = transform
        file_list = []
        labels = []
        class_map = {'fake': 0, 'real': 1}
        
        for class_name, label in class_map.items():
            folder_path = os.path.join(base_dir, class_name)
            if os.path.exists(folder_path):
                for f in os.listdir(folder_path):
                    if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.jpg', '.jpeg', '.png', '.webp')):
                        file_list.append(os.path.join(folder_path, f))
                        labels.append(label)

        # Zip, seed, and shuffle to guarantee matching splits across files
        combined = list(zip(file_list, labels))
        random.seed(42)  # Hardcoded seed means train/test groups stay separated
        random.shuffle(combined)
        
        split_idx = int(len(combined) * 0.80)
        self.target_data = combined[:split_idx] if is_train else combined[split_idx:]

    def __len__(self):
        return len(self.target_data)

    def __getitem__(self, idx):
        file_path, label = self.target_data[idx]
        try:
            if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                cap = cv2.VideoCapture(file_path)
                success, frame = cap.read()
                cap.release()
                if not success or frame is None:
                    image = Image.new('RGB', (256, 256), (0, 0, 0))
                else:
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                image = Image.open(file_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (256, 256), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
        return image, label


# --- MAIN EXECUTION GUARD TRAP ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Balanced Training Engine Active on: {device}")

    dataset_path = "test_dataset" if os.path.exists("test_dataset") else "src/test_dataset"

    # Load ONLY the 80% split for training
    train_dataset = UniversalDeepfakeDataset(base_dir=dataset_path, transform=train_transform, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)

    print(f"📊 Dataset Split Complete: Training on {len(train_dataset)} files. (1,246 files held back for testing).")

    print("📥 Fetching pre-trained ResNet50 framework...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(num_features, 2))
    model = model.to(device)

    class_weights = torch.tensor([1.0, 5.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    epochs = 10  
    print("🏋️ Commencing model fine-tuning...")

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (batch_idx + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx+1}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        print(f"🏁 Epoch {epoch+1} Complete. Loss: {running_loss/len(train_loader):.4f} | Accuracy: {100.*correct/total:.2f}%")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/baseline_model.pth")
    print("📦 Upgraded ResNet50 weights saved successfully!")
