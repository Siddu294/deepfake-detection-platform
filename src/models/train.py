import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from src.models.model import DeepfakeDetector

def train_model():
    print("🚀 Initializing Deepfake Detector Training Pipeline...")
    
    transform_pipeline = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    train_dir = "data/processed/train"
    val_dir = "data/processed/val"
    
    if not os.path.exists(train_dir) or len(os.listdir(train_dir)) == 0:
        print("❌ Error: Training splits are empty or don't exist!")
        return

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_pipeline)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform_pipeline)
    
    # Using batch size of 4 for slightly faster training steps
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    print(f"📊 Datasets Loaded. Training Images: {len(train_dataset)} | Validation Images: {len(val_dataset)}")
    
    model = DeepfakeDetector()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 15
    print(f"🏋️ Starting Training Loop for {epochs} Learning Epochs...")
    
    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        
        # --- VALIDATION PHASE (New Metric Tracking) ---
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_accuracy = 100 * correct / total
        print(f"📈 Epoch [{epoch+1}/{epochs}] -> Train Loss: {epoch_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")
        
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/baseline_model.pth")
    print("💾 Success! Upgraded weights saved to models/baseline_model.pth")

if __name__ == "__main__":
    train_model()
