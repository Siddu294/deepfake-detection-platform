import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from src.models.model import DeepfakeDetector

def train_model():
    print("🚀 Initializing Deepfake Detector Training Pipeline...")
    
    # 1. Paths
    train_dir = "data/processed/train"
    val_dir = "data/processed/val"
    checkpoint_path = "models/training_checkpoint.pth"
    final_model_path = "models/baseline_model.pth"
    os.makedirs("models", exist_ok=True)
    
    # 2. Advanced Data Transformations
    transform_pipeline = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    # 3. Data Loaders
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_pipeline)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform_pipeline)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"📊 Datasets Loaded. Training Images: {len(train_dataset)} | Validation Images: {len(val_dataset)}")
    
    # 4. Model Setup
    model = DeepfakeDetector()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    start_epoch = 0
    total_epochs = 15
    
    # 🔄 RESUME CHECKPOINT SYSTEM
    if os.path.exists(checkpoint_path):
        print(f"📦 Found saved training session! Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"▶️ Resuming training automatically from Epoch [{start_epoch + 1}/{total_epochs}]")
    else:
        print("🏋️ Starting fresh training session from Epoch 1...")
        
    # 5. Core Training Loop
    for epoch in range(start_epoch, total_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation evaluation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_accuracy = (correct / total) * 100
        print(f"📈 Epoch [{epoch + 1}/{total_epochs}] -> Train Loss: {epoch_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")
        
        # 💾 SAVE CHECKPOINT after every single epoch loop
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        
    # Save final model
    torch.save(model.state_dict(), final_model_path)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    print(f"💾 Success! Upgraded weights saved to {final_model_path}")

if __name__ == "__main__":
    train_model()
