import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from src.models.model import DeepfakeDetector

def train_model():
    print("🚀 Initializing Deepfake Detector Training Pipeline...")
    
    # 1. Image Transformations (Normalization & Resizing to match model input)
    transform_pipeline = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    train_dir = "data/processed/train"
    val_dir = "data/processed/val"
    
    # Safety Check: Make sure data exists
    if not os.path.exists(train_dir) or len(os.listdir(train_dir)) == 0:
        print("❌ Error: Training splits are empty or don't exist. Run split_data.py first!")
        return

    # 2. Setup Data Loaders
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_pipeline)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform_pipeline)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    print(f"📊 Datasets Loaded. Training Images: {len(train_dataset)} | Validation Images: {len(val_dataset)}")
    
    # 3. Instantiate Model, Loss, and Optimizer
    model = DeepfakeDetector()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Core Training Loop Execution (1 Baseline Epoch for MVP verification)
    epochs = 1
    print(f"🏋️ Starting Training Loop for {epochs} Verification Epoch...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            optimizer.zero_grad()           # Clear out residual gradients
            outputs = model(images)         # Forward pass
            loss = criterion(outputs, labels) # Calculate loss error
            loss.backward()                 # Backward pass (Backpropagation)
            optimizer.step()                # Update network weights
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        print(f"📉 Epoch [{epoch+1}/{epochs}] complete. Training Loss: {epoch_loss:.4f}")
        
    # 5. Save the baseline model checkpoint
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/baseline_model.pth")
    print("💾 Success! Baseline weights saved to models/baseline_model.pth")

if __name__ == "__main__":
    train_model()
