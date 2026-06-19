import torch
import torch.nn as nn

class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        
        # Convolutional Feature Extraction Layers
        self.features = nn.Sequential(
            # Layer 1: Input face image (3 channels: R,G,B) -> Extracts 16 basic edge features
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Downsamples image dimensions by 2
            
            # Layer 2: Deeper features (textures, geometric facial anomalies)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 3: High-level classification features
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fully Connected Classification Head
        # Assuming our preprocessing input image size is normalized to 128x128 pixels,
        # 3 max-pooling operations downsample 128 -> 64 -> 32 -> 16 pixels.
        self.classifier = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # Prevents overfitting by randomly turning off nodes
            nn.Linear(128, 2) # Final Output: 2 classes (0 = Real, 1 = Fake)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten multidimensional matrices to a single vector
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    # Smoke test to verify model instantiation and matrix transformations
    model = DeepfakeDetector()
    print("🧠 Model Architecture Created Successfully!")
    
    # Mock a batch of 4 preprocessed face image tensors (Batch size=4, Channels=3, H=128, W=128)
    mock_input = torch.randn(4, 3, 128, 128)
    mock_output = model(mock_input)
    print(f"📦 Input Shape: {mock_input.shape}")
    print(f"🎯 Output Logits Shape: {mock_output.shape} (Successfully matched 2 binary classes)")
