import os
import sys
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report

# Force Python to locate modules in the root folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train_advanced import UniversalDeepfakeDataset

# --- MATCHED ARCHITECTURE MAP ---
def get_evaluation_model():
    from torchvision import models
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, 2)
    )
    return model

transform_pipeline = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_evaluation_model().to(device)
model_path = "models/baseline_model.pth"

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("🟢 Successfully loaded trained ResNet50 keys!")
else:
    print(f"❌ Weight file not found at: {model_path}")
    sys.exit(1)

model.eval()

dataset_path = "test_dataset" if os.path.exists("test_dataset") else "src/test_dataset"
# Load ONLY the remaining 20% unseen data split
test_dataset = UniversalDeepfakeDataset(base_dir=dataset_path, transform=transform_pipeline, is_train=False)

true_labels, pred_labels = [], []
print(f"⏳ Running inference over {len(test_dataset)} completely unseen validation items...")

for idx in range(len(test_dataset)):
    tensor, label = test_dataset[idx]
    tensor = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted_idx = torch.max(outputs, dim=1)
    
    true_labels.append(label)
    pred_labels.append(predicted_idx.item())

if len(true_labels) > 0:
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
    print("\n==========================================")
    print("    🎉 UNSEEN TEST DATA CONFUSION MATRIX 🎉 ")
    print("==========================================")
    print(f"True Real Items: [ TN: {cm[1][1]} | FP: {cm[1][0]} ]")
    print(f"True Deepfakes:  [ FN: {cm[0][1]} | TP: {cm[0][0]} ]")
    print("------------------------------------------")
    print(classification_report(true_labels, pred_labels, target_names=['FAKE', 'REAL'], zero_division=0))
