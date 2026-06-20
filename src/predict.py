import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import sys

class DeepfakeDetectorMatched(nn.Module):
    def __init__(self):
        super().__init__()
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
            nn.Linear(16384, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def predict_image(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DeepfakeDetectorMatched()
    model_path = "models/baseline_model.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Trained model weights not found at {model_path}!")
        return
        
    state_dict = torch.load(model_path, map_location=device)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
        
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    
    # FIXED: Changed from (224, 224) to (128, 128) to match the model's expected 16384 features
    transform_pipeline = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found at {image_path}")
        return
        
    frame = cv2.imread(image_path)
    if frame is None:
        print("❌ Error: Could not read image file.")
        return
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        print("⚠️ Warning: No face detected. Evaluating full frame details...")
        face_patch = frame
    else:
        x, y, w, h = faces[0]
        face_patch = frame[y:y+h, x:x+w]
        print("🎯 Successfully localized face patch for analysis.")

    face_rgb = cv2.cvtColor(face_patch, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb)
    input_tensor = transform_pipeline(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)
        
    classes = ["FAKE", "REAL"]
    prediction = classes[predicted_idx.item()]
    confidence_score = confidence.item() * 100
    
    print("\n" + "="*40)
    print(f"🔍 ANALYSIS FOR: {os.path.basename(image_path)}")
    print(f"🤖 RESULT: The face is predicted to be ** {prediction} **")
    print(f"📊 CONFIDENCE: {confidence_score:.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 src/predict.py <path_to_image>")
    else:
        predict_image(sys.argv[1])
