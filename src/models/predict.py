import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from src.models.model import DeepfakeDetector

def download_cascade_if_needed():
    cascade_path = "models/haarcascade_frontalface_default.xml"
    return cascade_path

def predict_video(video_path):
    print(f"🎬 Analyzing Video: {os.path.basename(video_path)}...")
    
    # 1. Load the trained model architecture and weights
    model = DeepfakeDetector()
    model_path = "models/baseline_model.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file '{model_path}' not found! Run training first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # 2. Setup standard image transformation
    transform_pipeline = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    # 3. Initialize Face Detector
    cascade_path = download_cascade_if_needed()
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Cannot open video file!")
        return
        
    frame_count = 0
    face_predictions = []
    
    # Scan frames looking for faces
    while cap.isOpened() and len(face_predictions) < 10:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % 10 == 0:  # Sample every 10th frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
            
            for (x, y, w, h) in faces:
                # Crop face
                cropped = frame[y:y+h, x:x+w]
                # Convert BGR (OpenCV) to RGB (PIL/PyTorch)
                rgb_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_image)
                
                # Transform to tensor
                tensor_img = transform_pipeline(pil_img).unsqueeze(0)
                
                # Model Inference
                with torch.no_grad():
                    outputs = model(tensor_img)
                    probabilities = F.softmax(outputs, dim=1)
                    # Class 0: Fake, Class 1: Real (based on directory order alphabetical f vs r)
                    fake_prob = probabilities[0][0].item()
                    real_prob = probabilities[0][1].item()
                    
                    face_predictions.append((real_prob, fake_prob))
                    if len(face_predictions) >= 10:
                        break
        frame_count += 1
        
    cap.release()
    
    if not face_predictions:
        print("⚠️ Warning: No faces detected in this video clip to evaluate.")
        return
        
    # 4. Aggregate results across all sample faces found
    avg_real = sum(p[0] for p in face_predictions) / len(face_predictions)
    avg_fake = sum(p[1] for p in face_predictions) / len(face_predictions)
    
    print("\n--- 📊 DETECTION REPORT ---")
    if avg_fake > avg_real:
        print(f"🚨 ALERT: DEEPFAKE DETECTED (Confidence: {avg_fake * 100:.2f}%)")
    else:
        print(f"✅ VERIFIED: VIDEO IS REAL (Confidence: {avg_real * 100:.2f}%)")
    print("--------------------------")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("💡 Usage: python3 -m src.models.predict <path_to_video.mp4>")
    else:
        predict_video(sys.argv[1])
