import cv2
import os
import glob
import urllib.request

def download_cascade_if_needed():
    cascade_path = "models/haarcascade_frontalface_default.xml"
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(cascade_path):
        print("📥 Downloading Haar Cascade face detector weights...")
        cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        urllib.request.urlretrieve(cascade_url, cascade_path)
    return cascade_path

def process_entire_folder(input_folder, output_base_dir, max_frames_per_video=5):
    cascade_path = download_cascade_if_needed()
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    video_files = glob.glob(os.path.join(input_folder, "*.mp4"))
    print(f"📂 Found {len(video_files)} video(s) inside {input_folder}")
    
    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(output_base_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Cannot open {video_path}")
            continue
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 100  # Fallback fallback for stream web videos
        step = max(1, total_frames // max_frames_per_video)
        
        frame_count = 0
        saved_count = 0
        
        print(f"🎞️ Extracting faces from: {video_name} ({total_frames} frames)")
        
        while cap.isOpened() and saved_count < max_frames_per_video:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
                
                for idx, (x, y, w, h) in enumerate(faces):
                    pad_h, pad_w = int(h * 0.1), int(w * 0.1)
                    y1 = max(0, y - pad_h)
                    y2 = min(frame.shape[0], y + h + pad_h)
                    x1 = max(0, x - pad_w)
                    x2 = min(frame.shape[1], x + w + pad_w)
                    
                    cropped = frame[y1:y2, x1:x2]
                    output_filename = f"frame_{frame_count}_face_{idx}.jpg"
                    cv2.imwrite(os.path.join(video_output_dir, output_filename), cropped)
                    saved_count += 1
                    
                    if saved_count >= max_frames_per_video:
                        break
            frame_count += 1
            
        cap.release()
        print(f"✅ Extracted {saved_count} face patch(es) from {video_name}")

if __name__ == "__main__":
    print("🚀 Running Batch Processing Pipeline on ALL Raw Videos...")
    # Process Real Videos
    process_entire_folder("data/raw/real", "data/processed/real", max_frames_per_video=5)
    # Process Fake Videos (Added!)
    process_entire_folder("data/raw/fake", "data/processed/fake", max_frames_per_video=5)
