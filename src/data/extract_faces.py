import cv2
import os
import urllib.request

def download_cascade_if_needed():
    cascade_path = "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        print("📥 Downloading Haar Cascade face detector weights...")
        cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        urllib.request.urlretrieve(cascade_url, cascade_path)
    return cascade_path

def extract_faces_from_video(video_path, output_dir, max_frames=5):
    cascade_path = download_cascade_if_needed()
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video file {video_path}")
        return False
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // max_frames)
    
    frame_count = 0
    saved_count = 0
    
    print(f"🎞️ Processing video: {video_path} ({total_frames} total frames available)")
    
    while cap.isOpened() and saved_count < max_frames:
        # FIX: Use cap.read() instead of cap.get()
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            
            for idx, (x, y, w, h) in enumerate(faces):
                pad_h, pad_w = int(h * 0.1), int(w * 0.1)
                y1 = max(0, y - pad_h)
                y2 = min(frame.shape[0], y + h + pad_h)
                x1 = max(0, x - pad_w)
                x2 = min(frame.shape[1], x + w + pad_w)
                
                cropped = frame[y1:y2, x1:x2]
                output_filename = f"frame_{frame_count}_face_{idx}.jpg"
                cv2.imwrite(os.path.join(output_dir, output_filename), cropped)
                saved_count += 1
                
                if saved_count >= max_frames:
                    break
                    
        frame_count += 1
        
    cap.release()
    print(f"✅ Extracted and saved {saved_count} face patches to {output_dir}")
    return True

if __name__ == "__main__":
    print("🛠️ Utility module ready.")
