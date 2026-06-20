import cv2
import os

def extract_faces_from_videos():
    print("🎞️ Beginning face extraction to a safe holding folder...")
    
    output_base = "data/extracted_faces"
    os.makedirs(os.path.join(output_base, "real"), exist_ok=True)
    os.makedirs(os.path.join(output_base, "fake"), exist_ok=True)
    
    # CHANGED: Pointing directly to data/raw where real/ and fake/ subfolders live
    video_root = "data/raw"
    classes = ["real", "fake"]
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    for cls in classes:
        cls_dir = os.path.join(video_root, cls)
        if not os.path.exists(cls_dir):
            print(f"⚠️ Directory not found: {cls_dir}")
            continue
            
        videos = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.mp4', '.avi', '.mkv'))]
        print(f"🎬 Found {len(videos)} videos in class '{cls}'")
        
        for video_name in videos:
            video_path = os.path.join(cls_dir, video_name)
            cap = cv2.VideoCapture(video_path)
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < 30:
                continue
                
            interval = total_frames // 30
            saved_count = 0
            
            for i in range(30):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
                ret, frame = cap.read()
                if not ret:
                    break
                    
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    face_patch = frame[y:y+h, x:x+w]
                    
                    img_name = f"{os.path.splitext(video_name)[0]}_face_{saved_count}.jpg"
                    out_path = os.path.join(output_base, cls, img_name)
                    cv2.imwrite(out_path, face_patch)
                    saved_count += 1
            
            cap.release()
            print(f"✅ Extracted {saved_count} face patch(es) from {video_name}")

if __name__ == "__main__":
    extract_faces_from_videos()
