import os
import shutil
import random

def split_dataset():
    print("✂️ Beginning dataset distribution split with 50/50 balance...")
    
    # Read from the safe holding directory
    source_base = "data/extracted_faces"
    processed_dir = "data/processed"
    
    real_faces_source = os.path.join(source_base, "real")
    fake_faces_source = os.path.join(source_base, "fake")
    
    if not os.path.exists(real_faces_source) or not os.path.exists(fake_faces_source):
        print("⚠️ No extracted faces found in data/extracted_faces/. Please run extraction first.")
        return
        
    real_images = [os.path.join(real_faces_source, f) for f in os.listdir(real_faces_source) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    fake_images = [os.path.join(fake_faces_source, f) for f in os.listdir(fake_faces_source) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(real_images) == 0 or len(fake_images) == 0:
        print(f"⚠️ Missing data! Real images: {len(real_images)}, Fake images: {len(fake_images)}")
        return

    min_count = min(len(real_images), len(fake_images))
    
    random.shuffle(real_images)
    random.shuffle(fake_images)
    
    balanced_real = real_images[:min_count]
    balanced_fake = fake_images[:min_count]
    
    split_idx = int(min_count * 0.8)
    
    # Wipe and rebuild ONLY the processed directory safely
    shutil.rmtree(processed_dir, ignore_errors=True)
    
    train_real_dir = os.path.join(processed_dir, "train", "real")
    train_fake_dir = os.path.join(processed_dir, "train", "fake")
    val_real_dir = os.path.join(processed_dir, "val", "real")
    val_fake_dir = os.path.join(processed_dir, "val", "fake")
    
    for d in [train_real_dir, train_fake_dir, val_real_dir, val_fake_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Distribute balanced data safely
    for img in balanced_real[:split_idx]:
        shutil.copy(img, os.path.join(train_real_dir, os.path.basename(img)))
    for img in balanced_real[split_idx:]:
        shutil.copy(img, os.path.join(val_real_dir, os.path.basename(img)))
        
    for img in balanced_fake[:split_idx]:
        shutil.copy(img, os.path.join(train_fake_dir, os.path.basename(img)))
    for img in balanced_fake[split_idx:]:
        shutil.copy(img, os.path.join(val_fake_dir, os.path.basename(img)))
        
    print(f"📊 Class 'real': Assigned {split_idx} images to Train, {min_count - split_idx} images to Val.")
    print(f"📊 Class 'fake': Assigned {split_idx} images to Train, {min_count - split_idx} images to Val.")
    print("✅ Dataset successfully balanced and distributed 50/50!")

if __name__ == "__main__":
    split_dataset()
