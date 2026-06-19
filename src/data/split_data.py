import os
import glob
import random
import shutil

def split_processed_data(processed_base_dir="data/processed", train_ratio=0.8):
    print("✂️ Beginning dataset distribution split...")
    for split in ['train', 'val']:
        for label in ['real', 'fake']:
            os.makedirs(os.path.join(processed_base_dir, split, label), exist_ok=True)
            
    for label in ['real', 'fake']:
        class_path = os.path.join(processed_base_dir, label)
        if not os.path.exists(class_path):
            continue
            
        all_images = glob.glob(os.path.join(class_path, "**", "*.jpg"), recursive=True)
        random.seed(42)
        random.shuffle(all_images)
        
        split_idx = int(len(all_images) * train_ratio)
        train_imgs = all_images[:split_idx]
        val_imgs = all_images[split_idx:]
        
        for img in train_imgs:
            shutil.copy(img, os.path.join(processed_base_dir, 'train', label, os.path.basename(img)))
        for img in val_imgs:
            shutil.copy(img, os.path.join(processed_base_dir, 'val', label, os.path.basename(img)))
            
        print(f"📊 Class '{label}': Assigned {len(train_imgs)} images to Train, {len(val_imgs)} images to Val.")

if __name__ == "__main__":
    split_processed_data()
