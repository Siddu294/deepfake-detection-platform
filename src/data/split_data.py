"""Split extracted face images by source-video group to reduce leakage.

Expected input: data/extracted_faces/{real,fake}/<video_stem>_face_<n>.jpg
Output: data/processed/{train,val,test}/{real,fake}/...
"""
from collections import defaultdict
from pathlib import Path
import random
import shutil

SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def source_group(path: Path) -> str:
    # extract_faces.py names crops <original-video-stem>_face_<index>.jpg
    stem = path.stem
    marker = "_face_"
    return stem.rsplit(marker, 1)[0] if marker in stem else stem


def split_dataset(source_base="data/extracted_faces", output_base="data/processed", ratios=(0.8, 0.1, 0.1)):
    if abs(sum(ratios) - 1.0) > 1e-8 or any(r < 0 for r in ratios):
        raise ValueError("Split ratios must be non-negative and sum to 1.")
    source = Path(source_base)
    output = Path(output_base)
    rng = random.Random(SEED)
    class_groups = {}
    for cls in ("fake", "real"):
        folder = source / cls
        if not folder.is_dir():
            raise FileNotFoundError(f"Required extracted-face folder not found: {folder}")
        groups = defaultdict(list)
        for path in folder.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                groups[source_group(path)].append(path)
        if not groups:
            raise ValueError(f"No images found in {folder}")
        keys = sorted(groups)
        rng.shuffle(keys)
        n = len(keys)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        if n >= 3:
            n_train = max(1, min(n_train, n - 2))
            n_val = max(1, min(n_val, n - n_train - 1))
        partitions = {
            "train": keys[:n_train],
            "val": keys[n_train:n_train + n_val],
            "test": keys[n_train + n_val:],
        }
        class_groups[cls] = (groups, partitions)

    # Rebuild only output_base after all inputs have been validated.
    if output.exists():
        shutil.rmtree(output)
    counts = {}
    for cls, (groups, partitions) in class_groups.items():
        for split, group_keys in partitions.items():
            target = output / split / cls
            target.mkdir(parents=True, exist_ok=True)
            copied = 0
            for key in group_keys:
                for src in groups[key]:
                    shutil.copy2(src, target / src.name)
                    copied += 1
            counts[(split, cls)] = copied
    for split in ("train", "val", "test"):
        print(f"{split:5s} | fake={counts[(split, 'fake')]:5d} | real={counts[(split, 'real')]:5d}")
    print(f"Split complete. Source groups were assigned together. Seed={SEED}")


if __name__ == "__main__":
    split_dataset()
