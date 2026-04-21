
import os, shutil, random
from pathlib import Path

# Вихідні: C:\coach_data\seeds\coach1, C:\coach_data\seeds\other
SEED_COACH = r"C:\coach_data\seeds\coach1"
SEED_OTHER = r"C:\coach_data\seeds\other"
OUT_ROOT   = r"C:\coach_data\dataset_reid"
TRAIN_PCT  = 0.8

def list_images(root):
    exts = {".jpg",".jpeg",".png",".bmp"}
    paths = [str(p) for p in Path(root).glob("*") if p.suffix.lower() in exts]
    return sorted(paths)

def split_copy(src, dst_train, dst_val):
    imgs = list_images(src)
    random.shuffle(imgs)
    n_train = int(len(imgs) * TRAIN_PCT)
    train, val = imgs[:n_train], imgs[n_train:]
    os.makedirs(dst_train, exist_ok=True)
    os.makedirs(dst_val, exist_ok=True)

    for s in train:
        shutil.copy2(s, os.path.join(dst_train, os.path.basename(s)))
    for s in val:
        shutil.copy2(s, os.path.join(dst_val, os.path.basename(s)))
    return len(train), len(val)

def main():
    random.seed(42)
    paths = [
        (SEED_COACH, os.path.join(OUT_ROOT, "train","coach"), os.path.join(OUT_ROOT, "val","coach")),
        (SEED_OTHER, os.path.join(OUT_ROOT, "train","other"), os.path.join(OUT_ROOT, "val","other")),
    ]
    total = {}
    for src, dt, dv in paths:
        if not os.path.isdir(src):
            print(f"[WARN] source not found: {src}")
            continue
        nt, nv = split_copy(src, dt, dv)
        total[src] = (nt, nv)
        print(f"[OK] {src} -> train:{nt}, val:{nv}")
    print("[DONE]", total)

if __name__ == "__main__":
    main()
