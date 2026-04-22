# make_gallery.py
import os, sys, glob
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms
from train_binary_reid import Backbone



EMBEDDER_CKPT = r"coach_data/reid_runs/coach_binary/embedder.pt"
IMAGES_DIR = r"coach_data/gallery_coach"
OUT_NPY = r"coach_data/reid_runs/coach_binary/gallery_coach.npy"

def _load_embedder_state(device):
    sd = torch.load(EMBEDDER_CKPT, map_location=device)
    # якщо ключі з префіксом "backbone.", прибираємо його
    if isinstance(sd, dict) and any(isinstance(k, str) and k.startswith("backbone.") for k in sd.keys()):
        sd = {k.replace("backbone.", "", 1): v for k, v in sd.items()}
    return sd

def get_embedder(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = Backbone(out_dim=512)
    sd = _load_embedder_state(device)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    tf = transforms.Compose([
        transforms.Resize((256,128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return model, tf, device

@torch.inference_mode()
@torch.inference_mode()
def embed_img(model, tf, device, bgr):
    # BGR -> RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # numpy -> PIL.Image
    pil = Image.fromarray(rgb)
    # трансформації -> тензор
    x = tf(pil).unsqueeze(0).to(device)
    feat = model(x, return_embed=True)           # (1,512)
    v = feat.squeeze(0).cpu().numpy().astype("float32")
    v /= (np.linalg.norm(v) + 1e-12)
    return v

def main():
    if not os.path.isdir(IMAGES_DIR):
        print("[ERR] images dir not found:", IMAGES_DIR); sys.exit(1)

    model, tf, device = get_embedder()
    paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.*")))
    feats, keep = [], []

    for p in paths:
        if p.lower().endswith((".jpg",".jpeg",".png",".bmp")):
            img = cv2.imread(p)
            if img is None:
                continue
            v = embed_img(model, tf, device, img)
            feats.append(v); keep.append(os.path.basename(p))

    if not feats:
        print("[ERR] no images found in", IMAGES_DIR); sys.exit(1)

    feats = np.stack(feats, axis=0).astype("float32")   # (N,512)
    np.save(OUT_NPY, feats)
    print("[OK] saved gallery:", OUT_NPY, "shape:", feats.shape)

if __name__ == "__main__":
    main()
