# runtime_integration.py
import torch, torch.nn as nn
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
from train_binary_reid import Backbone

# Параметри
COSINE_T    = 0.80
VOTE_WINDOW = 15
VOTE_NEED   = 10

def _load_embedder_state(ckpt_embedder_path, device):
    sd = torch.load(ckpt_embedder_path, map_location=device)
    # прибираємо префікс "backbone." якщо він є
    if isinstance(sd, dict) and any(isinstance(k, str) and k.startswith("backbone.") for k in sd.keys()):
        sd = {k.replace("backbone.", "", 1): v for k, v in sd.items()}
    return sd

class RuntimeEmbedder(nn.Module):
    def __init__(self, ckpt_embedder_path, device=None, img_h=256, img_w=128):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = Backbone(out_dim=512).to(self.device).eval()
        sd = _load_embedder_state(ckpt_embedder_path, self.device)
        self.backbone.load_state_dict(sd, strict=True)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.tf = transforms.Compose([
            transforms.Resize((img_h, img_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    @torch.inference_mode()
    def embed_bgr(self, img_bgr: np.ndarray) -> np.ndarray:
        if img_bgr is None or img_bgr.size == 0 or img_bgr.shape[0] < 20 or img_bgr.shape[1] < 20:
            v = np.zeros((512,), dtype="float32")
            return v

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil = Image.fromarray(rgb)

        x = self.tf(pil).unsqueeze(0).to(self.device)
        feat = self.backbone(x, return_embed=True)
        v = feat.squeeze(0).detach().cpu().numpy().astype("float32")
        v /= (np.linalg.norm(v) + 1e-12)
        return v

class CoachRecognizer:
    def __init__(self, embedder: RuntimeEmbedder, gallery_feats: np.ndarray):
        self.emb = embedder
        self.gallery = gallery_feats.astype("float32")
        # safety: переконаємося, що галерея L2-нормована
        norms = np.linalg.norm(self.gallery, axis=1, keepdims=True) + 1e-12
        self.gallery = self.gallery / norms

        self.vote_window = VOTE_WINDOW
        self.vote_need = VOTE_NEED
        self.cosine_t = COSINE_T

    @staticmethod
    def cosine(a, b):
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
        return a.dot(b.T)

    def is_coach_crop(self, crop_bgr):
        vec = self.emb.embed_bgr(crop_bgr)[None, :]       # (1,512)
        score = float(self.cosine(vec, self.gallery).max())
        return (score >= self.cosine_t), score

    def vote_track(self, track_obj, is_positive):
        """
        Додає голос у вікно та повертає label: "COACH"/"OTHER".
        Підтримує як dict-треки (track_obj["coach_votes"]), так і об'єкти з атрибутами.
        """
        # dict-трек
        if isinstance(track_obj, dict):
            votes = track_obj.get("coach_votes", [])
            votes.append(is_positive)
            votes = votes[-self.vote_window:]
            track_obj["coach_votes"] = votes
            label = "COACH" if sum(votes) >= self.vote_need else "OTHER"
            track_obj["label"] = label
            return label

        # об'єкт-трек
        if not hasattr(track_obj, "coach_votes"):
            track_obj.coach_votes = []
        track_obj.coach_votes.append(is_positive)
        track_obj.coach_votes = track_obj.coach_votes[-self.vote_window:]
        label = "COACH" if sum(track_obj.coach_votes) >= self.vote_need else "OTHER"
        track_obj.label = label
        return label
