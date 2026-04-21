
import os, argparse, torch
import torch.nn as nn
from train_binary_reid import Backbone, ClassifierHead

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="C:\\coach_data\\reid_runs\\coach_binary\\model_best.pt")
    ap.add_argument("--out", type=str, required=True, help="C:\\coach_data\\reid_runs\\coach_binary\\embedder.pt")
    return ap.parse_args()

class Embedder(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
    @torch.inference_mode()
    def forward(self, x):
        # повертає L2-нормовані ембеддинги (B,512)
        return self.backbone(x, return_embed=True)

def main():
    args = get_args()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = Backbone(out_dim=512)
    model.load_state_dict(ckpt["model"], strict=True)
    emb = Embedder(model).eval()
    torch.save(emb.state_dict(), args.out)
    print("[OK] saved:", args.out)

if __name__ == "__main__":
    main()
