# recognizer_init.py
import numpy as np
from runtime_integration import RuntimeEmbedder, CoachRecognizer

EMBEDDER = r"C:\coach_data\reid_runs\coach_binary\embedder.pt"
GALLERY  = r"C:\coach_data\reid_runs\coach_binary\gallery_coach.npy"

def build_recognizer(device=None):
    emb = RuntimeEmbedder(EMBEDDER, device=device, img_h=256, img_w=128)
    gal = np.load(GALLERY).astype("float32")   # (N,512)
    recog = CoachRecognizer(emb, gal)
    # Базові пороги (потім підкрутиш):
    recog.cosine_t = 0.80      # 0.78–0.85 типово
    recog.vote_window = 15     # скільки останніх кадрів беремо
    recog.vote_need = 10       # скільки позитивів треба у вікні
    return recog
