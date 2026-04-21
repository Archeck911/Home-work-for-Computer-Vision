
FULL TRAIN (Binary ReID: coach vs other)

Структура очікуваних даних:
C:\coach_data\dataset_reid\
  train\
    coach\        # кропи тренера (200–800+)
    other\        # кропи інших (2–5x від coach)
  val\
    coach\
    other\

Кроки:
1) Підготуй дані (або запусти prepare_split.py, якщо вихідні кропи у C:\coach_data\seeds\coach1 і C:\coach_data\seeds\other):
   python prepare_split.py

2) Тренуй модель:
   python train_binary_reid.py --data_root "C:\coach_data\dataset_reid" --out "C:\coach_data\reid_runs\coach_binary" --epochs 60 --batch 64 --lr 3e-4

3) Експортуй ембеддер (обріже класифікатор, залишить фічі + нормалізацію):
   python export_embedder.py --ckpt "C:\coach_data\reid_runs\coach_binary\model_best.pt" --out "C:\coach_data\reid_runs\coach_binary\embedder.pt"

4) Інтегруй у пайплайн YOLO+ByteTrack:
   див. runtime_integration.py (паспортні функції для is_coach_crop, голосування, пороги).
