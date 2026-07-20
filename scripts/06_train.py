"""在合成資料上訓練 YOLOv11 偵測小地圖英雄。

用法:
    python scripts/06_train.py            # 預設 yolo11n, 60 epochs, imgsz=256
    python scripts/06_train.py yolo11s.pt 100

訓練好的權重會在 runs/detect/train*/weights/best.pt,
本腳本最後會複製一份到 models/minimap.pt 方便後續使用。
"""
import shutil
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
DATA = ROOT / "data" / "dataset" / "data.yaml"


def main(model_name: str, epochs: int) -> None:
    model = YOLO(model_name)
    model.train(
        data=str(DATA),
        epochs=epochs,
        imgsz=CFG["train_imgsz"],
        batch=64,
        device=0,            # 用 GPU 0 (RTX 4060)
        patience=20,
        project=str(ROOT / "runs" / "detect"),
        name="train",
        exist_ok=False,
    )
    # 找出最新 train* 的 best.pt 複製到 models/
    runs = sorted((ROOT / "runs" / "detect").glob("train*"), key=lambda p: p.stat().st_mtime)
    best = runs[-1] / "weights" / "best.pt"
    dest = ROOT / "models" / "minimap.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, dest)
    print(f"\n最佳權重 -> {dest}")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "yolo11n.pt"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    main(model_name, epochs)
