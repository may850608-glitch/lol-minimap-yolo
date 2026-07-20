"""用「中位數背景萃取」從比賽影片做出乾淨的小地圖底圖。

原理:地形靜態、英雄/視野/兵線會動。對整場取樣數百張小地圖逐像素取中位數,
動的東西被洗掉,留下乾淨地形(含靜態防禦塔)。這張底圖和真實轉播完全同源,
拿來合成訓練資料時 sim2real 落差最小。

用法:
    python scripts/03_make_background.py            # 自動找 data/vods 第一個影片
    python scripts/03_make_background.py "<影片路徑>"
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
MM = CFG["minimap"]
SIZE = CFG["train_imgsz"]
N_SAMPLES = 300  # 取樣張數;越多越乾淨但越慢


def find_vod() -> str:
    vods = sorted((ROOT / "data" / "vods").glob("*.mp4"))
    if not vods:
        raise SystemExit("data/vods 找不到影片")
    return str(vods[0])


def make_background(video: str) -> None:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"無法開啟影片:{video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 跳過開頭/結尾各 ~2 分鐘(轉場、暫停畫面),只取真正對戰段
    start, end = int(fps * 120), total - int(fps * 60)
    idxs = np.linspace(start, end, N_SAMPLES).astype(int)

    crops = []
    for i, fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        mm = frame[MM["y0"]:MM["y1"], MM["x0"]:MM["x1"]]
        mm = cv2.resize(mm, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        crops.append(mm)
        if (i + 1) % 50 == 0:
            print(f"  取樣 {i + 1}/{N_SAMPLES}")
    cap.release()

    stack = np.stack(crops, axis=0)
    bg = np.median(stack, axis=0).astype(np.uint8)

    out_dir = ROOT / "data" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "minimap_background.png"
    cv2.imwrite(str(out), bg)
    # 也存一張取樣範例,方便比對
    cv2.imwrite(str(out_dir / "minimap_sample.png"), crops[len(crops) // 2])
    print(f"完成,用了 {len(crops)} 張取樣 -> {out}")


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else find_vod()
    print(f"影片:{video}")
    make_background(video)
