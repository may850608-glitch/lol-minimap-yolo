"""對整場比賽影片做小地圖英雄偵測,輸出每個取樣時間點的英雄座標 CSV。

座標為「小地圖內的正規化位置」(0~1,左上=地圖左上),即遊戲地圖位置。

用法:
    python scripts/07_detect.py                       # 自動找影片,2 fps 取樣
    python scripts/07_detect.py "<影片>" --stride 30  # 每 30 幀取一次
輸出:outputs/positions.csv
"""
import argparse
import csv
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
MM = CFG["minimap"]
SIZE = CFG["train_imgsz"]
CHAMPS = CFG["champions"]
TEAM = [CFG["teams"][0]] * 5 + [CFG["teams"][1]] * 5


def find_vod() -> str:
    vods = sorted((ROOT / "data" / "vods").glob("*.mp4"))
    if not vods:
        raise SystemExit("data/vods 找不到影片")
    return str(vods[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=None)
    ap.add_argument("--stride", type=int, default=30, help="每幾幀偵測一次(預設30,約2fps)")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--model", default=str(ROOT / "models" / "minimap.pt"))
    args = ap.parse_args()

    video = args.video or find_vod()
    model = YOLO(args.model)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"無法開啟影片:{video}")
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = ROOT / "outputs" / "positions.csv"
    f = out.open("w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(["frame", "time_s", "champion", "team", "conf", "x", "y"])

    idx = 0
    n_rows = 0
    while True:
        if idx % args.stride == 0:
            ok, frame = cap.read()
            if not ok:
                break
            mm = frame[MM["y0"]:MM["y1"], MM["x0"]:MM["x1"]]
            mm = cv2.resize(mm, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            res = model.predict(mm, conf=args.conf, verbose=False)[0]
            # 每個 class 取信心最高的一個偵測
            best = {}
            for b in res.boxes:
                cid = int(b.cls)
                conf = float(b.conf)
                if cid not in best or conf > best[cid][0]:
                    xc, yc = float(b.xywh[0][0]) / SIZE, float(b.xywh[0][1]) / SIZE
                    best[cid] = (conf, xc, yc)
            t = idx / fps
            for cid, (conf, xc, yc) in best.items():
                w.writerow([idx, f"{t:.2f}", CHAMPS[cid], TEAM[cid], f"{conf:.3f}",
                            f"{xc:.4f}", f"{yc:.4f}"])
                n_rows += 1
            if idx % (args.stride * 200) == 0:
                print(f"  幀 {idx}  t={t/60:.1f}min  累計 {n_rows} 筆")
        else:
            if not cap.grab():
                break
        idx += 1

    cap.release()
    f.close()
    print(f"完成 -> {out}  共 {n_rows} 筆")


if __name__ == "__main__":
    main()
