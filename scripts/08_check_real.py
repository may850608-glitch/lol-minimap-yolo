"""在真實比賽畫面上測試模型(sim2real 驗證)。

從影片抽幾個時間點,裁小地圖、跑模型、把偵測框畫上去,拼成一張圖。
用來肉眼確認合成訓練的模型在真實小地圖上是否抓得到、認得對。

用法:
    python scripts/08_check_real.py                 # 預設幾個時間點
    python scripts/08_check_real.py 600 900 1500    # 指定秒數
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
MM = CFG["minimap"]
SIZE = CFG["train_imgsz"]
CHAMPS = CFG["champions"]


def find_vod() -> str:
    vods = sorted((ROOT / "data" / "vods").glob("*.mp4"))
    if not vods:
        raise SystemExit("data/vods 找不到影片")
    return str(vods[0])


def main(times):
    model = YOLO(str(ROOT / "models" / "minimap.pt"))
    cap = cv2.VideoCapture(find_vod())
    panels = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        mm = frame[MM["y0"]:MM["y1"], MM["x0"]:MM["x1"]]
        mm = cv2.resize(mm, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        res = model.predict(mm, conf=0.35, verbose=False)[0]
        vis = mm.copy()
        for b in res.boxes:
            cid = int(b.cls)
            conf = float(b.conf)
            x0, y0, x1, y1 = map(int, b.xyxy[0])
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 1)
            cv2.putText(vis, f"{CHAMPS[cid]} {conf:.2f}", (x0, max(8, y0 - 2)),
                        cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), 1)
        cv2.putText(vis, f"{t}s n={len(res.boxes)}", (3, SIZE - 5),
                    cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 255), 1)
        panels.append(vis)
    cap.release()

    # 拼成一列,放大
    grid = np.hstack(panels)
    grid = cv2.resize(grid, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    out = ROOT / "outputs" / "check_real.png"
    cv2.imwrite(str(out), grid)
    print(f"完成 -> {out}")


if __name__ == "__main__":
    times = [int(x) for x in sys.argv[1:]] or [600, 1000, 1500, 2000]
    main(times)
