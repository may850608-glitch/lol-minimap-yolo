"""從影片抽單一幀存成 PNG,用來檢查畫質、定位小地圖位置。

用法:
    python scripts/02_grab_frame.py "<影片路徑>" [秒數]
預設抽第 1200 秒(約 20 分鐘處,通常戰況清楚)。
"""
import sys
from pathlib import Path

import cv2

OUT = Path(__file__).resolve().parent.parent / "outputs"


def grab(video: str, seconds: float) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"無法開啟影片:{video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"解析度 {w}x{h}  fps={fps:.2f}  總幀數={total}  時長~{total / fps / 60:.1f} 分")

    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("讀取該時間點失敗")

    out = OUT / f"frame_{int(seconds)}s.png"
    cv2.imwrite(str(out), frame)
    print(f"已存:{out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 1200.0
    grab(sys.argv[1], secs)
