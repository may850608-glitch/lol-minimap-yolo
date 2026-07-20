"""把 positions.csv 畫成走位熱區圖,並計算量化指標。

產出:
  outputs/heatmap_all.png        — 10 隻英雄熱區小圖總覽
  outputs/heatmap_<champ>.png    — 指定英雄(預設 Keria=Poppy)的熱區大圖
  並在終端機印出量化指標。

用法:
    python scripts/09_analyze.py            # 預設分析 Poppy
    python scripts/09_analyze.py Rell       # 分析其他英雄
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
CHAMPS = CFG["champions"]
BG = ROOT / "data" / "assets" / "minimap_background.png"
CSV = ROOT / "outputs" / "positions.csv"
SIZE = 480  # 輸出圖尺寸


def load_positions():
    import csv
    rows = []
    with CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["champion"], float(r["time_s"]), float(r["x"]), float(r["y"]), float(r["conf"])))
    return rows


def heatmap(points, bg):
    """points: list of (x,y) 正規化。回傳疊上熱區的 BGR 圖。"""
    acc = np.zeros((SIZE, SIZE), np.float32)
    for x, y in points:
        px, py = int(x * SIZE), int(y * SIZE)
        if 0 <= px < SIZE and 0 <= py < SIZE:
            acc[py, px] += 1
    acc = cv2.GaussianBlur(acc, (0, 0), sigmaX=SIZE * 0.025)
    if acc.max() > 0:
        acc = acc / acc.max()
    cmap = cv2.applyColorMap((acc * 255).astype(np.uint8), cv2.COLORMAP_JET)
    mask = (acc > 0.06)[..., None]
    out = np.where(mask, cv2.addWeighted(bg, 0.45, cmap, 0.55, 0), bg)
    return out.astype(np.uint8)


def metrics(name, pts, times):
    pts = np.array(pts)
    if len(pts) == 0:
        print(f"{name}: 無資料"); return
    cover = len(np.unique((pts * 12).astype(int), axis=0)) / (12 * 12) * 100
    # 上/中/下:用 y- x 對角線粗略分區(LoL 地圖左下=己方,右上=敵方)
    diag = pts[:, 0] - pts[:, 1]
    print(f"\n== {name}(Keria 追蹤目標)量化指標 ==" if name == "Poppy" else f"\n== {name} ==")
    print(f"  偵測樣本數 : {len(pts)}  (約涵蓋 {times[-1]/60:.0f} 分鐘)")
    print(f"  地圖覆蓋率 : {cover:.0f}%  (走過的格子 / 12x12 網格)")
    print(f"  平均位置   : x={pts[:,0].mean():.2f}, y={pts[:,1].mean():.2f}  (0=左/上,1=右/下)")
    print(f"  活動範圍   : x±{pts[:,0].std():.2f}, y±{pts[:,1].std():.2f}  (越大代表游走範圍越廣)")


def main(target):
    bg0 = cv2.imread(str(BG))
    bg = cv2.resize(bg0, (SIZE, SIZE))
    rows = load_positions()

    # 總覽:10 小圖
    tiles = []
    for c in CHAMPS:
        pts = [(x, y) for ch, t, x, y, cf in rows if ch == c]
        tile = cv2.resize(heatmap(pts, bg), (240, 240))
        cv2.putText(tile, c, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(tile)
    grid = np.vstack([np.hstack(tiles[:5]), np.hstack(tiles[5:])])
    cv2.imwrite(str(ROOT / "outputs" / "heatmap_all.png"), grid)

    # 目標英雄大圖 + 指標
    pts = [(x, y) for ch, t, x, y, cf in rows if ch == target]
    ts = [t for ch, t, x, y, cf in rows if ch == target]
    big = heatmap(pts, bg)
    cv2.putText(big, f"{target} heatmap", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(str(ROOT / "outputs" / f"heatmap_{target}.png"), big)

    metrics(target, pts, ts or [0])
    print(f"\n輸出:outputs/heatmap_all.png 與 outputs/heatmap_{target}.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Poppy")
