"""合成 YOLO 訓練資料:把英雄頭像做成小地圖圖示,隨機貼到乾淨底圖上。

每張合成圖:
  - 以乾淨小地圖底圖為基底,做亮度/對比抖動、隨機霧區(模擬戰爭迷霧)。
  - 隨機選 3~10 隻英雄,隨機位置貼上(圓形遮罩 + 隊伍色環),可重疊(模擬團戰)。
  - 自動寫出 YOLO 標註 (class x_center y_center w h,皆正規化)。

class id = config.yaml 的 champions 順序。T1(前5)藍環、BLG(後5)紅環。

用法:
    python scripts/05_gen_synthetic.py            # 預設 2000 train / 200 val
    python scripts/05_gen_synthetic.py 4000 400
"""
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

cv2.setNumThreads(1)  # 多執行緒在部分環境會拋 Unknown C++ exception;生成不吃重,單執行緒即可

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
CHAMPS = CFG["champions"]
SIZE = CFG["train_imgsz"]
ASSETS = ROOT / "data" / "assets"
DATASET = ROOT / "data" / "dataset"

# 隊伍色環 (BGR):由真實畫面取樣得出(壓縮後偏柔和)。前 5 = T1 藍,後 5 = BLG 紅
BLUE = (183, 159, 101)
RED = (84, 73, 176)
TEAM_COLOR = [BLUE] * 5 + [RED] * 5

DIAM_RANGE = (20, 32)  # 圖示直徑(在 256 底圖上),對應真實 ~24px(後期會放大)


def load_assets():
    bg = cv2.imread(str(ASSETS / "minimap_background.png"))
    bg = cv2.resize(bg, (SIZE, SIZE))
    squares = [cv2.imread(str(ASSETS / "icons" / f"{c}.png")) for c in CHAMPS]
    return bg, squares


def make_icon(square, diameter, ring_color):
    """方形頭像 -> 圓形遮罩 + 隊伍色環,模擬真實小地圖圖示。回傳 (bgr, alpha)。"""
    # 裁掉方形頭像外緣 ~8%,讓臉填滿(真實圖示臉部幾乎填滿圓)
    h, w = square.shape[:2]
    m = int(round(w * 0.08))
    face = square[m:h - m, m:w - m]
    D = diameter
    icon = cv2.resize(face, (D, D), interpolation=cv2.INTER_AREA)

    c = (round(D / 2), round(D / 2))
    R = D * 0.47
    thick = max(2, round(D * 0.14))  # 粗環
    jit = lambda v: int(np.clip(v + random.randint(-22, 22), 0, 255))
    rc = tuple(jit(v) for v in ring_color)

    cv2.circle(icon, c, round(R), rc, thick, cv2.LINE_AA)            # 亮色環
    cv2.circle(icon, c, round(R + thick / 2), (28, 26, 30), 1, cv2.LINE_AA)  # 薄深色外緣
    alpha = np.zeros((D, D), np.uint8)
    cv2.circle(alpha, c, round(R + thick / 2), 255, -1, cv2.LINE_AA)

    if random.random() < 0.8:  # 壓縮模糊:真實圖示偏柔
        k = random.choice([3, 3, 5])
        icon = cv2.GaussianBlur(icon, (k, k), 0)
    return icon, alpha


def draw_viewport(canvas):
    """畫上觀戰視野框(白色矩形),真實畫面常壓在圖示上,需讓模型習慣。"""
    if random.random() < 0.85:
        w = random.randint(int(SIZE * 0.30), int(SIZE * 0.55))
        h = int(w * random.uniform(0.65, 0.82))
        x0 = random.randint(0, SIZE - w)
        y0 = random.randint(0, SIZE - h)
        col = random.randint(190, 255)
        cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (col, col, col), 1, cv2.LINE_AA)


def paste(bg, icon, alpha, cx, cy):
    d = icon.shape[0]
    x0, y0 = cx - d // 2, cy - d // 2
    x1, y1 = x0 + d, y0 + d
    if x0 < 0 or y0 < 0 or x1 > bg.shape[1] or y1 > bg.shape[0]:
        return
    roi = bg[y0:y1, x0:x1]
    a = (alpha.astype(np.float32) / 255.0)[..., None]
    roi[:] = (icon.astype(np.float32) * a + roi.astype(np.float32) * (1 - a)).astype(np.uint8)


def augment_bg(bg):
    out = bg.astype(np.float32)
    out *= random.uniform(0.55, 1.15)            # 亮度(後期戰鬥畫面明顯偏暗,下限放寬)
    out += random.uniform(-12, 12)               # 偏移
    out = np.clip(out, 0, 255).astype(np.uint8)
    # HSV 色相/飽和抖動(不同場/光照)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + random.randint(-6, 6)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.8, 1.2), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    # 隨機霧區:幾塊半透明深藍橢圓(後期戰鬥煙霧更多更暗)
    if random.random() < 0.8:
        fog = out.copy()
        for _ in range(random.randint(1, 4)):
            cx, cy = random.randint(0, SIZE), random.randint(0, SIZE)
            ax, ay = random.randint(30, 110), random.randint(30, 110)
            cv2.ellipse(fog, (cx, cy), (ax, ay), 0, 0, 360, (40, 25, 15), -1)
        out = cv2.addWeighted(out, random.uniform(0.55, 0.7), fog, random.uniform(0.3, 0.45), 0)
    # 守衛/眼位雜訊:滿地圖的小色點(藍/紅/黃綠小方塊與圓點)
    for _ in range(random.randint(0, 28)):
        px, py = random.randint(4, SIZE - 4), random.randint(4, SIZE - 4)
        col = random.choice([(200, 150, 60), (60, 60, 200), (60, 200, 200), (120, 220, 160)])
        if random.random() < 0.5:
            cv2.rectangle(out, (px - 2, py - 2), (px + 2, py + 2), col, -1)
        else:
            cv2.circle(out, (px, py), random.randint(2, 3), col, -1, cv2.LINE_AA)
    return out


def gen_split(n, split):
    img_dir = DATASET / "images" / split
    lbl_dir = DATASET / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    bg, squares = load_assets()
    margin = 18
    for k in range(n):
        canvas = augment_bg(bg)
        ids = random.sample(range(len(CHAMPS)), random.randint(3, len(CHAMPS)))
        lines = []
        for cid in ids:
            d = random.randint(*DIAM_RANGE)
            icon, alpha = make_icon(squares[cid], d, TEAM_COLOR[cid])
            cx = random.randint(margin, SIZE - margin)
            cy = random.randint(margin, SIZE - margin)
            paste(canvas, icon, alpha, cx, cy)
            bw = bh = d / SIZE
            lines.append(f"{cid} {cx / SIZE:.6f} {cy / SIZE:.6f} {bw:.6f} {bh:.6f}")
        draw_viewport(canvas)                       # 視野框畫在圖示之上
        if random.random() < 0.5:                   # 整體輕微模糊(影片壓縮感)
            canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
        cv2.imwrite(str(img_dir / f"{split}_{k:05d}.png"), canvas)
        (lbl_dir / f"{split}_{k:05d}.txt").write_text("\n".join(lines), encoding="utf-8")
        if (k + 1) % 500 == 0:
            print(f"  {split}: {k + 1}/{n}")


def write_data_yaml():
    names = {i: c for i, c in enumerate(CHAMPS)}
    y = {
        "path": str(DATASET),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    (DATASET / "data.yaml").write_text(yaml.dump(y, allow_unicode=True), encoding="utf-8")


if __name__ == "__main__":
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    n_val = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    random.seed(0)
    print(f"產生 {n_train} train / {n_val} val ...")
    gen_split(n_train, "train")
    gen_split(n_val, "val")
    write_data_yaml()
    print(f"完成 -> {DATASET}")
