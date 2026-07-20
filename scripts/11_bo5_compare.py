"""跨系列賽彙總比較兩位上路選手(每場先算同場排名,再做系列彙總)。

系列定義在 SERIES:每場對應 games/<key>.yaml 與 outputs/positions_<key>.csv。
每場計算目標選手的:轉向角(含同場 10 人排名)、地圖覆蓋率、游走比例、
每分鐘路徑長(以該場偵測時間跨度正規化,樣本數不同時較可比)。
系列彙總:排名取平均,其餘以樣本數加權平均。
另輸出每位選手整個系列疊加的走位熱區圖 docs/heatmap_<player>_series.png。

用法:
    python scripts/11_bo5_compare.py
"""
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
SERIES = {
    "Doran": ["wf25_g1", "wf25_g2", "wf25_g3", "wf25_g4", "wf25_g5"],
    "Bin": ["tb25_g1", "tb25_g2", "tb25_g3"],
}
D_MIN = 0.012
SMOOTH = 3
HM_SIZE = 480


def load_csv(path):
    by_champ = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_champ[r["champion"]].append((int(r["frame"]), float(r["time_s"]),
                                            float(r["x"]), float(r["y"])))
    for ch in by_champ:
        by_champ[ch].sort()
    return by_champ


JUMP = 0.08     # 1 秒內不可能的位移(正規化);孤立跳點視為誤判
DT_OK = 1.01    # 視為「連續取樣」的最大時間間隔(秒)


def clean_track(track):
    """剔除孤立跳點:與前後鄰居都距離 > JUMP 的樣本(真 TP 會停留,不會被刪)。"""
    for _ in range(2):
        keep = []
        for i, (fr, t, x, y) in enumerate(track):
            prev = track[i - 1] if i > 0 else None
            nxt = track[i + 1] if i < len(track) - 1 else None
            d_prev = np.hypot(x - prev[2], y - prev[3]) if prev else 0
            d_next = np.hypot(x - nxt[2], y - nxt[3]) if nxt else 0
            if prev and nxt and d_prev > JUMP and d_next > JUMP:
                continue
            keep.append((fr, t, x, y))
        track = keep
    return track


def smooth(xy):
    if len(xy) < SMOOTH:
        return xy
    k = np.ones(SMOOTH) / SMOOTH
    return np.column_stack([np.convolve(xy[:, 0], k, "same"), np.convolve(xy[:, 1], k, "same")])


def turning_angle(track):
    xy = smooth(np.array([(x, y) for _, _, x, y in track])) if track else np.zeros((0, 2))
    ts = [t for _, t, _, _ in track]
    angles = []
    for i in range(1, len(xy) - 1):
        if ts[i] - ts[i - 1] > DT_OK or ts[i + 1] - ts[i] > DT_OK:
            continue  # 取樣中斷處不算轉向
        v1, v2 = xy[i] - xy[i - 1], xy[i + 1] - xy[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > D_MIN and n2 > D_MIN:
            cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
            angles.append(np.degrees(np.arccos(cos)))
    return float(np.mean(angles)) if angles else float("nan")


def coverage(track, grid=12):
    cells = {(min(grid - 1, int(x * grid)), min(grid - 1, int(y * grid))) for _, _, x, y in track}
    return len(cells) / (grid * grid) * 100


def path_per_min(track):
    """連續取樣段的平均移動速度(正規化距離/分),不受偵測率差異影響。"""
    dist = dur = 0.0
    for i in range(1, len(track)):
        dt = track[i][1] - track[i - 1][1]
        if dt <= DT_OK:
            dist += float(np.hypot(track[i][2] - track[i - 1][2], track[i][3] - track[i - 1][3]))
            dur += dt
    return dist / (dur / 60) if dur > 0 else float("nan")


def analyze_game(key, target_player):
    cfg = yaml.safe_load((ROOT / "games" / f"{key}.yaml").read_text(encoding="utf-8"))
    by_champ = load_csv(ROOT / "outputs" / f"positions_{key}.csv")
    by_champ = {c: clean_track(tr) for c, tr in by_champ.items()}
    team = next(t for t, p in cfg["duel_player"].items() if p == target_player)
    champ = cfg["duel_champ"][team]
    # 同場 10 人轉向角排名
    angles = {c: turning_angle(by_champ.get(c, [])) for c in cfg["champions"]}
    ranked = sorted([c for c in angles if not np.isnan(angles[c])], key=lambda c: -angles[c])
    rank = ranked.index(champ) + 1 if champ in ranked else None
    tr = by_champ.get(champ, [])
    return dict(key=key, champ=champ, n=len(tr), angle=angles.get(champ, float("nan")),
                rank=rank, n_ranked=len(ranked), cov=coverage(tr), ppm=path_per_min(tr),
                pts=[(x, y) for _, _, x, y in tr])


def heatmap(points, bg):
    acc = np.zeros((HM_SIZE, HM_SIZE), np.float32)
    for x, y in points:
        px, py = int(x * HM_SIZE), int(y * HM_SIZE)
        if 0 <= px < HM_SIZE and 0 <= py < HM_SIZE:
            acc[py, px] += 1
    acc = cv2.GaussianBlur(acc, (0, 0), sigmaX=HM_SIZE * 0.025)
    if acc.max() > 0:
        acc /= acc.max()
    cmap = cv2.applyColorMap((acc * 255).astype(np.uint8), cv2.COLORMAP_JET)
    mask = (acc > 0.06)[..., None]
    return np.where(mask, cv2.addWeighted(bg, 0.45, cmap, 0.55, 0), bg).astype(np.uint8)


def main():
    bg = cv2.resize(cv2.imread(str(ROOT / "data" / "assets" / "minimap_background.png")),
                    (HM_SIZE, HM_SIZE))
    summary = {}
    for player, keys in SERIES.items():
        print(f"\n================  {player} — 每場數據  ================")
        rows, all_pts = [], []
        for k in keys:
            g = analyze_game(k, player)
            rows.append(g)
            all_pts += g["pts"]
            print(f"  {k}  {g['champ']:<12} 轉向角 {g['angle']:5.1f}°"
                  f"(#{g['rank']}/{g['n_ranked']})  覆蓋 {g['cov']:4.1f}%"
                  f"  路徑長/分 {g['ppm']:5.2f}  樣本 {g['n']}")
        w = np.array([g["n"] for g in rows], dtype=float)
        agg = dict(
            angle=float(np.average([g["angle"] for g in rows], weights=w)),
            rank=float(np.mean([g["rank"] for g in rows if g["rank"]])),
            cov=float(np.average([g["cov"] for g in rows], weights=w)),
            ppm=float(np.average([g["ppm"] for g in rows], weights=w)),
            n=int(w.sum()), games=len(rows),
        )
        summary[player] = agg
        hm = heatmap(all_pts, bg)
        cv2.putText(hm, f"{player} ({agg['games']} games)", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out = ROOT / "docs" / f"heatmap_{player}_series.png"
        cv2.imwrite(str(out), hm)

    print("\n================  系列彙總(樣本加權)  ================")
    print(f"{'':14}{'轉向角':>8}{'平均排名':>10}{'覆蓋率':>8}{'路徑長/分':>10}{'樣本':>8}")
    for p, a in summary.items():
        print(f"  {p:<12}{a['angle']:7.1f}°{a['rank']:9.1f}{a['cov']:7.1f}%{a['ppm']:9.2f}{a['n']:8d}")


if __name__ == "__main__":
    main()
