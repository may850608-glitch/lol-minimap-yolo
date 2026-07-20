"""依 CHI'24 論文的標定精神,從 YOLO 位置資料計算可量化指標,並做排名校準。

論文的 8 個指標來自滑鼠/鍵盤記錄,影片轉播只拿得到位置,因此:
  - 忠實對應 1 個:Positioning Skill(位置版)= 走位軌跡「轉向角」平均。
      論文:連續右鍵向量夾角;這裡:連續位置移動向量夾角。越大=方向變化多=難預測。
  - 位置原生延伸(對輔助 Keria 有意義,非論文原指標):
      活動量(路徑長)、地圖覆蓋率、與我方 ADC 平均距離、游走比例。
校準方式比照論文:對同場 10 名選手算同一指標 → 排名,看 Keria(Poppy)落在哪。

用法:
    python scripts/10_skill_index.py
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
CHAMPS = CFG["champions"]
TEAMS = CFG["teams"]                              # [藍方, 紅方]
TEAM = {c: (TEAMS[0] if i < 5 else TEAMS[1]) for i, c in enumerate(CHAMPS)}
ADC = CFG["adc_champ"]                            # 各隊 ADC
SUPPORT = CFG["support_champ"]                    # 各隊輔助
PLAYER = CFG["support_player"]                    # 輔助選手名(顯示用)
CSV = ROOT / "outputs" / "positions.csv"

D_MIN = 0.012   # 視為「有移動」的最小步距(正規化);過濾靜止時的偵測抖動
SMOOTH = 3      # 軌跡平滑視窗


def load():
    by_champ = defaultdict(list)          # champ -> [(frame, t, x, y)]
    by_frame = defaultdict(dict)          # frame -> {champ: (x,y)}
    with CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fr = int(r["frame"]); ch = r["champion"]
            x, y = float(r["x"]), float(r["y"])
            by_champ[ch].append((fr, float(r["time_s"]), x, y))
            by_frame[fr][ch] = (x, y)
    for ch in by_champ:
        by_champ[ch].sort()
    return by_champ, by_frame


def smooth(xy):
    if len(xy) < SMOOTH:
        return xy
    k = np.ones(SMOOTH) / SMOOTH
    xs = np.convolve(xy[:, 0], k, mode="same")
    ys = np.convolve(xy[:, 1], k, mode="same")
    return np.column_stack([xs, ys])


def positioning_index(track):
    """走位轉向角平均(度)。論文 Positioning 指標的位置版。"""
    xy = smooth(np.array([(x, y) for _, _, x, y in track]))
    angles = []
    i = 1
    while i < len(xy) - 1:
        v1 = xy[i] - xy[i - 1]
        v2 = xy[i + 1] - xy[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > D_MIN and n2 > D_MIN:
            cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
            angles.append(np.degrees(np.arccos(cos)))
        i += 1
    return float(np.mean(angles)) if angles else float("nan"), len(angles)


def path_length(track):
    xy = np.array([(x, y) for _, _, x, y in track])
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))


def map_coverage(track, grid=12):
    cells = {(min(grid - 1, int(x * grid)), min(grid - 1, int(y * grid))) for _, _, x, y in track}
    return len(cells) / (grid * grid) * 100


def dist_to_adc(champ, by_frame):
    team = TEAM[champ]; adc = ADC[team]
    ds = []
    for fr, pos in by_frame.items():
        if champ in pos and adc in pos and champ != adc:
            d = np.hypot(*(np.array(pos[champ]) - np.array(pos[adc])))
            ds.append(d)
    if not ds:
        return float("nan"), float("nan")
    ds = np.array(ds)
    roam = float((ds > 0.22).mean() * 100)        # 距 ADC 較遠的比例 ~ 游走
    return float(ds.mean()), roam


def main():
    by_champ, by_frame = load()

    print("\n================  位置版 Positioning 指標(轉向角,越高=走位方向變化越多)================")
    rows = []
    for ch in CHAMPS:
        idx, n = positioning_index(by_champ.get(ch, []))
        rows.append((ch, TEAM[ch], idx, n))
    focus = CFG.get("focus_champion")
    rows_sorted = sorted([r for r in rows if not np.isnan(r[2])], key=lambda r: -r[2])
    for rank, (ch, tm, idx, n) in enumerate(rows_sorted, 1):
        star = f"  ← {PLAYER[TEAM[ch]]}" if ch == focus else ""
        print(f"  #{rank}  {ch:<10}({tm:<3}) {idx:5.1f}°   (n={n}){star}")

    duel = " vs ".join(f"{PLAYER[t]}({SUPPORT[t]})" for t in TEAMS)
    print(f"\n================  輔助對位:{duel}================")
    for team in TEAMS:
        sp = SUPPORT[team]
        track = by_champ.get(sp, [])
        idx, _ = positioning_index(track)
        cov = map_coverage(track) if track else float("nan")
        plen = path_length(track) if track else float("nan")
        md, roam = dist_to_adc(sp, by_frame)
        who = PLAYER[team]
        print(f"\n  {who} - {sp} ({team})  樣本 {len(track)}")
        print(f"    Positioning 轉向角 : {idx:5.1f}°")
        print(f"    地圖覆蓋率         : {cov:5.1f}%")
        print(f"    活動量(路徑長)   : {plen:5.2f}")
        print(f"    與 ADC 平均距離    : {md:5.3f}")
        print(f"    游走比例(離 ADC 遠): {roam:5.1f}%")

    print("\n說明:這些是位置版/原生指標,非論文原始 8 指標(那些需滑鼠鍵盤記錄)。")
    print("校準目前是『同場 10 人相對排名』;要做段位母體校準需處理更多場比賽。")


if __name__ == "__main__":
    main()
