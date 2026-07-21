#!/bin/bash
# 逐場:合成資料 -> 訓練 10 類模型 -> 全場偵測。可中斷續跑(已完成的場次自動跳過)。
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe

declare -A VOD=(
  [wf25_g1]="KT vs T1 - Game 1"
  [wf25_g2]="KT vs T1 - Game 2"
  [wf25_g3]="KT vs T1 - Game 3"
  [wf25_g4]="KT vs T1 - Game 4"
  [wf25_g5]="KT vs T1 - Game 5"
  [tb25_g1]="TES vs BLG - Game 1"
  [tb25_g2]="TES vs BLG - Game 2"
  [tb25_g3]="TES vs BLG - Game 3"
  [msi26_g1]="T1 vs BLG - Game 1 ｜ Round 1 LoL MSI"
  [msi26_g2]="T1 vs BLG - Game 2 ｜ Round 1 LoL MSI"
  [msi26_g3]="T1 vs BLG - Game 3 ｜ Round 1 LoL MSI"
  [msi26_g4]="T1 vs BLG - Game 4 ｜ Round 1 LoL MSI"
  [msi26_g5]="T1 vs BLG - Game 5 ｜ Round 1 LoL MSI"
)

for k in wf25_g1 wf25_g2 wf25_g3 wf25_g4 wf25_g5 tb25_g1 tb25_g2 tb25_g3 msi26_g1 msi26_g2 msi26_g3 msi26_g4 msi26_g5; do
  if [ -f "outputs/positions_$k.csv" ]; then echo "SKIP $k (done)"; continue; fi
  vod=$(ls data/vods/"${VOD[$k]}"*.mp4 | head -1)
  echo "=== $k  ($vod) ==="
  if [ ! -f "models/minimap_$k.pt" ]; then
    cp "games/$k.yaml" config.yaml
    "$PY" scripts/03_make_background.py "$vod" 2>&1 | tail -1  # 每場用自己的乾淨底圖(版面/版本可能不同)
    rm -rf data/dataset
    "$PY" scripts/05_gen_synthetic.py 2>&1 | tail -1
    "$PY" scripts/06_train.py yolo11s.pt 80 2>&1 | tail -2
    cp models/minimap.pt "models/minimap_$k.pt"
  fi
  MM_CONFIG="games/$k.yaml" "$PY" scripts/07_detect.py "$vod" \
      --model "models/minimap_$k.pt" --out "outputs/positions_$k.csv" 2>&1 | tail -1
  echo "=== $k DONE ==="
done
echo ALL_GAMES_DONE
