# LoL 小地圖偵測與職業上路選手操作量化(YOLO11)

> **發想出處**:YouTube 頻道**圓某人和四某人**的影片[《硬核分析Keria的天賦有多高?用數據告訴你為什麼Keria是歷天怪》](https://www.youtube.com/watch?v=wQOFX4SB6Go)用模型演算法量化了輔助選手 Keria 的操作上限。看完這個測試後,我想把同一套的想法應用到上路——最吃個人操作上限的位置，指標定義對齊該影片同樣引用的 CHI'24 延世大學論文,分析對象是兩位頂尖上路:Doran(T1)與 Bin(BLG)**。

- 以程式合成訓練資料訓練 YOLO11,合成驗證集 mAP50 ≈ 0.99,直接泛化到真實轉播畫面
- 已分析 **9 場職業比賽**(Worlds 2025 決賽 BO5、Worlds 2025 瑞士輪 R5、MSI 2026 八強 G5),共 **183,408 筆**英雄座標(2 fps)
- 每場比賽獨立訓練 10 類模型(`run_all_games.sh` 全自動批次),分析前以「瞬移剔除」清洗誤判

## Pipeline

```
比賽 VOD ─→ 抽幀 ─→ 裁切小地圖 ─→ YOLO11 偵測(每場獨立 10 類模型)
                                        │
   合成訓練資料(免標註)──訓練──────────┘
                                        ▼
                  (英雄, 隊伍, x, y, t) → positions_<場次>.csv
                                        │
                     瞬移剔除 → 熱區圖 + 走位量化指標 + 系列彙總
```

## 核心方法:合成資料 + sim2real

1. **乾淨底圖**:對整場影片的小地圖做中位數背景萃取,自動去除英雄圖示與特效(`03`)
2. **英雄圖示**:從 Riot Data Dragon 抓頭像,程式化做成圓形遮罩 + 隊伍色環(`04`)
3. **合成訓練集**:隨機貼 3~10 隻英雄(可重疊模擬團戰),同步輸出 YOLO 標註;疊加亮度/色相抖動、戰爭迷霧、觀戰視野框、壓縮模糊、守衛雜訊點(`05`)

| 合成訓練資料 | 真實轉播幀偵測(模型只看過合成資料)|
|---|---|
| ![synth](docs/synth_sample.png) | ![real](docs/check_real.png) |

**sim2real 迭代紀錄**:(1) 首版合成太乾淨,真實畫面幾乎抓不到 → 取樣真實環色、加粗色環、壓縮模糊後泛化成功;(2) 後期戰鬥煙霧與守衛雜訊使 recall 驟降 → 加入暗亮度、濃霧、雜訊點重訓解決;(3) 58 類聯集模型在真實畫面混淆嚴重(每幀僅 3.2 隻)→ 改為每場獨立 10 類模型(每幀 7~10 隻)。

## 指標定義(位置版,對齊 CHI'24 論文精神)

- **走位轉向角**:連續移動向量的夾角平均,越高=方向變化越多、越難預測(論文 Positioning 指標的位置版;僅取連續取樣的三點)
- **地圖覆蓋率**:12×12 網格中走過的格子比例
- **路徑長/分**:連續取樣段的平均移動速度(不受偵測率差異影響)
- **校準**:每場對同場 10 名選手算同一指標後排名,再對系列做樣本加權彙總

## 成果:Doran vs Bin

### Worlds 2025 各自的最終系列賽

Doran:**冠軍戰 BO5**(T1 3-2 KT,2025-11-09,奪冠);Bin:**淘汰戰**(TES 2-1 BLG,瑞士輪 R5,2025-10-25,出局)。

| | Doran(5 場)| Bin(3 場)|
|---|---|---|
| 走位轉向角(加權)| **21.7°** | 11.8° |
| 轉向角平均排名 | **#6.0** / 10 | #8.0 / 10 |
| 地圖覆蓋率(加權)| **70.6%** | 55.4% |
| 路徑長/分(加權)| **2.33** | 1.47 |
| 偵測樣本 | 10,822 | 7,805 |

逐場數據:

| 場次 | 英雄 | 轉向角(排名)| 覆蓋率 | 路徑長/分 | 樣本 |
|---|---|---|---|---|---|
| 決賽 G1 | Ambessa | 32.3°(#4)| 25.0%* | 0.42* | 506* |
| 決賽 G2 | Sion | 39.4°(#7)| 79.9% | 4.55 | 2,791 |
| 決賽 G3 | Renekton | 11.8°(#9)| 71.5% | 1.40 | 2,775 |
| 決賽 G4 | Gragas | 14.1°(#7)| 53.5% | 1.41 | 1,531 |
| 決賽 G5 | Camille | 17.0°(#3)| 77.1% | 1.95 | 3,219 |
| 瑞士輪 G1 | Ornn | 9.0°(#10)| 57.6% | 1.13 | 2,142 |
| 瑞士輪 G2 | Renekton | 17.2°(#4)| 52.1% | 2.10 | 2,382 |
| 瑞士輪 G3 | Jax | 9.7°(#10)| 56.2% | 1.25 | 3,281 |

\* G1 的 Ambessa 偵測率偏低(新英雄頭像偏暗),該場數據僅供參考。

| Doran 系列疊加熱區 | Bin 系列疊加熱區 |
|---|---|
| ![doran](docs/heatmap_Doran_series.png) | ![bin](docs/heatmap_Bin_series.png) |

### MSI 2026 八強 G5:同場直接對位

同一場比賽的頭對頭(T1 vs BLG 決勝局,2026-07-04,BLG 勝):

| 指標 | Doran(Gnar)| Bin(Rumble)|
|---|---|---|
| 走位轉向角(同場排名)| 9.2°(#10)| **16.8°(#7)**|
| 地圖覆蓋率 | 56.9% | **63.9%** |
| 活動量(路徑長)| 45.0 | **92.4** |

![duel](docs/heatmap_doran_vs_bin.png)

### 解讀與侷限

- 同場對決(MSI)Bin 全面壓過 Doran,與該場戰果一致;但在各自 Worlds 2025 的最終系列賽,Doran 的活動指標更高——其一,對手與比賽節奏不同(奪冠隊 vs 被淘汰隊);其二,**英雄選擇影響大**(Doran 拿到 Sion/Camille 等高機動英雄,Bin 多為 Ornn/Jax 前排)。位置指標量的是「這場比賽中的走位輸出」,不等於選手絕對實力。
- 嚴謹的比較是**同場排名**;跨場/跨系列的絕對值僅供參考。要做選手母體校準需要更多場次。

## 快速開始

```powershell
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
# 先依 https://pytorch.org/get-started/locally/ 裝對應 CUDA 的 PyTorch
pip install -r requirements.txt

# 單場流程(config.yaml 已設為 Worlds 2025 決賽 G5)
python scripts/01_download_vod.py "<VOD 網址>"
python scripts/02_grab_frame.py      # 抽幀、校正小地圖座標
python scripts/03_make_background.py # 中位數背景萃取
python scripts/04_fetch_icons.py     # 抓英雄頭像
python scripts/05_gen_synthetic.py   # 合成訓練資料
python scripts/06_train.py yolo11s.pt 80
python scripts/07_detect.py          # → outputs/positions.csv
python scripts/09_analyze.py         # 熱區圖
python scripts/10_skill_index.py     # 同場排名與對位比較

# 多場批次(場次設定在 games/*.yaml)
bash scripts/run_all_games.sh        # 逐場 合成→訓練→偵測(可中斷續跑)
python scripts/11_bo5_compare.py     # 系列彙總:Doran vs Bin
```

已附上 9 場的偵測結果(`outputs/positions_*.csv`)與示範模型 `models/minimap.pt`(決賽 G5),可直接跑 `09`/`10`/`11` 分析。

## 專案結構

```
lol-minimap-yolo/
├─ config.yaml               # 單場示範設定(英雄、小地圖座標、隊伍、對位)
├─ games/*.yaml              # Worlds 2025 八場的場次設定
├─ scripts/01~11 + run_all_games.sh
├─ models/minimap.pt         # 示範權重(各場權重由批次腳本重建)
├─ outputs/positions_*.csv   # 9 場偵測結果
└─ docs/                     # README 成果圖
```

## 技術棧

Python · PyTorch (CUDA) · Ultralytics YOLO11 · OpenCV · NumPy · yt-dlp

## 參考資料

- 發想出處:圓某人和四某人,[《硬核分析Keria的天賦有多高?》](https://www.youtube.com/watch?v=wQOFX4SB6Go)(題目與思路來源;pipeline 與程式為獨立實作)
- Lee et al., *Characterizing and Quantifying Expert Input Behavior in League of Legends*, CHI 2024(指標定義;[資料集](https://github.com/narey54541/LLL))
- [DeepLeague](https://github.com/farzaa/DeepLeague)、[LeagueAI](https://github.com/Oleffa/LeagueAI) — 小地圖偵測與合成資料先行研究
- 比賽數據:[gol.gg](https://gol.gg/)(選角/勝負);畫面來源:Worlds 2025 與 MSI 2026 賽事 VOD(僅供個人研究)

## 聲明

本專案為個人研究/學習用途。英雄頭像來自 Riot Data Dragon;賽事畫面截圖僅作方法示意,版權屬原轉播方,請勿散布抽取的完整畫面。League of Legends © Riot Games — 本專案與 Riot Games 無關。
