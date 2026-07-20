"""下載 config.yaml 指定英雄的官方方形頭像 (Data Dragon)。

存到 data/assets/icons/<ChampId>.png (120x120)。
之後合成時會做圓形遮罩 + 加上隊伍色環,模擬小地圖上的英雄圖示。

用法:
    python scripts/04_fetch_icons.py
"""
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
VER = CFG["ddragon_version"]
CHAMPS = CFG["champions"]

OUT = ROOT / "data" / "assets" / "icons"


def fetch() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for c in CHAMPS:
        url = f"https://ddragon.leagueoflegends.com/cdn/{VER}/img/champion/{c}.png"
        dest = OUT / f"{c}.png"
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"  OK  {c}")
        except Exception as e:
            print(f"  FAIL {c}: {e}  ({url})")
    print(f"完成 -> {OUT}")


if __name__ == "__main__":
    fetch()
