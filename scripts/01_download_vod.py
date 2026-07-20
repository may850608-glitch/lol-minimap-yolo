"""下載一場比賽 VOD 到 data/vods/。

用法:
    python scripts/01_download_vod.py "<YouTube網址>"

只供個人研究分析使用,請勿散布。
"""
import sys
from pathlib import Path

import yt_dlp

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "vods"


def download(url: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    opts = {
        # 只抓影像、不要聲音(小地圖分析用不到音軌,省空間也免去合併)。
        # 偏好 1080p 的 h264(avc1),OpenCV 讀取最相容;抓不到再退而求其次。
        "format": (
            "bestvideo[height<=1080][vcodec^=avc1]/"
            "bestvideo[height<=1080]/best[height<=1080]"
        ),
        "merge_output_format": "mp4",
        "outtmpl": str(OUT_DIR / "%(title).80s.%(ext)s"),
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    print(f"\n下載完成,存到:{OUT_DIR}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    download(sys.argv[1])
