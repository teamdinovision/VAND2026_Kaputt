#!/usr/bin/env python3
"""扫描 crop 目录，找出宽度 < 阈值的图像，输出到 CSV。"""

import csv
import os
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}


def main():
    data_root = "/data/public/dataset/kaputt2"
    crop_dir = f"{data_root}/data/test/query-data/crop/"
    output_csv = "./small_image_k2.csv"
    threshold = 200

    results = []
    for f in sorted(Path(crop_dir).rglob("*")):
        if f.suffix.lower() not in IMAGE_EXTS or not f.is_file():
            continue
        try:
            with Image.open(f) as img:
                w, _ = img.size
        except Exception:
            continue
        if w < threshold:
            capture_id = f.stem
            results.append((capture_id, w))

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["capture_id", "width"])
        writer.writerows(results)

    print(f"共 {len(results)} 张小图 (width < {threshold}) -> {output_csv}")


if __name__ == "__main__":
    main()
