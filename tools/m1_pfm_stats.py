#!/usr/bin/env python3
"""Print unclamped M1 film statistics; --roi uses top-left image coordinates.

This inspects saved GPU output. It does not render or claim MIS convergence.
Python standard library only.
"""
import argparse
from array import array
import json
import math
from pathlib import Path
import sys


def statistics(path, roi):
    with path.open("rb") as stream:
        def header():
            while True:
                line = stream.readline()
                if not line:
                    raise ValueError("truncated PFM header")
                if line.strip() and not line.startswith(b"#"):
                    return line.strip()
        if header() != b"PF":
            raise ValueError("expected RGB PFM (PF)")
        width, height = map(int, header().split())
        scale = float(header())
        if width <= 0 or height <= 0 or not math.isfinite(scale) or scale == 0:
            raise ValueError("invalid PFM dimensions/scale")
        payload = stream.read()
    if len(payload) != width * height * 12:
        raise ValueError("PFM payload size does not match dimensions")
    pixels = array("f")
    if pixels.itemsize != 4:
        raise ValueError("requires 32-bit C float")
    pixels.frombytes(payload)
    if (scale < 0) != (sys.byteorder == "little"):
        pixels.byteswap()
    x, y, w, h = roi or (0, 0, width, height)
    if min(x,y) < 0 or min(w,h) <= 0 or x+w > width or y+h > height:
        raise ValueError("ROI lies outside image")
    channels = [[], [], []]
    for row in range(y, y+h):
        for col in range(x, x+w):
            offset = ((height - 1 - row) * width + col) * 3
            for channel in range(3):
                value = pixels[offset + channel] * abs(scale)
                if not math.isfinite(value):
                    raise ValueError(f"non-finite pixel at ({col},{row})")
                channels[channel].append(value)
    count = w * h
    mean = [math.fsum(values) / count for values in channels]
    result = {"file": str(path), "width": width, "height": height, "roi_top_left": [x,y,w,h],
              "pixels": count, "mean_linear_rgb": mean,
              "mean_raw_Y": math.fsum(a*b for a,b in zip(mean, [.2126390059,.7151686788,.0721923154])),
              "min_rgb": [min(v) for v in channels], "max_rgb": [max(v) for v in channels],
              "negative_component_count": sum(v < 0 for c in channels for v in c)}
    sidecar = path.with_suffix(".json")
    if sidecar.is_file():
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        result["render_settings"] = {k: metadata[k] for k in
            ("total_spp", "direct_mode", "seed", "max_bounce", "rr_depth") if k in metadata}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X","Y","WIDTH","HEIGHT"))
    args = parser.parse_args()
    try:
        print(json.dumps([statistics(path, args.roi) for path in args.images], indent=2, allow_nan=False))
    except (OSError, ValueError) as error:
        parser.exit(1, "ERROR: " + str(error) + "\n")


if __name__ == "__main__":
    main()
