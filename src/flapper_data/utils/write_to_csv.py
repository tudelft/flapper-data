# -*- coding: utf-8 -*-
"""
code to decode the uSD-card binary logs of a crazyflie to csv, one csv per log

usage: uv run src/flapper_data/utils/write_to_csv.py ../log_rpm_data/bin [--outdir ../log_rpm_data/onboard]

every binary log in the directory is decoded (files are recognised by the log format's
magic byte, whatever their name). the csv is named after the log and is skipped if it
already exists.
"""
import argparse
from pathlib import Path

import pandas as pd

from flapper_data.utils import cfusdlog

MAGIC = 0xBC  # first byte of a crazyflie uSD-card log


def is_usd_log(path):
    with open(path, "rb") as f:
        return f.read(1) == bytes([MAGIC])


def process_file(path, outdir):
    out = (outdir / path.name).with_suffix(".csv")
    if out.exists():
        print(f"skipping {path} ({out} already exists)")
        return
    log = cfusdlog.decode(path)
    if not log or "fixedFrequency" not in log:
        print(f"skipping {path} (no fixedFrequency data)")
        return
    pd.DataFrame(log["fixedFrequency"]).to_csv(out, index=False)
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser(description="decode the crazyflie uSD-card logs in a directory to csv")
    parser.add_argument("indir", type=Path, help="directory with the binary logs")
    parser.add_argument("--outdir", type=Path, help="output directory for the csv files (default: indir)")
    args = parser.parse_args()

    if not args.indir.is_dir():
        parser.error(f"{args.indir} is not a directory")
    outdir = args.outdir or args.indir
    outdir.mkdir(parents=True, exist_ok=True)

    logs = [p for p in sorted(args.indir.iterdir()) if p.is_file() and is_usd_log(p)]
    if not logs:
        print(f"no uSD-card logs in {args.indir}")
    for path in logs:
        process_file(path, outdir)


if __name__ == "__main__":
    main()
