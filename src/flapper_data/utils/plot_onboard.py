# -*- coding: utf-8 -*-
"""
code to plot all signals of onboard log csvs (as written by write_to_csv.py)

usage: uv run src/flapper_data/utils/plot_onboard.py ../log_rpm_data/onboard/some_log.csv [more.csv ...]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy.signal import butter, filtfilt

CUTOFF = 10.0  # Hz, low-pass applied to the plotted signals
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.6,
    "axes.edgecolor": "#c3c2b7",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.labelcolor": "#52514e",
    "axes.titlecolor": "#0b0b0b",
    "axes.titleweight": "bold",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "legend.labelcolor": "#52514e",
    "lines.linewidth": 1.5,
})


def lowpass(signals, cutoff, fs):
    # zero-phase, so the filtered signals are not shifted in time
    b, a = butter(4, cutoff, fs=fs)
    return pd.DataFrame(filtfilt(b, a, signals, axis=0), columns=signals.columns, index=signals.index)


def panels(data):
    return [
        ("Controller attitude", "deg", {k: data[f"controller.{k}"] for k in ("roll", "pitch", "yaw")}),
        ("Thrust command", "", {"thrust": data["controller.cmd_thrust"]}),
        ("Accelerometer", "g", {k: data[f"acc.{k}"] for k in "xyz"}),
        ("Gyroscope", "deg/s", {k: data[f"gyro.{k}"] for k in "xyz"}),
        ("Mocap orientation", "quaternion", {k: data[f"locSrv.{k}"] for k in ("qx", "qy", "qz", "qw")}),
        ("Motor output", "PWM", {f"m{i}": data[f"motor.m{i}"] for i in range(1, 5)}),
    ]


def plot_file(path):
    data = pd.read_csv(path)
    t = (data["timestamp"] - data["timestamp"].iloc[0]) / 1000
    signals = lowpass(data.drop(columns="timestamp"), CUTOFF, fs=1 / t.diff().median())
    groups = panels(signals)
    fig, axes = plt.subplots(len(groups), 1, sharex=True, figsize=(18.5, 9.6), layout="constrained")
    for ax, (title, unit, series) in zip(axes, groups):
        for (label, values), color in zip(series.items(), COLORS):
            ax.plot(t, values, color=color, label=label)
        ax.set_title(title, loc="left")
        ax.set_ylabel(unit)
        ax.yaxis.set_major_locator(MaxNLocator(3))
        if len(series) > 1:
            ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1), frameon=False)
    axes[-1].set_xlabel("time since log start (s)")
    fig.suptitle(path.name, x=0.01, ha="left", fontweight="bold")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path, help="log csv files to plot")
    args = parser.parse_args()
    for path in args.files:
        plot_file(path)
    plt.show()


if __name__ == "__main__":
    main()
