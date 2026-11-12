# -*- coding: utf-8 -*-
"""
code to plot RL-controller flight log csvs (as written by write_to_csv.py)

usage: uv run src/flapper_data/utils/plot_rl.py ../rl_data/pre_flapping/rlearning00.csv [more.csv ...]

signals are converted to the sim frame used by flapper_rl (NED, yaw sign-flipped,
gyro y/z swapped, nnOut3 sign-flipped) so the plots compare directly with the sim.
gate crossings come from rlapp.tgtGate when the log has it, otherwise they are
inferred by replaying the firmware's gate check on the fig8 track. the window where
the NN is in control is inferred from the motor PWM update rate (RL task ~200 Hz,
PID hover loop ~500 Hz), so it is only shown for logs sampled fast enough.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy.signal import butter, filtfilt

CUTOFF = 10.0  # Hz, low-pass applied to the plotted signals
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#52514e"
MUTED = "#898781"
SHADE = "#e1e0d9"
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
    "axes.labelcolor": INK,
    "axes.titlecolor": "#0b0b0b",
    "axes.titleweight": "bold",
    "axes.titlesize": 11,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "legend.labelcolor": INK,
    "lines.linewidth": 1.5,
})

# fig8 track (flapper_rl/configs/tracks/fig8.yaml), each gate is [x, y, yaw_deg] in NED
GATES = np.array([
    [1.5, -1.5, 90.0],
    [0.0, 0.0, 180.0],
    [-1.5, 1.5, 90.0],
    [0.0, 3.0, 0.0],
    [1.5, 1.5, -90.0],
    [0.0, 0.0, -180.0],
    [-1.5, -1.5, -90.0],
    [0.0, -3.0, 0.0],
])

RATE_WINDOW = 0.1  # s, window used to count motor PWM updates
RL_RATE = (100.0, 350.0)  # Hz, motor update rate while the NN is in control
MIN_FS_FOR_NN = 400.0  # Hz, slower logs can't tell 200 Hz from 500 Hz motor updates


def lowpass(signals, cutoff, fs):
    # zero-phase, so the filtered signals are not shifted in time
    b, a = butter(4, cutoff, fs=fs)
    return pd.DataFrame(filtfilt(b, a, signals, axis=0), columns=signals.columns, index=signals.index)


def to_sim_frame(data):
    # firmware ENU -> sim NED, same conventions as flapper_rl.eval.flight_log.load_usd_log
    # angles stay in deg (unwrapped so the filter and the plot don't jump at +-180)
    if "stateEstimate.vx" in data:
        vx = data["stateEstimate.vx"]
    else:
        # newer logs replace vx with kalman.statePX/PY, the velocity in the heading frame
        psi = np.deg2rad(data["stateEstimate.yaw"])
        vx = np.cos(psi) * data["kalman.statePX"] - np.sin(psi) * data["kalman.statePY"]
    sim = pd.DataFrame({
        "x": data["stateEstimate.x"],
        "y": -data["stateEstimate.y"],
        "z": -data["stateEstimate.z"],
        "vx": vx,
        "vy": -data["stateEstimate.vy"],
        "vz": -data["stateEstimate.vz"],
        "roll": np.unwrap(data["stateEstimate.roll"], period=360),
        "pitch": np.unwrap(data["stateEstimate.pitch"], period=360),
        "yaw": -np.unwrap(data["stateEstimate.yaw"], period=360),
        "p": data["gyro.x"],
        "q": -data["gyro.z"],
        "r": -data["gyro.y"],
    })
    if "rlapp.nnOut0" in data:
        for i in range(4):
            sim[f"cmd{i}"] = data[f"rlapp.nnOut{i}"].clip(-1, 1)
        sim["cmd3"] = -sim["cmd3"]
    else:
        for i, k in enumerate(("roll", "pitch", "yaw")):
            sim[f"cmd{i}"] = data[f"controller.cmd_{k}"]
    motor = "motor.m{}" if "motor.m1" in data else "rlapp.pwmM{}"
    for i in range(1, 5):
        sim[f"m{i}"] = data[motor.format(i)]
    return sim


def body_velocity(sim):
    # inertial NED velocity rotated into the sim body frame (fwd, right, down), ZYX euler
    phi, theta, psi = (np.deg2rad(sim[k]) for k in ("roll", "pitch", "yaw"))
    cr, sr = np.cos(phi), np.sin(phi)
    cp, sp = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi), np.sin(psi)
    vx, vy, vz = sim["vx"], sim["vy"], sim["vz"]
    u = cy * cp * vx + sy * cp * vy - sp * vz
    v = (cy * sp * sr - sy * cr) * vx + (sy * sp * sr + cy * cr) * vy + cp * sr * vz
    w = (cy * sp * cr + sy * sr) * vx + (sy * sp * cr - cy * sr) * vy + cp * cr * vz
    return u, v, w


def nn_window(t, motor):
    # longest stretch where the motor PWM changes at the RL rate, as (t_start, t_end)
    changed = np.r_[False, np.diff(motor) != 0]
    count = np.cumsum(changed)
    lo = np.searchsorted(t, t - RATE_WINDOW, side="left")
    rate = (count - count[lo] + changed[lo]) / RATE_WINDOW
    rl = (rate >= RL_RATE[0]) & (rate <= RL_RATE[1])
    if not rl.any():
        return None
    edges = np.flatnonzero(np.diff(np.r_[0, rl.astype(int), 0]))
    starts, ends = edges[::2], edges[1::2] - 1
    best = np.argmax(ends - starts)
    return max(t[starts[best]] - RATE_WINDOW, 0.0), t[ends[best]]


def gate_crossings(t, data, sim):
    # (time, gate that was crossed)
    if "rlapp.tgtGate" in data:
        # the firmware logged its target gate, it advances on every crossing
        gate = data["rlapp.tgtGate"].to_numpy()
        return [(t[k], gate[k - 1]) for k in np.flatnonzero(np.diff(gate)) + 1]
    # otherwise replay the firmware's gate check on the fig8 track: the target gate
    # advances when the drone crosses the plane through the gate along its normal
    x, y = sim["x"].to_numpy(), sim["y"].to_numpy()
    crossings, cur = [], 0
    for k in range(1, len(x)):
        gx, gy, yaw = GATES[cur]
        nx, ny = np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))
        if (x[k - 1] - gx) * nx + (y[k - 1] - gy) * ny < 0 <= (x[k] - gx) * nx + (y[k] - gy) * ny:
            crossings.append((t[k], cur))
            cur = (cur + 1) % len(GATES)
    return crossings


def panels(sim, nn_commands):
    if nn_commands:
        commands = ("NN commands", "[-1, 1]",
                    dict(zip(("fL (m2)", "fR (m4)", "Γ dih (m1)", "Λ yaw (m3)"), (sim[f"cmd{i}"] for i in range(4)))))
    else:
        commands = ("Controller commands (firmware)", "",
                    {k: sim[f"cmd{i}"] for i, k in enumerate(("roll", "pitch", "yaw"))})
    return [
        ("Position", "m", {"x": sim["x"], "y": sim["y"], "alt (-z)": -sim["z"]}),
        ("Altitude", "m", {"alt": -sim["z"]}),
        ("Body velocity (sim frame)", "m/s", {"u fwd": sim["u"], "v right": sim["v"], "w down": sim["w"]}),
        ("Speed", "m/s", {"speed": sim["speed"]}),
        ("Attitude (sim euler)", "deg", {k: sim[k] for k in ("roll", "pitch", "yaw")}),
        ("Body rates", "deg/s", {k: sim[k] for k in "pqr"}),
        commands,
        ("Motor output", "PWM", {f"m{i}": sim[f"m{i}"] for i in range(1, 5)}),
    ]


def legend(ax, ncols):
    # above the axes on the right, level with the left-aligned title
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1), ncols=ncols, frameon=False, fontsize=8,
              handlelength=1.2, columnspacing=1.0)


def plot_trajectory(ax, sim, t, nn):
    names = {}
    for i, (gx, gy, yaw) in enumerate(GATES):
        # gate drawn as a 2 m bar across its normal
        tx, ty = -np.sin(np.deg2rad(yaw)), np.cos(np.deg2rad(yaw))
        ax.plot([gx - tx, gx + tx], [gy - ty, gy + ty], color=INK, linewidth=2.5, solid_capstyle="round",
                label="gates" if i == 0 else None)
        names.setdefault((gx, gy), []).append(str(i))
    for (gx, gy), gate_names in names.items():
        # gates 1 and 5 share the centre of the track
        ax.annotate("/".join(gate_names), (gx, gy), textcoords="offset points", xytext=(6, 6), fontsize=8, color=INK)
    if nn is None:
        ax.plot(sim["x"], sim["y"], color=COLORS[0])
    else:
        active = (t >= nn[0]) & (t <= nn[1])
        ax.plot(sim["x"], sim["y"], color=MUTED, linewidth=1, label="PID")
        ax.plot(sim["x"].where(active), sim["y"].where(active), color=COLORS[0], label="NN")
    ax.plot(sim["x"].iloc[0], sim["y"].iloc[0], "o", color=COLORS[2], markersize=8, label="start")
    ax.plot(sim["x"].iloc[-1], sim["y"].iloc[-1], "X", color=COLORS[1], markersize=9, label="end")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Trajectory, top view (NED)", loc="left")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    legend(ax, ncols=5)


def mark_events(ax, nn, gates, label_gates=False):
    if nn is not None:
        ax.axvspan(*nn, color=SHADE, alpha=0.6, linewidth=0, zorder=0)
    for j, (tc, gate) in enumerate(gates):
        ax.axvline(tc, color=MUTED, linestyle="--", linewidth=0.8)
        if label_gates:
            # staggered over three rows so close crossings don't overprint
            ax.text(tc, 1 - 0.06 * (j % 3), f" G{gate}", transform=ax.get_xaxis_transform(), va="top", fontsize=7,
                    color=MUTED)


def plot_file(path):
    data = pd.read_csv(path)
    t = ((data["timestamp"] - data["timestamp"].iloc[0]) / 1000).to_numpy()
    fs = 1 / np.median(np.diff(t))
    sim = to_sim_frame(data)
    # events come from the raw log, the filter would smear the motor updates and gate crossings
    nn = nn_window(t, sim["m1"].to_numpy()) if fs >= MIN_FS_FOR_NN else None
    gates = gate_crossings(t, data, sim)
    sim = lowpass(sim, CUTOFF, fs=fs)
    sim["u"], sim["v"], sim["w"] = body_velocity(sim)
    sim["speed"] = np.linalg.norm(sim[["vx", "vy", "vz"]], axis=1)

    fig, axes = plt.subplots(3, 3, figsize=(18.5, 9.6), layout="constrained")
    axes = axes.ravel()
    plot_trajectory(axes[0], sim, t, nn)
    for i, (ax, (title, unit, series)) in enumerate(zip(axes[1:], panels(sim, "rlapp.nnOut0" in data)), start=1):
        if i > 1:
            ax.sharex(axes[1])
        for (label, values), color in zip(series.items(), COLORS):
            ax.plot(t, values, color=color, label=label)
        if len(series) == 1:
            ax.fill_between(t, 0, next(iter(series.values())), color=COLORS[0], alpha=0.12, linewidth=0)
        mark_events(ax, nn, gates, label_gates=(i == 1))
        ax.set_title(title, loc="left")
        ax.set_ylabel(unit)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        if len(series) > 1:
            legend(ax, ncols=len(series))
        if i < 6:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("time since log start (s)")

    title = path.name
    if nn is not None:
        title += f"   ·   shaded: NN in control {nn[0]:.1f}–{nn[1]:.1f} s"
    title += f"   ·   dashed: {len(gates)} gate crossing{'' if len(gates) == 1 else 's'}"
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path, help="log csv files to plot")
    args = parser.parse_args()
    for path in args.files:
        plot_file(path)
    plt.show()


if __name__ == "__main__":
    main()
