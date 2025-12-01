"""plot_flow
=================

Utilities to load Crazyflie SD log data and OptiTrack motion‑capture
data for a given experiment and generate comparison plots.

The script focuses on experiments logged in the ``data/flowdeck_cyberzoo_XXX``
folders created by :mod:`convert_test_data`. For a selected test name it:

* loads and pre‑processes the SD log (``sd.csv``) and OptiTrack file
    (``opti.csv``),
* time‑aligns both data sources based on height (``z``) using an
    MSE‑minimising shift,
* computes Cartesian velocities from OptiTrack positions and converts
    them to the body frame using quaternions,
* converts Crazyflie estimator velocities to the body frame as well,
* and produces a selection of plots comparing estimator vs measurement
    (positions, velocities, attitude, and flow/gyro signals).

Run this script from the repository root.  Adjust ``CONFIG['test_name']``
below to select which ``data/<test_name>/`` folder to analyse.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.signal import medfilt

# ==========================
# CONFIGURATION
# ==========================
CONFIG = {
    "test_name": "flowdeck_cyberzoo_032",
    "timestamp_scale": 990,
    "smooth_windows": {
        "x": 20,
        "y": 20,
        "position": 5
    },
    "median_filter_kernel": 9,
    "velocity_smooth": {
        "vx": 10,
        "vy": 10,
        "vz": 10
    }
}

# ==========================
# DATA LOADING
# ==========================
def load_sd_data(test_name):
    """Load and pre‑process SD log data for ``test_name``.

    Parameters
    ----------
    test_name:
        Name of the experiment folder inside ``data/``, e.g.
        ``"flowdeck_cyberzoo_032"``.

    Returns
    -------
    pandas.DataFrame
        SD log signals with a normalised and scaled ``timestamp`` column.
        Initial stationary samples (before strong motion around ``gyro.y``)
        are removed.
    """
    df = pd.read_csv(f"data/{test_name}/sd.csv")
    start_index = (df['gyro.y'] > 5).idxmax() # find first index where gyro.y > 5
    df = df.loc[start_index:].reset_index(drop=True) # remove rows where drone is stationary
    df['timestamp'] = df['timestamp'] - df['timestamp'].iloc[0]  # normalize to start at 0
    df['timestamp'] /= CONFIG["timestamp_scale"]  # scale timestamps
    return df


def load_opti_data(test_name):
    """Load and pre‑process OptiTrack data for ``test_name``.

    The function looks for the rigid body named ``"Flapper"`` in the
    OptiTrack CSV header, extracts its quaternion and position columns,
    and normalises position and time to start at zero.
    """
    # Read the header row (row 4, which is at index 3) to find the columns for the "Flapper" rigid body
    header_df = pd.read_csv(f"data/{test_name}/opti.csv", skiprows=3, nrows=1, header=None)
    
    # Find columns that are "Flapper"
    flapper_cols = [i for i, col_name in enumerate(header_df.iloc[0]) if isinstance(col_name, str) and col_name == "Flapper"]
    # The columns to use are the first two (index, timestamp) and the flapper columns
    cols_to_use = [0, 1] + flapper_cols
    
    # Define the names for the columns we are loading
    col_names = ["index", "timestamp", "qx", "qy", "qz", "qw", "x", "y", "z"]

    df = pd.read_csv(
        f"data/{test_name}/opti.csv",
        skiprows=7,
        usecols=cols_to_use,
        header=None,
        index_col=0,
        names=col_names
    )
    df[["x", "y", "z"]] -= df[["x", "y", "z"]].iloc[0]
    df["timestamp"] = df["timestamp"] - df["timestamp"].iloc[0]  # normalize to start at 0
    return df

# ==========================
# TIME SYNCHRONIZATION
# ==========================
def synchronize_by_z(sd_data, opti_data):
    """Synchronise SD and OptiTrack data by matching height (z/y).

    Both signals are interpolated on a common time grid.  A range of
    time shifts is tested, and the shift with the lowest mean squared
    error between SD ``stateEstimate.z`` and OptiTrack ``y`` is chosen.
    The OptiTrack timestamps are shifted accordingly.
    """
    n_samples = 5000
    opti_data = opti_data.dropna(subset=['y'])  # ensure no NaN in z-axis

    # Create a common regular time grid
    start_time = max(sd_data['timestamp'].min(), opti_data['timestamp'].min())
    end_time = min(sd_data['timestamp'].max(), opti_data['timestamp'].max())
    common_time = np.linspace(start_time, end_time, num=n_samples)[:-500]

    # Interpolate both signals to the common time grid
    sd_z_interp = np.interp(common_time, sd_data['timestamp'], sd_data['stateEstimate.z'])
    opti_z_interp = np.interp(common_time, opti_data['timestamp'], opti_data['y'])

    # --- Find optimal lag using MSE ---
    dt = common_time[1] - common_time[0]
    max_shift = int(len(common_time) / 2)  # allow half-window shift
    lags = np.arange(-max_shift, max_shift + 1)
    mse_values = []

    for lag in lags:
        shifted_opti = np.interp(common_time, common_time + lag * dt, opti_z_interp)
        mse = np.mean((sd_z_interp - shifted_opti) ** 2)
        mse_values.append(mse)

    mse_values = np.array(mse_values)
    best_lag = lags[np.argmin(mse_values)]
    time_offset = best_lag * dt

    # --- Plot results ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot aligned signals
    shifted_opti_best = np.interp(common_time, common_time + time_offset, opti_z_interp)
    ax1.plot(common_time, sd_z_interp, label='SD Z-axis')
    ax1.plot(common_time, opti_z_interp, label='Opti Z-axis (original)')
    ax1.plot(common_time, shifted_opti_best, label=f'Opti Z-axis (shifted by {time_offset:.3f}s)', linestyle='--')
    ax1.set_ylabel('Z Position')
    ax1.set_title('Optimal alignment based on MSE')
    ax1.legend()
    ax1.grid(True)

    # Plot MSE vs lag
    lag_times = lags * dt
    ax2.plot(lag_times, mse_values, label='MSE vs Time Lag')
    ax2.axvline(time_offset, color='r', linestyle='--', label=f'Selected Offset = {time_offset:.3f}s')
    ax2.set_xlabel('Time Lag (s)')
    ax2.set_ylabel('MSE')
    ax2.legend()
    ax2.grid(True)

    plt.show()

    print(f"[Sync-MSE] Optimal time offset (s): {time_offset:.3f}")
    opti_data.loc[:, 'timestamp'] += time_offset

    return opti_data


# ==========================
# VELOCITY PROCESSING
# ==========================
def compute_filtered(df, df_time, kernel, smooth):
    """Differentiate a position series and apply median + rolling filters.

    This approximates a velocity from positions sampled at times
    ``df_time`` while removing high‑frequency noise.
    """
    raw = df.diff() / df_time.diff()
    filtered = pd.Series(medfilt(raw, kernel_size=kernel), index=df.index)
    return filtered.rolling(smooth).mean()


def rotate_velocity_to_body_frame(global_velocity, quaternion):
    """Rotate a velocity vector from world frame into body frame.

    Expects a quaternion in the convention ``[x, y, z, w]``.
    """
    global_velocity = np.asarray(global_velocity)
    quaternion = np.asarray(quaternion)
    rot = R.from_quat(quaternion)
    return rot.inv().apply(global_velocity)


def body_velocity_row(row):
    if any(pd.isna([
        row['vx_opti'], row['vy_opti'], row['vz_opti'],
        row['qz'], row['qx'], row['qy'], row['qw']
    ])):
        return [np.nan, np.nan, np.nan]
    global_vel = [row['vx_opti'], row['vy_opti'], row['vz_opti']]
    quat = [row['qz'], row['qx'], row['qy'], row['qw']]
    return rotate_velocity_to_body_frame(global_vel, quat)


def add_opti_velocities(opti_df):
    """Compute body‑frame velocities from OptiTrack position traces.

    Adds ``vx_opti``, ``vy_opti``, ``vz_opti`` (world frame, filtered
    derivatives) and ``vx_body``, ``vy_body``, ``vz_body`` (body frame)
    columns to the provided DataFrame.
    """
    opti_df['vx_opti'] = compute_filtered(
        opti_df["z"], opti_df["timestamp"], CONFIG["median_filter_kernel"], CONFIG["velocity_smooth"]["vx"])
    opti_df['vy_opti'] = compute_filtered(
        opti_df["x"], opti_df["timestamp"], CONFIG["median_filter_kernel"], CONFIG["velocity_smooth"]["vy"])
    opti_df['vz_opti'] = compute_filtered(
        opti_df["y"], opti_df["timestamp"], CONFIG["median_filter_kernel"], CONFIG["velocity_smooth"]["vz"])
    body_velocities = opti_df.apply(body_velocity_row, axis=1, result_type='expand')
    body_velocities.columns = ['vx_body', 'vy_body', 'vz_body']
    return pd.concat([opti_df, body_velocities], axis=1)


def add_sd_velocities(sd_df):
    """Augment SD log with body‑frame estimator velocities.

    Uses estimator roll/pitch/yaw to create a quaternion per sample, then
    rotates ``stateEstimate.vx/y/z`` into body coordinates, producing
    ``vx_body_sd``, ``vy_body_sd``, ``vz_body_sd``.
    """
    # Convert roll, pitch, yaw to quaternion
    r = R.from_euler('xyz', sd_df[['stateEstimate.roll', 'stateEstimate.pitch', 'stateEstimate.yaw']].values, degrees=True)
    quat = r.as_quat()
    sd_df[['qx_sd', 'qy_sd', 'qz_sd', 'qw_sd']] = quat

    # Fill NaN vz with 0 for rotation
    sd_df_filled = sd_df.copy()
    sd_df_filled['stateEstimate.vz'] = sd_df_filled.get('stateEstimate.vz', 0)


    def body_velocity_row_sd(row):
        if any(pd.isna([
            row['stateEstimate.vx'], row['stateEstimate.vy'], row['stateEstimate.vz'],
            row['qx_sd'], row['qy_sd'], row['qz_sd'], row['qw_sd']
        ])):
            return [np.nan, np.nan, np.nan]
        global_vel = [row['stateEstimate.vx'], row['stateEstimate.vy'], row['stateEstimate.vz']]
        quat = [row['qx_sd'], row['qy_sd'], row['qz_sd'], row['qw_sd']]
        return rotate_velocity_to_body_frame(global_vel, quat)

    body_velocities = sd_df_filled.apply(body_velocity_row_sd, axis=1, result_type='expand')
    body_velocities.columns = ['vx_body_sd', 'vy_body_sd', 'vz_body_sd']
    return pd.concat([sd_df, body_velocities], axis=1)


# ==========================
# PLOTTING HELPERS
# ==========================
def plot_dual_axis(ax, x1, y1, label1, x2, y2, label2, ylabel1, ylabel2):
    """Plot two signals with different y‑axes but shared x.

    Convenience helper used mainly for flow vs gyro comparisons.
    """
    ax.plot(x1, y1, label=label1)
    ax_twin = ax.twinx()
    ax_twin.plot(x2, y2, color='orange', label=label2)
    ax.set_ylabel(ylabel1)
    ax_twin.set_ylabel(ylabel2)
    lim1 = max(abs(y1.min()), abs(y1.max()))
    lim2 = max(abs(y2.min()), abs(y2.max()))
    ax.set_ylim(-lim1, lim1)
    ax_twin.set_ylim(-lim2, lim2)
    ax.legend(loc="upper left")
    ax_twin.legend(loc="upper right")
    ax.grid(True, color='lightgrey', linestyle='-', linewidth=0.5)


def plot_comparison(ax, t1, d1, label1, t2, d2, label2, ylabel, t3=None, d3=None, label3=None, ylim=None):
    """Plot 2–3 signals on the same axes for visual comparison."""
    ax.plot(t2, d2, label=label2, color='orange')
    ax.plot(t1, d1, label=label1)
    if t3 is not None and d3 is not None:
        ax.plot(t3, d3, label=label3, color='green', linestyle='--')
    if ylim is not None:
        ax.set_ylim([-ylim, ylim])
    ax.axhline(0, color='grey', linewidth=2)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, color='lightgrey', linestyle='-', linewidth=0.5)


# ==========================
# PLOTTING FUNCTIONS
# ==========================
def plot_motion_and_gyro(sd_data):
    """Compare flowdeck motion deltas with on‑board gyro measurements."""
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
    y1 = sd_data["motion.deltaY"].rolling(CONFIG["smooth_windows"]["y"]).mean()
    y2 = sd_data["gyro.y"].rolling(CONFIG["smooth_windows"]["y"]).mean()
    plot_dual_axis(ax1, sd_data.index, y1, "motion.deltaY (smoothed)",
                   sd_data.index, y2, "gyro.y (smoothed)",
                   "motion.deltaY (smoothed)", "gyro.y (smoothed)")

    x1 = -sd_data["motion.deltaX"].rolling(CONFIG["smooth_windows"]["x"]).mean()
    x2 = sd_data["gyro.x"].rolling(CONFIG["smooth_windows"]["x"]).mean()
    plot_dual_axis(ax2, sd_data.index, x1, "motion.deltaX (smoothed)",
                   sd_data.index, x2, "gyro.x (smoothed)",
                   "motion.deltaX (smoothed)", "gyro.x (smoothed)")
    plt.tight_layout()


def plot_pred_vs_meas(sd_data):
    """Plot Kalman filter predicted vs measured flow (NX/NY)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
    n_filt = 80
    y1 = medfilt(sd_data["kalman_pred.predNX"].rolling(n_filt).mean(), 7)
    y2 = medfilt(sd_data["kalman_pred.measNX"].rolling(n_filt).mean(), 7)
    plot_comparison(ax1, sd_data["timestamp"], y1, "predNX",
                   sd_data["timestamp"], y2, "measNX", "pixels")

    x1 = medfilt(sd_data["kalman_pred.predNY"].rolling(n_filt).mean(), 9)
    x2 = medfilt(sd_data["kalman_pred.measNY"].rolling(n_filt).mean(), 9)
    plot_comparison(ax2, sd_data["timestamp"], x1, "predNY",
                   sd_data["timestamp"], x2, "measNY", "pixels")
    plt.tight_layout()


def plot_velocity_comparison(sd_data, opti_data):
    """Compare body‑frame velocities from estimator, OptiTrack and targets."""
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
    vx = sd_data["vx_body_sd"]
    vy = sd_data["vy_body_sd"]
    target_vx = sd_data["posCtl.targetVX"]
    target_vy = sd_data["posCtl.targetVY"]
    
    plot_comparison(ax1, sd_data["timestamp"], vx,
                    "est body vx",
                    opti_data["timestamp"], opti_data["vx_body"],
                    "opti body vx", "velocity [m/s]",
                    sd_data["timestamp"], target_vx, 
                    "target vx", ylim=0.7)

    plot_comparison(ax2, sd_data["timestamp"], vy,
                    "est body vy",
                    opti_data["timestamp"], opti_data["vy_body"],
                    "opti body vy", "velocity [m/s]",
                    sd_data["timestamp"], target_vy, 
                    "target vy", ylim=0.7)
    plt.tight_layout()


def plot_position_comparison(sd_data, opti_data):
    """Compare estimated and OptiTrack positions (x, y, z)."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
    dt = np.median(np.diff(sd_data["timestamp"]))

    def integrate(series):
        result = np.cumsum(series.fillna(0) * dt)
        return result - result.iloc[0] + series.dropna().iloc[0]

    x = sd_data["stateEstimate.x"].rolling(CONFIG["smooth_windows"]["position"]).mean()
    x = x - x.dropna().iloc[0]  # normalize to start at 0
    y = sd_data["stateEstimate.y"].rolling(CONFIG["smooth_windows"]["position"]).mean()
    y = y - y.dropna().iloc[0]  # normalize to start at 0
    z = sd_data["stateEstimate.z"].rolling(CONFIG["smooth_windows"]["position"]).mean()

    x_int = integrate(sd_data["stateEstimate.vx"].rolling(10).mean())
    y_int = integrate(sd_data["stateEstimate.vy"].rolling(10).mean())

    plot_comparison(ax1, sd_data["timestamp"], x,
                    "est x",
                    opti_data["timestamp"], opti_data["z"].rolling(10).mean(),
                    "opti x", "x [m]", sd_data["timestamp"], x_int, "integrated x")

    plot_comparison(ax2, sd_data["timestamp"], y,
                    "est y",
                    opti_data["timestamp"], opti_data["x"].rolling(5).mean(),
                    "opti y", "y [m]", sd_data["timestamp"], y_int, "integrated y")

    plot_comparison(ax3, sd_data["timestamp"], z,
                    "est z",
                    opti_data["timestamp"], opti_data["y"].rolling(5).mean(),
                    "opti z", "z [m]")
    plt.tight_layout()


def plot_attitude_comparison(sd_data, opti_data):
    """Compare estimated attitude with OptiTrack attitude and commands."""
    valid_quat = opti_data[['qx', 'qy', 'qz', 'qw']].notna().all(axis=1)
    opti_data_valid = opti_data[valid_quat]
    opti_headings = R.from_quat(opti_data_valid[['qz', 'qx', 'qy', 'qw']].values).as_euler('xyz', degrees=True)
    opti_pitch = pd.Series(medfilt(-opti_headings[:, 1], 9))  # Extract pitch
    opti_roll = pd.Series(medfilt(opti_headings[:, 0], 9))   # Extract roll
    opti_yaw = pd.Series(medfilt(opti_headings[:, 2], 9))  # Extract yaw (heading)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # opti_pitch  = compute_filtered(
    #     opti_pitch, opti_data_valid['timestamp'], CONFIG["median_filter_kernel"], CONFIG["velocity_smooth"]["vx"])

    plot_comparison(ax1, sd_data['timestamp'], sd_data['stateEstimate.pitch'].rolling(10).mean(), 'est pitch',
                    opti_data_valid['timestamp'], opti_pitch, 'opti pitch',
                    'Pitch [deg]',
                    sd_data['timestamp'], sd_data['controller.pitch'], 'pitch command', ylim=20)

    plot_comparison(ax2, sd_data['timestamp'], sd_data['stateEstimate.roll'].rolling(10).mean(), 'est roll',
                    opti_data_valid['timestamp'], opti_roll, 'opti roll',
                    'Roll [deg]',
                    sd_data['timestamp'], sd_data['controller.roll'], 'roll command', ylim=20)

    plot_comparison(ax3, sd_data['timestamp'], sd_data['stateEstimate.yaw'].rolling(10).mean(), 'est yaw',
                    opti_data_valid['timestamp'], opti_yaw - opti_yaw[0], 'opti yaw',
                    'Heading [deg]',
                    sd_data['timestamp'], sd_data['controller.yaw'], 'yaw command', ylim=60)

    ax3.set_xlabel('Timestamp (s)')
    plt.tight_layout()


# ==========================
# MAIN
# ==========================
def main():
    """Entry point for plotting comparisons for one configured test.

    The test folder is selected via ``CONFIG['test_name']``.  This
    function orchestrates loading, synchronisation, augmentation of
    data with velocities, and all plotting routines.
    """
    sd_data = load_sd_data(CONFIG["test_name"])
    opti_data = load_opti_data(CONFIG["test_name"])
    opti_data = add_opti_velocities(opti_data)
    sd_data = add_sd_velocities(sd_data)

    # --- synchronize by z-axis ---
    opti_data = synchronize_by_z(sd_data, opti_data)


    # plot_motion_and_gyro(sd_data)
    plot_pred_vs_meas(sd_data)
    plot_velocity_comparison(sd_data, opti_data)
    plot_position_comparison(sd_data, opti_data)
    plot_attitude_comparison(sd_data, opti_data)
    plt.show()


if __name__ == "__main__":
    main()
