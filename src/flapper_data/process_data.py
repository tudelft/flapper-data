import argparse
import os

import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from flapper_data import data_loader

OUTPUT_RATE = 200  # Hz
ONBOARD_RATE = 200  # Hz, nominal rate of the onboard log
MAX_GAP = 0.05  # s, mocap dropouts longer than this are NaN in the output
CHECK_CUTOFF = 5  # Hz, low-pass of the gyro and mocap rates compared to check the sync

# Frames that moved more than this from the last good frame are rigid-body mis-solves,
# e.g. an orientation flipped for a single frame, and are dropped
JUMP_ANGLE = np.radians(20)  # rad, plus MAX_ANGULAR_RATE * time since the last good frame
MAX_ANGULAR_RATE = np.radians(3000)  # rad/s
JUMP_DISTANCE = 0.02  # m, plus MAX_SPEED * time since the last good frame
MAX_SPEED = 10  # m/s

# Search ranges of the sync, around the initial guess
SCALE_RANGE = 0.005  # relative
OFFSET_RANGE = 0.5  # s

# Maps the Motive frame (Z forward, X left, Y up) to NED. The rigid body axes follow
# the same convention, so the same matrix maps them to the FRD body frame.
MOTIVE_TO_NED = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])

POS = ["ned.x", "ned.y", "ned.z"]
QUAT = ["ned.qw", "ned.qx", "ned.qy", "ned.qz"]
EULER = ["ned.roll", "ned.pitch", "ned.yaw"]
VEL_NED = ["ned.velx", "ned.vely", "ned.velz"]
VEL_FRD = ["frd.velx", "frd.vely", "frd.velz"]
RATES = ["frd.p", "frd.q", "frd.r"]


def _long_gaps(tracked, max_frames):
    """Mask of the untracked frames that belong to a dropout longer than max_frames."""
    edges = np.diff(np.r_[0, (~tracked).astype(int), 0])
    mask = np.zeros(len(tracked), dtype=bool)
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if end - start > max_frames:
            mask[start:end] = True
    return mask


def _mis_solves(t, pos, quat, tracked):
    """Mask of the frames that jumped further from the last good frame than the drone can move."""
    bad = np.zeros(len(t), dtype=bool)
    valid = np.flatnonzero(tracked)
    last = valid[0]
    for k in valid[1:]:
        dt = t[k] - t[last]
        angle = 2 * np.arccos(min(1.0, abs(quat[k] @ quat[last])))
        distance = np.linalg.norm(pos[k] - pos[last])
        if angle > JUMP_ANGLE + MAX_ANGULAR_RATE * dt or distance > JUMP_DISTANCE + MAX_SPEED * dt:
            bad[k] = True
        else:
            last = k
    return bad


def mocap_states(mocap, fps, cutoff):
    """
    Pose, velocity and body rates of the drone at the mocap frame rate.

    The position and quaternion are low-passed (zero phase) and then differentiated:
    the velocity by central differences, the body rates from the relative rotation
    between the neighbouring frames.

    Untracked and mis-solved frames are dropouts: they are interpolated before filtering,
    and the 'gap' column marks the ones in dropouts longer than MAX_GAP.

    Returns a DataFrame with the time, the POS, QUAT, VEL_NED, VEL_FRD and RATES columns,
    and the 'gap' column.
    """
    t = mocap["time"].to_numpy()
    tracked = mocap.notna().all(axis=1).to_numpy()

    pos = mocap[["x", "y", "z"]].to_numpy() @ MOTIVE_TO_NED.T
    # The vector part of a quaternion rotates like a vector
    quat = np.column_stack([mocap["qw"], mocap[["qx", "qy", "qz"]].to_numpy() @ MOTIVE_TO_NED.T])

    mis_solved = _mis_solves(t, pos, quat, tracked)
    gap = _long_gaps(tracked & ~mis_solved, int(MAX_GAP * fps))
    print(f"Mocap dropouts: {(~tracked).sum()} untracked and {mis_solved.sum()} mis-solved frames, "
          f"{gap.sum()} of them in dropouts longer than {MAX_GAP} s")
    tracked = tracked & ~mis_solved
    pos[~tracked] = np.nan
    quat[~tracked] = np.nan

    # q and -q are the same rotation: pick the sign that keeps the quaternion continuous
    valid = np.flatnonzero(tracked)
    dots = np.einsum("ij,ij->i", quat[valid[1:]], quat[valid[:-1]])
    quat[valid] *= np.cumprod(np.r_[1, np.where(dots < 0, -1, 1)])[:, np.newaxis]

    # Fill the dropouts, then low-pass
    pos = pd.DataFrame(pos).interpolate(limit_direction="both").to_numpy()
    quat = pd.DataFrame(quat).interpolate(limit_direction="both").to_numpy()

    b, a = signal.butter(4, cutoff, fs=fps)
    pos = signal.filtfilt(b, a, pos, axis=0)
    quat = signal.filtfilt(b, a, quat, axis=0)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)

    rot = R.from_quat(quat, scalar_first=True)  # FRD -> NED

    vel_ned = np.gradient(pos, t, axis=0)
    vel_frd = rot.inv().apply(vel_ned)

    # Central differences, one-sided at the ends
    n = len(t)
    before = np.r_[0, np.arange(n - 2), n - 2]
    after = np.r_[1, np.arange(2, n), n - 1]
    rates = (rot[before].inv() * rot[after]).as_rotvec() / (t[after] - t[before])[:, np.newaxis]

    states = pd.DataFrame(np.column_stack([pos, quat, vel_ned, vel_frd, rates]), columns=POS + QUAT + VEL_NED + VEL_FRD + RATES)
    states.insert(0, "time", t)
    states["gap"] = gap
    return states


def _xcorr(t_ref, ref, t_sig, sig, offset_range=None):
    """
    Offset that best aligns sig with ref, so that ref(t + offset) ~ sig(t), and the
    correlation at that offset (summed over the columns).
    Both signals are resampled to a uniform grid at OUTPUT_RATE.
    """
    dt = 1 / OUTPUT_RATE

    def uniform(t, x):
        grid = np.arange(t[0], t[-1], dt)
        x = np.column_stack([np.interp(grid, t, col) for col in x.T])
        return (x - x.mean(axis=0)) / x.std(axis=0)

    ref, sig = uniform(t_ref, ref), uniform(t_sig, sig)
    corr = sum(signal.correlate(ref[:, i], sig[:, i], method="fft") for i in range(ref.shape[1]))
    offsets = signal.correlation_lags(len(ref), len(sig)) * dt + t_ref[0] - t_sig[0]

    search = np.arange(1, len(corr) - 1)
    if offset_range is not None:
        search = search[(offsets[search] >= offset_range[0]) & (offsets[search] <= offset_range[1])]
    k = search[np.argmax(corr[search])]

    # Parabolic interpolation of the peak, for sub-sample resolution
    c0, c1, c2 = corr[k - 1 : k + 2]
    shift = 0.5 * (c0 - c2) / (c0 - 2 * c1 + c2)
    return offsets[k] + shift * dt, c1


def sync(onboard, states, cutoff):
    """
    Fits the onboard clock to the mocap clock: mocap time = scale * onboard time + offset,
    with the onboard time starting at 0 at the first onboard sample.

    The coarse offset comes from the position streamed to the drone (locSrv) and the
    mocap position. Scale and offset are then found by maximising the cross-correlation
    of the gyro and the mocap body rates, low-passed alike.
    """
    t_onboard = (onboard["timestamp"].to_numpy() - onboard["timestamp"].iloc[0]) / 1000
    t_mocap = states["time"].to_numpy()

    # Initial guess: the log runs at its nominal rate
    scale0 = 1 / (ONBOARD_RATE * np.mean(np.diff(t_onboard)))

    # locSrv is in x forward, y left, z up
    loc = onboard[["locSrv.x", "locSrv.y", "locSrv.z"]].to_numpy() * [1, -1, -1]
    offset0, _ = _xcorr(t_mocap, states[POS].to_numpy(), scale0 * t_onboard, loc)

    # The gyro is in x forward, y left, z up, in deg/s
    gyro = np.radians(onboard[["gyro.x", "gyro.y", "gyro.z"]].to_numpy()) * [1, -1, -1]
    gyro = signal.filtfilt(*signal.butter(4, cutoff, fs=ONBOARD_RATE), gyro, axis=0)
    rates = states[RATES].to_numpy()
    offset_range = (offset0 - OFFSET_RANGE, offset0 + OFFSET_RANGE)

    def best(scales):
        fits = [(*_xcorr(t_mocap, rates, s * t_onboard, gyro, offset_range), s) for s in scales]
        offset, _, scale = max(fits, key=lambda fit: fit[1])
        return scale, offset

    scale, _ = best(scale0 * (1 + np.arange(-SCALE_RANGE, SCALE_RANGE + 1e-9, 1e-4)))
    scale, offset = best(scale * (1 + np.arange(-1e-4, 1e-4 + 1e-9, 1e-5)))
    return scale, offset


def sync_quality(onboard, states, scale, offset):
    """
    Correlation of the gyro and the mocap body rates while flying (thrust > 0), per axis,
    both low-passed at CHECK_CUTOFF.
    """
    t_onboard = (onboard["timestamp"].to_numpy() - onboard["timestamp"].iloc[0]) / 1000
    t = scale * t_onboard + offset
    t_mocap = states["time"].to_numpy()
    flying = (onboard["controller.cmd_thrust"].to_numpy() > 0) & (t > t_mocap[0]) & (t < t_mocap[-1])

    gyro = np.radians(onboard[["gyro.x", "gyro.y", "gyro.z"]].to_numpy()) * [1, -1, -1]
    gyro = signal.filtfilt(*signal.butter(4, CHECK_CUTOFF, fs=ONBOARD_RATE), gyro, axis=0)
    fps = 1 / np.mean(np.diff(t_mocap))
    rates = signal.filtfilt(*signal.butter(4, CHECK_CUTOFF, fs=fps), states[RATES].to_numpy(), axis=0)
    return [np.corrcoef(gyro[flying, i], np.interp(t[flying], t_mocap, rates[:, i]))[0, 1] for i in range(3)]


def merge(onboard, states, scale, offset):
    """
    Combines the onboard and mocap data on a uniform OUTPUT_RATE grid, over the time
    both were recording. Each grid point takes the nearest onboard sample, unchanged,
    and the mocap states interpolated from the mocap frames.
    """
    t_onboard = (onboard["timestamp"].to_numpy() - onboard["timestamp"].iloc[0]) / 1000
    t_onboard = scale * t_onboard + offset
    t_mocap = states["time"].to_numpy()

    start = max(t_onboard[0], t_mocap[0])
    end = min(t_onboard[-1], t_mocap[-1])
    grid = start + np.arange(int((end - start) * OUTPUT_RATE) + 1) / OUTPUT_RATE

    # Nearest onboard sample
    idx = np.clip(np.searchsorted(t_onboard, grid), 1, len(t_onboard) - 1)
    idx -= grid - t_onboard[idx - 1] < t_onboard[idx] - grid
    merged = onboard.iloc[idx].reset_index(drop=True).add_prefix("onboard.")
    merged.insert(0, "time", np.arange(len(grid)) / OUTPUT_RATE)

    cols = POS + QUAT + VEL_NED + VEL_FRD + RATES
    mocap = pd.DataFrame(CubicSpline(t_mocap, states[cols].to_numpy())(grid), columns=cols)
    quat = mocap[QUAT].to_numpy()
    mocap[QUAT] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    yaw, pitch, roll = R.from_quat(mocap[QUAT].to_numpy(), scalar_first=True).as_euler("ZYX").T
    mocap[EULER] = np.column_stack([roll, pitch, yaw])

    gap = np.interp(grid, t_mocap, states["gap"].astype(float)) > 0
    mocap.loc[gap] = np.nan

    mocap = mocap[POS + QUAT + EULER + VEL_NED + VEL_FRD + RATES].add_prefix("optitrack.")
    return pd.concat([merged, mocap], axis=1)


def process(cfg):
    if not cfg.mocap_cutoff_hz < OUTPUT_RATE / 2:
        raise ValueError(f"mocap_cutoff_hz must be below {OUTPUT_RATE / 2} Hz")

    onboard = pd.read_csv(cfg.onboard_path)
    mocap, fps = data_loader.read_mocap(cfg.mocap_path, cfg.rigid_body)
    print(f"Onboard: {cfg.onboard_path}, {len(onboard)} samples")
    print(f"Mocap:   {cfg.mocap_path}, {len(mocap)} frames at {fps:g} Hz")

    states = mocap_states(mocap, fps, cfg.mocap_cutoff_hz)

    if cfg.clock_scale is not None and cfg.time_offset is not None:
        scale, offset = cfg.clock_scale, cfg.time_offset
        print("Sync from flights.yaml")
    else:
        scale, offset = sync(onboard, states, cfg.mocap_cutoff_hz)

    rate = 1 / (scale * np.mean(np.diff(onboard["timestamp"])) / 1000)
    corr = sync_quality(onboard, states, scale, offset)
    print(f"Sync: mocap time = {scale:.5f} * onboard time + {offset:.3f} s (onboard log at {rate:.2f} Hz)")
    print(f"Gyro vs mocap rates correlation while flying: p {corr[0]:.2f}, q {corr[1]:.2f}, r {corr[2]:.2f}")
    if np.mean(corr) < 0.5:
        print("Warning: poor correlation, check the sync")

    processed = merge(onboard, states, scale, offset)
    os.makedirs(os.path.dirname(cfg.processed_path), exist_ok=True)
    processed.to_csv(cfg.processed_path, index=False)
    print(f"Saved {len(processed)} rows ({len(processed) / OUTPUT_RATE:.1f} s) to {cfg.processed_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync and merge the onboard and mocap data of a flight")
    parser.add_argument("flights", nargs="*", help="Flights from flights.yaml (default: all)")
    args = parser.parse_args()

    for flight in args.flights or data_loader.flights():
        print(f"=== {flight}")
        process(data_loader.load(flight))
