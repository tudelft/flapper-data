import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# ----------------------
# Parameters
# ----------------------
data_dir = "data/flowdeck_cyberzoo_010"
FLOW_RESOLUTION = 0.05       # from firmware
Npix = 35.0
thetapix = 0.71674
offset_sec = 5
# offset_sec = 57.4            # your known time offset (007)
# offset_sec = 40.7            # your known time offset (006)
# offset_sec = 33.1            # your known time offset (005)
# offset_sec = 25.3            # your known time offset (004)
ma_window = 20                 # centered moving avg window (samples @ SD rate)


# ----------------------
# Load your CSVs
# ----------------------
opti = pd.read_csv(f"{data_dir}/opti.csv", skiprows=7, usecols=[0,1,2,3,4,5,6,7,8],
                   header=None, index_col=0,
                   names=["index","timestamp","qx","qy","qz","qw","x","y","z"])
sd   = pd.read_csv(f"{data_dir}/sd.csv")

# ----------------------
# Clean Opti quaternions and compute Rzz
# ----------------------
# Keep only rows where quaternions are finite and non-zero norm
q = opti[["qx","qy","qz","qw"]].to_numpy()
q_norm = np.linalg.norm(q, axis=1)
valid = np.isfinite(q).all(axis=1) & np.isfinite(opti["timestamp"].to_numpy()) & (q_norm > 1e-6)

opti_valid = opti.loc[valid].copy()
# Renormalize quaternions
qv = opti_valid[["qx","qy","qz","qw"]].to_numpy()
qv = (qv.T / np.linalg.norm(qv, axis=1)).T
opti_valid[["qx","qy","qz","qw"]] = qv

# Time in seconds (already seconds in Opti)
t_o = opti_valid["timestamp"].to_numpy()
x_o = opti_valid["z"].to_numpy()
y_o = opti_valid["x"].to_numpy()
z_o = opti_valid["y"].to_numpy()

# Velocities (simple gradient; replace with Savitzky–Golay if you want smoother)
vx_o = np.gradient(x_o, t_o)
vy_o = np.gradient(y_o, t_o)

# Rzz from rotation matrices
Rzz_o = R.from_quat(opti_valid[["qz","qx","qy","qw"]].to_numpy()).as_matrix()[:, 2, 2]

# ----------------------
# SD prep + alignment
# ----------------------
t_sd = sd["timestamp"].to_numpy(dtype=float) / 990.0  # ms -> s
tq = t_sd - offset_sec                                 # align SD to Opti timeline

# Restrict to overlap in time
mask = (tq >= t_o[0]) & (tq <= t_o[-1])
tq_valid = tq[mask]

# Interpolate Opti signals at SD (aligned) timestamps
vxq  = np.interp(tq_valid, t_o, vx_o)
vyq  = np.interp(tq_valid, t_o, vy_o)
zq   = np.interp(tq_valid, t_o, z_o)
Rzzq = np.interp(tq_valid, t_o, Rzz_o)

# SD signals (gyro deg/s -> rad/s; flow scaling 0.1)
gyro_x = (sd["gyro.x"].to_numpy(dtype=float) * np.pi/180.0)[mask]
gyro_y = (sd["gyro.y"].to_numpy(dtype=float) * np.pi/180.0)[mask]
dx_meas = (sd["motion.deltaX"].to_numpy(dtype=float) * FLOW_RESOLUTION)[mask]
dy_meas = (sd["motion.deltaY"].to_numpy(dtype=float) * FLOW_RESOLUTION)[mask]

# Δt per SD sample
t_sd_valid = t_sd[mask]
dt = np.diff(t_sd_valid, prepend=t_sd_valid[0])
# Guard against zeros/negatives
if np.any(dt <= 0):
    med = np.median(dt[dt > 0]) if np.any(dt > 0) else 1/200.0
    dt[(dt <= 0) | ~np.isfinite(dt)] = med

# ----------------------
# Predicted vs measured flow (per SD interval)
# ----------------------
zq_clip = np.maximum(zq, 0.1)

nx_pred = (dt * Npix / thetapix) * ((vxq * Rzzq / zq_clip) - gyro_y)
ny_pred = (dt * Npix / thetapix) * ((vyq * Rzzq / zq_clip) + gyro_x)

# ----------------------
# (Optional) centered moving average smoothing
# ----------------------
def moving_average(x, w=5, mode="causal"):
    if w <= 1: return x
    if mode == "causal":
        return np.convolve(x, np.ones(w)/w, mode="full")[:len(x)]
    elif mode == "centered":
        return np.convolve(x, np.ones(w)/w, mode="same")

# def causal_ma(x, w=5):
#     if w <= 1: return x
#     return np.convolve(x, np.ones(w)/w, mode="full")[:len(x)]

nx_pred_f = moving_average(nx_pred, ma_window, "centered")
ny_pred_f = moving_average(ny_pred, ma_window, "centered")
dx_meas_f = moving_average(dx_meas, ma_window, "centered")
dy_meas_f = moving_average(dy_meas, ma_window, "centered")

# ----------------------
# Done: use *_f for filtered, or raw arrays otherwise
# ----------------------
out = pd.DataFrame({
    "t": t_sd_valid,
    "nx_pred": nx_pred, "ny_pred": ny_pred,
    "dx_meas": dx_meas, "dy_meas": dy_meas,
    "nx_pred_f": nx_pred_f, "ny_pred_f": ny_pred_f,
    "dx_meas_f": dx_meas_f, "dy_meas_f": dy_meas_f,
})

fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# X flow
axs[0].plot(out["t"], out["nx_pred_f"], label="Predicted nx (smoothed)", color='blue')
axs[0].plot(out["t"], -out["dy_meas_f"], label="Measured dy (smoothed)", color='red')
axs[0].set_ylabel("Flow X (m/s)")
axs[0].set_title("Flow Prediction vs Measurement (X)")
axs[0].legend()
axs[0].grid()
axs[0].set_ylim(-0.7, 0.7)

# Y flow
axs[1].plot(out["t"], out["ny_pred_f"], label="Predicted ny (smoothed)", color='green')
axs[1].plot(out["t"], -out["dx_meas_f"], label="Measured dx (smoothed)", color='orange')
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Flow Y (m/s)")
axs[1].set_title("Flow Prediction vs Measurement (Y)")
axs[1].legend()
axs[1].grid()
axs[1].set_ylim(-0.7, 0.7)

plt.tight_layout()
plt.show()
