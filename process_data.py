"""
Processes onboard and optitrack data 

Federico Angioni
"""


import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from math import gcd
from scipy import signal
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from utils.state_estimator import MahonyIMU
import os
import config



WINDOW_SIZE = 16
TARGET_FFT_SIZE = 256
FREQ_RANGE = (5, 25)

FREE_FLIGHT = False

# OptiTrack z,x,y --> x,y,z

onboard_freq = 200  # Hz
filter_cutoff_freq = 2  # Hz
g0 = 9.80665  # m/s

# Columns to use to sync the optitrack and IMU data
columns_sync = ["acc.z", "pitch", "roll", "yaw"]
show = False



body_to_CoM = np.array([+0.001, 0.0, -0.13])

def get_optitrack_meta(optitrack_csv):
    print("Obtaining the optitrack metadata ...")

    with open(optitrack_csv, "r") as f:
        lines = f.readline()

    metadata_raw = lines.strip().split(",")
    metadata = dict(zip(metadata_raw[::2], metadata_raw[1::2]))

    if metadata["Format Version"] != "1.23":
        print("The code has not been tested with this Optitrack file version")

    fps = float(metadata["Capture Frame Rate"])
    print(f"Found .csv ... optitrack data recorded at {fps} fps")

    return metadata


def calculate_dihedral_angle(forward_body, norm_dihedral):
    """Calculate dihedral angle with proper sign based on wing side"""
    cross_products = np.cross(forward_body, norm_dihedral)
    
    # Norms
    norms_cross = np.linalg.norm(cross_products, axis=1)
    norms_forward = np.linalg.norm(forward_body, axis=1)
    norms_dihedral = np.linalg.norm(norm_dihedral, axis=1)
    
    # Unsigned angles
    angles = np.arcsin(norms_cross / (norms_forward * norms_dihedral))
    
    signs = np.where(cross_products[:, 0] >= 0, 1, -1)
    return angles * signs


def align_to_original_length(dominant_freqs, N, window_size):
    aligned = np.full(N, 0)  # same length as original
    offset = window_size // 2
    aligned[offset : offset + len(dominant_freqs)] = dominant_freqs
    return aligned


def calculate_frequency(
    norm_dihedral,
    wing_vector,
    sample_rate,
    window_size=16,
    fft_size=256,
    freq_range=(5, 25),
):
    dot = np.einsum("ij,ij->i", wing_vector, norm_dihedral)

    wing_norms = np.linalg.norm(wing_vector, axis=1)

    dihedral_norms = np.linalg.norm(norm_dihedral, axis=1)

    flapping_angle = np.arcsin(dot / (wing_norms * dihedral_norms))

    flapping_windows = np.lib.stride_tricks.sliding_window_view(
        flapping_angle, window_size
    )
    windows = flapping_windows - flapping_windows.mean(axis=1, keepdims=True)

    # Zero-pad each window to fft_size
    padded_signal = np.zeros((windows.shape[0], fft_size))
    padded_signal[:, :window_size] = windows

    # FFT for all windows at once
    freqs = np.fft.fftfreq(fft_size, 1 / sample_rate)
    fft_vals = np.abs(np.fft.fft(padded_signal, axis=1))

    # Restrict to target frequency range
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    valid_freqs = freqs[mask]
    valid_fft = fft_vals[:, mask]

    # Pick dominant frequency per window
    dominant_idx = np.argmax(valid_fft, axis=1)
    dominant_freqs = valid_freqs[dominant_idx]

    aligned_dominant = align_to_original_length(
        dominant_freqs, len(norm_dihedral), window_size
    )

    return aligned_dominant


def handle_nan(data, frame_rate, time_limit):
    """
    Handles the NaN or missing values in the OptiTrack .csv file using
    cubic interpolation across each recorded variable.

    Parameters:
    -----------
        data: pandas.DataFrame
            DataFrame containing the raw OptiTrack data
        frame_rate: int
            Frame rate at which the OptiTrack captured data
        time_limit: int
            Limit in seconds of maximum allowed gap between missing logs

    Returns:
    --------
        interpolated_data: pandas.DataFrame
            Data with interpolated values and removed NaNs
    """

    frame_limit = int(frame_rate / time_limit)

    interpolated_data = data.interpolate(method="cubic", axis=0, limit=frame_limit)

    return interpolated_data


def resample_data(data, up, down, fs_original):
    columns_name = data.columns[1:]
    g = gcd(up, down)
    up //= g
    down //= g

    original_N = len(data)
    duration = (original_N - 1) / fs_original

    data_resampled = signal.resample_poly(data.iloc[:, 1:], up, down)
    data_resampled = pd.DataFrame(data_resampled, columns=columns_name)

    time_array = np.linspace(0, duration, len(data_resampled))
    data_resampled.insert(0, "time", time_array)

    return data_resampled


def filter_data(data, cutoff_freq, sampling_freq):
    """
    filter --> scipy.signal.butter
    use scipy.integrate.simpson or trapezoidal
    """

    columns_name = data.columns

    b, a = signal.butter(4, cutoff_freq, fs=sampling_freq)

    filtered_data = signal.filtfilt(b, a, data.iloc[:, 1:], axis=0)

    filtered_df = pd.DataFrame(filtered_data, columns=columns_name[1:])

    time_array = np.linspace(0, filtered_df.shape[0] / sampling_freq, filtered_df.shape[0])

    filtered_df.insert(0, "time", time_array)

    return filtered_df


def process_optitrack(data, com_body, optitrack_fps):
    """
    Orients the optitrack data to the correct body orientation. Optitrack defines body axes
    as RightForwardUp respectively for x, y, z. Thus use a Euler intrinsic rotation, 'yxz',
    corresponding to roll, pitch, and yaw.

    Parameters:
    -----------
        data : pandas.DataFrame
            DataFrame containing the raw OptiTrack data
        reference_frame : str
            Defines the orientation the data is processed to.
            Can be either "ForwardLeftUp" or "ForwardRightDown" as from aerospace convention,
            indicating respectively x, y, z.
        com_body: numpy.array # (3,)
            Position of the center of mass with respect to the optitrack defined geometric center in the body frame.

    Returns:
    --------
        oriented_data: pandas.DataFrame
    """

    # Rotational kinematics
    quats = np.vstack((data["fbqx"], data["fbqy"], data["fbqz"], data["fbqw"])).T

    # Compute the norm (length) of each quaternion (row-wise)
    norms = np.linalg.norm(quats, axis=1, keepdims=True)

    # Normalize each quaternion
    quats = quats / norms

    r = R.from_quat(quats, scalar_first=False)

    # Optitrack defines Z-up, Y-left, X-forward
    euler_angles = r.as_euler("YXZ", degrees=False)

    # First extract yaw, pitch and then roll
    psi, theta, phi = -euler_angles[:, 0], -euler_angles[:, 1], euler_angles[:, 2] # all checked correct orientation

    oriented_euler = np.asarray([phi, theta, psi])

    euler_rates = np.gradient(oriented_euler, data["time"], axis=1)  # (3, N)

    rotations_rates = np.array(
        [
            [np.ones_like(theta), np.zeros_like(theta), -np.sin(theta)],
            [np.zeros_like(theta), np.cos(phi), np.sin(phi) * np.cos(theta)],
            [np.zeros_like(theta), -np.sin(phi), np.cos(phi) * np.cos(theta)],
        ]
    )  # (3, 3, N)

    # Body rates found from the rotation matrix above
    roll_rate, pitch_rate, yaw_rate = np.einsum(
        "ijk,jk->ik", rotations_rates, euler_rates
    )  # (N,), (N,), (N,)

    
    rates_body_wrt_ref = np.asarray([roll_rate, pitch_rate, yaw_rate])  # (3, N)

    alpha_body_wrt_ref = np.gradient(rates_body_wrt_ref, data["time"], axis=1)  # (3, N)


    # Now move onto translational kinematics, Optitrack defines Z-up, Y-left, X-forward
    body_pos_ref = np.asarray([data["fbz"], -data["fbx"], -data["fby"]])  # (3, N)

    vel_ref = np.gradient(body_pos_ref, data["time"], axis=1)  # (3, N)

    acc_ref = np.gradient(vel_ref, data["time"], axis=1)  # (3, N)

    # Position of the CoM in body frame
    pos_com_ref = body_pos_ref + com_body[:, np.newaxis]  # reshape com_body to (3, N)

    vel_com_ref = vel_ref + np.cross(rates_body_wrt_ref, com_body, axis=0)

    acc_com_ref = (
        acc_ref
        + np.cross(alpha_body_wrt_ref, com_body, axis=0)
        + np.cross(
            rates_body_wrt_ref, np.cross(rates_body_wrt_ref, com_body, axis=0), axis=0
        )
    )

    # Rotations matrix from global frame to body
    rotations_RefToBody = np.array(
        [
            [
                np.cos(theta) * np.cos(psi),
                np.cos(theta) * np.sin(psi),
                -np.sin(theta),
            ],
            [
                (
                    -np.cos(phi) * np.sin(psi)
                    + np.sin(phi) * np.sin(theta) * np.cos(psi)
                ),
                (np.cos(phi) * np.cos(psi) + np.sin(phi) * np.sin(theta) * np.sin(psi)),
                np.sin(phi) * np.cos(theta),
            ],
            [
                (np.sin(phi) * np.sin(psi) + np.cos(phi) * np.sin(theta) * np.cos(psi)),
                (
                    -np.sin(phi) * np.cos(psi)
                    + np.cos(phi) * np.sin(theta) * np.sin(psi)
                ),
                np.cos(phi) * np.cos(theta),
            ],
        ]
    )

    velx_com_body, vely_com_body, velz_com_body = np.einsum("ijk,jk->ik", rotations_RefToBody, vel_com_ref)

    accx_com_body, accy_com_body, accz_com_body = np.einsum("ijk,jk->ik", rotations_RefToBody, acc_com_ref)

    
    processed_data = pd.DataFrame(
        {
            "time": data["time"],
            "roll": phi,
            "pitch": theta,
            "yaw": psi,
            "p": roll_rate,
            "q": pitch_rate,
            "r": yaw_rate,
            "p_dot": alpha_body_wrt_ref[0, :],
            "q_dot": alpha_body_wrt_ref[1, :],
            "r_dot": alpha_body_wrt_ref[2, :],
            "vel.x": velx_com_body,
            "vel.y": vely_com_body,
            "vel.z": velz_com_body,
            "acc.x": accx_com_body,
            "acc.y": accy_com_body,
            "acc.z": accz_com_body,

        }
    )

    # Include all the markers position in the processed dataframe
    processed_data = pd.concat([processed_data, data.iloc[:, 1:]], axis=1)

    return processed_data   

def process_frequency_dihedral(data, optitrack_fps):
    body_pos_ref = np.asarray([data["fbz"], -data["fbx"], -data["fby"]])
    wing_rootR_ref = np.array([data["fbrw2z"], -data["fbrw2x"], -data["fbrw2y"]])
    wing_lastR_ref = np.array([data["fbrw3z"], -data["fbrw3x"], -data["fbrw3y"]])

    wing_rootL_ref = np.array([data["fblw3z"], -data["fblw3x"], -data["fblw3y"]])
    wing_lastL_ref = np.array([data["fblw1z"], -data["fblw1x"], -data["fblw1y"]])
    top_body_marker_ref = np.array([data["fb1z"], -data["fb1x"], -data["fb1y"]])

    # Rotational kinematics
    quats = np.vstack((data["fbqz"], data["fbqx"], data["fbqy"], data["fbqw"])).T

    # Compute the norm (length) of each quaternion (row-wise)
    norms = np.linalg.norm(quats, axis=1, keepdims=True)

    # Normalize each quaternion
    quats = quats / norms

    r = R.from_quat(quats, scalar_first=False)


    forward_body = r.apply([1, 0, 0])

    # Define point B, center of body "fbx, fby, fbz" in optitrack
    # Define point A, top of the drone, usually fb1x, ...
    # Define point C, root of each wing
    AB = top_body_marker_ref - body_pos_ref
    AC_right = wing_rootR_ref - top_body_marker_ref
    AC_left = wing_rootL_ref - top_body_marker_ref

    norm_dihedral_right = np.cross(AB.T, AC_right.T)
    norm_dihedral_left = np.cross(AB.T, AC_left.T)

    dihedral_right = calculate_dihedral_angle(forward_body, norm_dihedral_right)
    dihedral_left = calculate_dihedral_angle(forward_body, norm_dihedral_left)

    right_wing_vector = (wing_lastR_ref - wing_rootR_ref).T


    freq_right = calculate_frequency(
        norm_dihedral_right,
        right_wing_vector,
        optitrack_fps,
        WINDOW_SIZE,
        TARGET_FFT_SIZE,
        FREQ_RANGE,
    )

    left_wing_vector = (wing_lastL_ref - wing_rootL_ref).T

    freq_left = calculate_frequency(
        norm_dihedral_left,
        left_wing_vector,
        optitrack_fps,
        WINDOW_SIZE,
        TARGET_FFT_SIZE,
        FREQ_RANGE,
    )

    output = pd.DataFrame({"time":data["time"],
                            "freq.right": freq_right,
                            "freq.left": freq_left,
                            "dihedral.right": dihedral_right,
                            "dihedral.left": dihedral_left})
    
    return output


def process_onboard(data, sampling_freq):
    """
    Processes onboard data by using a Mahony filter for estimating attude.
    Orients accelerations and removes the gravity vector.

    Return:
        - Processed dataframe with all angle values in radians, and correctly oriented in the aerospace reference frane NED
    """
    estimator = MahonyIMU()

    attitude = {"pitch": [], "roll": [], "yaw": []}
    
    # IMU uses x forward, y to the left, z up
    p = np.radians(data["gyro.x"])
    q = np.radians(-data["gyro.y"])
    r = np.radians(-data["gyro.z"])

    # calculate correct attitude with Mahony filter
    for i in range(len(data)):
        gx_i, gy_i, gz_i = data.loc[i, ["gyro.x", "gyro.y", "gyro.z"]]

        ax_i, ay_i, az_i = data.loc[i, ["acc.x", "acc.y", "acc.z"]]

        qx, qy, qz, qw = estimator.sensfusion6Update(
            gx_i, gy_i, gz_i, ax_i, ay_i, az_i, 1 / sampling_freq
        )

        yaw_i, pitch_i, roll_i = R.from_quat([qx, qy, qz, qw]).as_euler("ZYX")

        attitude["roll"].append(roll_i)
        attitude["pitch"].append(-pitch_i)
        attitude["yaw"].append(-yaw_i)

    # rotational accelerations
    roll_acc = np.gradient(p, 1 / sampling_freq)
    pitch_acc = np.gradient(q, 1 / sampling_freq)
    yaw_acc = np.gradient(r, 1 / sampling_freq)


    # translational accelerations ->> check how this behaves
    acc_x = data["acc.x"]
    acc_y = - data["acc.y"]
    acc_z = - data["acc.z"]


    acc_x = acc_x * g0 - np.sin(attitude["pitch"]) * g0 
    acc_y = acc_y * g0 + np.sin(attitude["roll"]) * np.cos(attitude["pitch"]) * g0
    acc_z = acc_z * g0 + np.cos(attitude["pitch"]) * np.cos(attitude["roll"]) * g0

    # translational velocity
    vel_x = cumulative_trapezoid(acc_x, dx=1 / sampling_freq, initial=0)
    vel_y = cumulative_trapezoid(acc_y, dx=1 / sampling_freq, initial=0)
    vel_z = cumulative_trapezoid(acc_z, dx=1 / sampling_freq, initial=0)

    processed_data = pd.DataFrame({
            "roll": attitude["roll"],
            "pitch": attitude["pitch"],
            "yaw": attitude["yaw"],
            "p": p,
            "q": q,
            "r": r,
            "p_dot": roll_acc,
            "q_dot": pitch_acc,
            "r_dot": yaw_acc,
            "vel.x": vel_x,
            "vel.y": vel_y,
            "vel.z": vel_z,
            "acc.x": acc_x,
            "acc.y": acc_y,
            "acc.z": acc_z,
        })
    
    # Drop useless columns
    data = data.drop(["gyro.x", "gyro.y", "gyro.z", "acc.x", "acc.y", "acc.z"], axis=1)

    # Concatenate the processed data
    output = pd.concat([data, processed_data], axis=1)
    
    return output


def find_lag(onboard, optitrack, columns_sync):

    # Extract single column as 1D array
    x = onboard[columns_sync[0]].values
    y = optitrack[columns_sync[0]].values
    
    # Normalize the signals (helps with correlation)
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    
    correlation = signal.correlate(x, y, mode="full")
    lags = signal.correlation_lags(x.size, y.size, mode="full")
    lag = lags[np.argmax(correlation)]
    
    # Debug: print correlation strength
    max_corr = np.max(correlation)
    print(f"Lag: {lag} samples, Max correlation: {max_corr:.4f}")
    
    return lag


def shift_data(data, lag, sampling_freq):
    time_shift = lag / sampling_freq

    data["time"] = np.linspace(time_shift, data["time"].iloc[-1] + time_shift, len(data["time"]))
    return data, time_shift


# Save everything in radians
def merge_dfs(onboard, optitrack, sampling_freq):
    onboard = onboard.rename(
        columns={col: f"onboard.{col}" for col in onboard.columns if col != "time"}
    )
    optitrack = optitrack.rename(
        columns={col: f"optitrack.{col}" for col in optitrack.columns if col != "time"}
    )

    # Use merge_asof for nearest time matching
    merged = pd.merge_asof(
        onboard.sort_values('time'), 
        optitrack.sort_values('time'), 
        on="time", 
        direction="nearest",
        tolerance=1/(2*sampling_freq)  # Allow matching within half a sample period
    )
    
    # Remove any rows with NaN (outside tolerance)
    merged = merged.dropna()
    
    # Recreate uniform time array
    num_samples = len(merged)
    merged["time"] = np.linspace(0, (num_samples - 1) / sampling_freq, num_samples)

    return merged


def orient_onboard(data, sampling_freq, time_shift):
    """
    Orients the onboard logged data to the aerospace standard convention,
    all the angles and setpoints are converted to radians to keep consistency with the outputs from the open loop equations of motion
    """

    oriented_data = pd.DataFrame(
        {
            "controller.pitch": data["controller.pitch"],
            "controller.roll": data["controller.roll"],
            "controller.yaw": data["controller.yaw"],
            "controller.pitchRate": data["controller.pitchRate"],
            "controller.rollRate": data["controller.rollRate"],
            "controller.yawRate": data["controller.yawRate"],
            "controller.cmd_pitch": data["controller.cmd_pitch"],
            "controller.cmd_roll": data["controller.cmd_roll"],
            "controller.cmd_yaw": data["controller.cmd_yaw"],
            "controller.cmd_thrust": data["controller.cmd_thrust"],
            "motor.m1": data["motor.m1"],
            "motor.m2": data["motor.m2"],
            "motor.m3": data["motor.m3"],
            "motor.m4": data["motor.m4"],
            "p": data["p"],
            "q": -data["q"],
            "r": -data["r"],
            "acc.x": -data["acc.x"],
            "acc.y": data["acc.y"],
            "acc.z": data["acc.z"],
        }
    )

    time_array = np.arange(0, len(oriented_data) / sampling_freq, 1 / sampling_freq)

    idx = np.argmin(np.abs(time_array - time_shift))

    oriented_data = oriented_data.iloc[idx:, :]

    time_array =  np.arange(0, len(oriented_data) / sampling_freq, 1 / sampling_freq)


    oriented_data.insert(0, "time", time_array)

    return oriented_data


def optitrack_pipeline(data, filter_freq, CoM_vector):

    print("Processing the Optitrack data ...")

    # Get Optitrack meta data
    optitrack_meta = get_optitrack_meta(config.optitrack_path)
    optitrack_fps = int(float(optitrack_meta["Capture Frame Rate"]))

    # Handle NaNs in both dataframes
    optitrack_data_nonan = handle_nan(data, optitrack_fps, 2)

    optitrack_freq_dihedral = process_frequency_dihedral(optitrack_data_nonan, optitrack_fps)

    optitrack_freq_dihedral = filter_data(
        optitrack_freq_dihedral, filter_freq, optitrack_fps
    ).iloc[:, 1:]

    print(optitrack_freq_dihedral.columns)
    # Filtering
    optitrack_filtered = filter_data(
        optitrack_data_nonan, filter_freq, optitrack_fps
    )

    print("Computing states in body coordinates ...")
    # Process the optitrack data
    optitrack_processed = process_optitrack(optitrack_filtered, CoM_vector, optitrack_fps)

    output = pd.concat([optitrack_processed, optitrack_freq_dihedral], axis=1)

    print("Processing of the Optitrack data completed.")

    return optitrack_fps, output

def onboard_pipeline(data, freq, filter_freq, optitrack_freq):

    print("Processing the Optitrack data ...")
    # Handle NaNs in both dataframes
    onboard_data_nonan = handle_nan(data, onboard_freq, 2)

    # Filtering
    onboard_filtered = filter_data(onboard_data_nonan, filter_freq, freq)

    print("Orienting the IMU data")
    # Process the filtered onboard data
    onboard_processed = process_onboard(onboard_filtered, freq)

    print(f"Resampling down the IMU data from {freq} Hz to the Optitrack frame rate of {optitrack_freq} Hz")
    # Resample the onboard data down to optitrack_fps
    onboard_sampled = resample_data(onboard_processed, optitrack_freq, freq, freq)

    print("Processing of the Onboard data completed.")
    return onboard_sampled

def sync_dataframes(onboard, optitrack, optitrack_fps, cols_sync):
    print("Combine the data")

    # Match the data
    lag = find_lag(onboard, optitrack, cols_sync)

    # Shift the optitrack data
    optitrack_processed_shifted, time_shift = shift_data(
        optitrack_processed, lag, optitrack_fps
    )

    # Merge synced DataFrames
    processed_merged = merge_dfs(
        onboard_processed, optitrack_processed_shifted, optitrack_fps
    )

    # After calculating lag
    print(f"Lag: {lag} samples = {lag/optitrack_fps:.3f} seconds")

    return processed_merged


if __name__ == "__main__":
    # Read the .csv file into a pandas df
    optitrack_data = pd.read_csv(
        config.optitrack_path,
        skiprows=7,
        usecols=range(1, len(config.optitrack_cols) + 1),
        names=config.optitrack_cols,
        header=None,
    )

    onboard_data = pd.read_csv(config.onboard_path, header=0, names=config.onboard_cols)

    optitrack_fps, optitrack_processed = optitrack_pipeline(optitrack_data, filter_cutoff_freq, body_to_CoM)

    onboard_processed = onboard_pipeline(onboard_data, onboard_freq, filter_cutoff_freq, optitrack_fps)

    processed_merged = sync_dataframes(onboard_processed, optitrack_processed, optitrack_fps, columns_sync)


    # Save merged DataFrame
    os.makedirs(os.path.dirname(config.processed_path), exist_ok=True)
    processed_merged.to_csv(f"{config.processed_path}{config.flight_exp}-processed.csv", index=False)

    print("Processed data saved at", f"{config.processed_path}{config.flight_exp}-processed.csv")
    # # Process the onboard data at 500 Hz
    # onboard_data = pd.read_csv(onboard_csv, header=0, names=names_onboard)
    # oriented_data = orient_onboard(onboard_data, onboard_freq, time_shift)
    # oriented_data.to_csv(
    #     f"{processed_dir}/{flight_exp}_oriented_onboard.csv", index=False
    # )

    # Plot to verify
    if show:
        fig, axes = plt.subplots(3, 1, figsize=(18, 6))

        # After shifting
        axes[0].plot(processed_merged["time"], processed_merged["onboard.acc.x"], label="Onboard")
        axes[0].plot(processed_merged["time"], processed_merged["optitrack.acc.x"], label="OptiTrack (shifted)")
        axes[0].legend()
        axes[0].set_ylim([-2, 2])
        axes[0].set_title("acc x")

        axes[1].plot(processed_merged["time"], processed_merged["onboard.acc.y"], label="Onboard")
        axes[1].plot(processed_merged["time"], processed_merged["optitrack.acc.y"], label="OptiTrack (shifted)")
        axes[1].legend()
        axes[1].set_title("acc y")

        axes[2].plot(processed_merged["time"], processed_merged["onboard.acc.z"], label="Onboard")
        axes[2].plot(processed_merged["time"], processed_merged["optitrack.acc.z"], label="OptiTrack (shifted)")
        axes[2].legend()
        axes[2].set_title("acc z")
        plt.tight_layout()
        plt.show()






