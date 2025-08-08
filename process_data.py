import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from math import gcd
from scipy import signal
from scipy import integrate
import matplotlib.pyplot as plt

"""
TODO:
    - define the clearly the body and global values, dd _glob or _body to each variable

    rigid_body_names = ["FlapperBody", "FlapperRightWing", "FlapperLeftWing"]

    - Handle column names in a different manner
    - log flapping frequency
    - log dihedral
    - shift to CoM instead of geometric body
"""

# OptiTrack z,x,y --> x,y,z, switch also for quaternions

# Select the flight number
flight_exp = "flight_001"
onboard_freq = 500  # Hz
filter_cutoff_freq = 3  # Hz
g0 = 9.80665  # m/s

columns_sync = ["roll_rate"]

body_to_CoM = np.array([0, 0, 0]) # Not yet implemented


def get_optitrack_meta(optitrack_csv):
    with open(optitrack_csv, "r") as f:
        lines = f.readline()

    metadata_raw = lines.strip().split(",")
    metadata = dict(zip(metadata_raw[::2], metadata_raw[1::2]))

    if metadata["Format Version"] != "1.23":
        print("The code has not been tested with this Optitrack file version")

    return metadata


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


def resample_data(data, up, down):
    columns_name = data.columns[1:]
    g = gcd(up, down)
    up //= g
    down //= g

    data = signal.resample_poly(data.iloc[:, 1:], up, down)

    data = pd.DataFrame(data, columns=columns_name)

    time_array = np.round(np.linspace(0, data.shape[0] / 100, data.shape[0]), 2)

    data.insert(0, "time", time_array)

    return data


def filter_data(data, cutoff_freq, sampling_freq):
    """
    filter --> scipy.signal.butter
    use scipy.integrate.simpson or trapezoidal
    """
    columns_name = data.columns

    b, a = signal.butter(4, cutoff_freq, fs=sampling_freq)  # type: ignore

    filtered_data = signal.filtfilt(b, a, data.iloc[:, 1:], axis=0)

    filtered_df = pd.DataFrame(filtered_data, columns=columns_name[1:])

    time_array = np.round(np.linspace(0, filtered_df.shape[0] / sampling_freq, filtered_df.shape[0]), 2)

    filtered_df.insert(0, "time", time_array)

    return filtered_df


def process_optitrack(data, reference_frame, body_to_CoM):
    """
    Orients the optitrack data to the correct body orientation. Optitrack defines body axes
    as RightForwardUp respectively for x, y, z. Thus use a Euler intrinsic rotation, 'yxz',
    corresponding to roll, pitch, and yaw.

    Parameters:
    -----------
        data : pandas.DataFrame
            DataFrame containing the raw OptiTrack data
        frame : str
            Defines the orientation the data is processed to.
            Can be either "ForwardLeftUp" or "ForwardRightDown" as from aerospace convention,
            indicating respectively x, y, z.

    Returns:
    --------
        oriented_data: pandas.DataFrame
            TBD: DataFrame containing 6 columns, ['x', 'y', 'z', 'roll', 'pitch', 'yaw'],
            angles are in radians.
    """

    # For now process only the body data
    if reference_frame == "ForwardRightDown":
        # Rotational kinematics
        quats = np.vstack((data["fbqx"], data["fbqy"], data["fbqz"], data["fbqw"])).T

        r = R.from_quat(quats, scalar_first=False)

        # Optitrack defines Y-up, Z-right, X-forward
        euler_angles = r.as_euler("YZX", degrees=False)

        # First extract yaw, pitch and then roll
        psi, theta, phi = -euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]
        
        phi_rate = np.gradient(phi, data["time"])
        theta_rate = np.gradient(theta, data["time"])
        psi_rate = np.gradient(psi, data["time"])
        euler_rates = np.asarray([phi_rate, theta_rate, psi_rate]) # (3, N)
        
        
        rotations_rates = np.array(
            [[np.ones_like(theta), np.zeros_like(theta), -np.sin(theta)], 
             [np.zeros_like(theta), np.cos(phi), np.sin(phi) * np.cos(theta)], 
             [np.zeros_like(theta), -np.sin(phi), np.cos(phi) * np.cos(theta)]]
        ) # (3, 3, N)
        
        roll_rate, pitch_rate, yaw_rate = np.einsum('ijk,jk->ik', rotations_rates, euler_rates)
        
        
        # Now move onto translational kinematics
        
        x_glob = np.array(data["fbx"])
        y_glob = np.array(data["fbz"])
        z_glob = np.array(-data["fby"])
        
    else:
        print("Reference frame not recognised, use 'ForwardLeftUp' or 'ForwardRightDown' (aerospace standard)")

    processed_data = pd.DataFrame(
        {
            "time": data["time"],
            "roll": phi,
            "pitch": theta,
            "yaw": psi,
            "roll_rate": roll_rate,
            "pitch_rate": pitch_rate,
            "yaw_rate": yaw_rate,
        }
    )

    return processed_data


def process_onboard(data, reference_frame):
    
    if reference_frame == "ForwardRightDown":
        # Using reference frame Forward, Right, Down
        pitch_rate = np.radians(-data["gyro.y"])
        roll_rate = np.radians(data["gyro.x"])
        yaw_rate = np.radians(-data["gyro.z"])
        
        roll_alpha = np.gradient(roll_rate, data["time"])
        pitch_alpha = np.gradient(pitch_rate, data["time"])
        yaw_alpha = np.gradient(yaw_rate, data["time"])
    else:
        print("Reference frame not recognised, use 'ForwardLeftUp' or 'ForwardRightDown' (aerospace standard)")

    processed_data = pd.DataFrame(
        {
            "time": data["time"],
            "roll_rate": roll_rate,
            "pitch_rate": pitch_rate,
            "yaw_rate": yaw_rate,
            "roll_alpha": roll_alpha,
            "pitch_alpha": pitch_alpha,
            "yaw_alpha": yaw_alpha,
        }
    )

    return processed_data


def sync_timestamps(onboard, optitrack, columns_sync):
    x = onboard[columns_sync].values
    y = optitrack[columns_sync].values

    correlation = signal.correlate(x, y, mode="full")
    lags = signal.correlation_lags(x.size, y.size, mode="full")
    lag = lags[np.argmax(correlation)]

    return lag


def shift_data(data, lag, sampling_freq):
    time_shift = lag / sampling_freq

    data["time"] = np.round(np.linspace(time_shift, data["time"].iloc[-1] + time_shift, len(data["time"])), 2)

    return data


if __name__ == "__main__":
    data_dir = f"data/{flight_exp}/{flight_exp}"
    optitrack_csv = f"{data_dir}_optitrack.csv"
    onboard_csv = f"{data_dir}_flapper.csv"

    # Give a name to the columns
    names = [
        "time",
        "fbqx",
        "fbqy",
        "fbqz",
        "fbqw",
        "fbx",
        "fby",
        "fbz",
        "fb1x",
        "fb1y",
        "fb1z",
        "fb2x",
        "fb2y",
        "fb2z",
        "fb3x",
        "fb3y",
        "fb3z",
        "fb4x",
        "fb4y",
        "fb4z",
        "fb5x",
        "fb5y",
        "fb5z",
        "fbrwqx",
        "fbrwqy",
        "fbrwqz",
        "fbrwqw",
        "fbrwx",
        "fbrwy",
        "fbrwz",
        "fbrw1x",
        "fbrw1y",
        "fbrw1z",
        "fbrw2x",
        "fbrw2y",
        "fbrw2z",
        "fbrw3x",
        "fbrw3y",
        "fbrw3z",
        "fblwqx",
        "fblwqy",
        "fblwqz",
        "fblwqw",
        "fblwx",
        "fblwy",
        "fblwz",
        "fblw1x",
        "fblw1y",
        "fblw1z",
        "fblw2x",
        "fblw2y",
        "fblw2z",
        "fblw3x",
        "fblw3y",
        "fblw3z",
    ]

    # Read the .csv file into a pandas df
    optitrack_data = pd.read_csv(
        optitrack_csv,
        skiprows=7,
        usecols=range(1, len(names) + 1),
        names=names,
        header=None,
    )

    onboard_data = pd.read_csv(onboard_csv, usecols=lambda col: col != "timestamp")

    optitrack_meta = get_optitrack_meta(optitrack_csv)

    optitrack_fps = int(float(optitrack_meta["Capture Frame Rate"]))

    # Handle NaNs in both dataframes
    optitrack_data = handle_nan(optitrack_data, optitrack_fps, 2)
    onboard_data = handle_nan(onboard_data, onboard_freq, 2)
    
    # Filter the onboard data
    onboard_filtered = filter_data(onboard_data, filter_cutoff_freq, onboard_freq)
    
    # Filter the optitrack data
    optitrack_filtered = filter_data(optitrack_data, filter_cutoff_freq, optitrack_fps)

    # Resample the onboard data down to 100 hz or optitrack_fps
    onboard_sampled = resample_data(onboard_filtered, optitrack_fps, onboard_freq)

    # Process the filtered onboard data
    onboard_processed = process_onboard(onboard_sampled, "ForwardRightDown")
    # Process the optitrack data
    optitrack_processed = process_optitrack(optitrack_filtered, "ForwardRightDown", body_to_CoM)

    # Match the data
    lag = sync_timestamps(onboard_processed, optitrack_processed, columns_sync)

    optitrack_processed = shift_data(optitrack_processed, lag, optitrack_fps)

    plt.plot(onboard_processed["time"], onboard_processed["roll_rate"], label="onboard")
    plt.plot(optitrack_processed["time"], optitrack_processed["roll_rate"], label="optitrack")
    plt.ylim(-4, 4)
    plt.legend()
    plt.show()
