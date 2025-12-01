import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from scipy.signal import medfilt
from scipy.spatial.transform import Rotation as R

from ahrs.filters import Complementary, Mahony
from ahrs.common.orientation import q2euler

dataset_name = "data/flight_001/flight_001"

def load_opti_data():
        # The order of the markers might be different per logged dataset, but I don't think so.
    names = [
        "timestamp",
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
    df = pd.read_csv(
        f"{dataset_name}_optitrack.csv",
        skiprows=7,
        usecols=range(1, len(names) + 1),
        names=names,
        header=None,
    )
    return df

if __name__ == "__main__":
    data = pd.read_csv(f"{dataset_name}_flapper.csv")
    data["timestamp"] *= 1 / 990  # Convert timestamp from ms to seconds
    data["timestamp"] = data["timestamp"] - data["timestamp"].iloc[0]  # Normalize timestamp to start at 0
    data["timestamp"] -= 9.5
    opti_data = load_opti_data()

    # gyro from degrees/s to radians 
    data["gyro.x"] = np.deg2rad(data["gyro.x"])
    data["gyro.y"] = np.deg2rad(data["gyro.y"])
    data["gyro.z"] = np.deg2rad(data["gyro.z"])

    #acc from g to m/s^2
    data["acc.x"] = data["acc.x"] * 9.80665
    data["acc.y"] = data["acc.y"] * 9.80665
    data["acc.z"] = data["acc.z"] * 9.80665

    # Run complementary filter for entire dataset
    comp_filter = Mahony(gyr=data[["gyro.x", "gyro.y", "gyro.z"]].to_numpy(),
                               acc=data[["acc.x", "acc.y", "acc.z"]].to_numpy(),
                               frequency=500, kp=0.8, ki=0.002)

    eulers = []
    for i in range(len(data)):
        euler = q2euler(comp_filter.Q[i])
        eulers.append(euler)

    eulers = np.array(eulers)

    valid_quat = opti_data[['fbqx', 'fbqy', 'fbqz', 'fbqw']].notna().all(axis=1)
    opti_data_valid = opti_data[valid_quat]
    opti_headings = R.from_quat(opti_data_valid[['fbqz', 'fbqx', 'fbqy', 'fbqw']].values).as_euler('xyz')
    opti_pitch = pd.Series(medfilt(-opti_headings[:, 1], 9))  # Extract pitch
    opti_roll = pd.Series(medfilt(opti_headings[:, 0], 9))   # Extract roll
    opti_yaw = pd.Series(medfilt(opti_headings[:, 2], 9))  # Extract yaw (heading)

    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(data["timestamp"], eulers[:, 0], label="pitch")
    ax[0].plot(data["timestamp"], np.deg2rad(data["controller.pitch"]), label="pitch command", linestyle='--')
    ax[0].plot(opti_data_valid["timestamp"], opti_roll, label="opti pitch", linestyle='--')
    ax[0].axhline(0, color='grey', linewidth=2)
    ax[0].legend()
    ax[0].set_ylabel("Angle (rad)")

    ax[1].plot(data["timestamp"], eulers[:, 1], label="roll")
    ax[1].plot(data["timestamp"], np.deg2rad(data["controller.roll"]), label="roll command", linestyle='--')
    ax[1].plot(opti_data_valid["timestamp"], -opti_pitch, label="opti roll", linestyle='--')
    ax[1].axhline(0, color='grey', linewidth=2)
    ax[1].legend()
    ax[1].set_ylabel("Angle (rad)")

    ax[2].plot(data["timestamp"], -eulers[:, 2], label="yaw")
    ax[2].plot(data["timestamp"], np.deg2rad(data["controller.yaw"]), label="yaw command", linestyle='--')
    ax[2].plot(opti_data_valid["timestamp"], opti_yaw, label="opti yaw", linestyle='--')
    ax[2].axhline(0, color='grey', linewidth=2)
    ax[2].legend()
    ax[2].set_ylabel("Angle (rad)")

    plt.show()