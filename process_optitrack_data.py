import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

# OptiTrack z,x,y --> x,y,z, switch also for quaternions

# Select the flight number
flight_n = "flight_001"
data_path = f"data/{flight_n}/{flight_n}_optitrack.csv"


def orient_data(data, reference_frame):
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
            DataFrame containing 6 columns, ['x', 'y', 'z', 'roll', 'pitch', 'yaw'],
            angles are in radians.
    """

    # For now process only the body data
    if reference_frame == "ForwardLeftUp":
        x = list(data["fbz"])
        y = list(data["fbx"])
        z = list(data["fby"])

        quats = np.vstack((data["fbqz"], data["fbqx"], data["fbqy"], data["fbqw"])).T
        r = R.from_quat(quats, scalar_first=False)
        euler_angles = r.as_euler("yxz", degrees=False)

        roll = euler_angles[:, 0]
        pitch = -euler_angles[:, 1]  # Due to moving the axis from right to left
        yaw = euler_angles[:, 2]

    elif reference_frame == "ForwardRightDown":
        x = list(data["fbz"])
        y = list(-data["fbx"])
        z = list(-data["fby"])

        quats = np.vstack((data["fbqz"], data["fbqx"], data["fbqy"], data["fbqw"])).T
        r = R.from_quat(quats, scalar_first=False)
        euler_angles = r.as_euler("yxz", degrees=False)

        roll = euler_angles[:, 0]
        pitch = euler_angles[:, 1]
        yaw = -euler_angles[:, 2]  # Due to moving the axis from up to down
    else:
        print(
            "Reference frame not recognised, use ForwardLeftUp or ForwardRightDown (aerospace standard)"
        )

    oriented_data = pd.DataFrame(
        {"x": x, "y": y, "z": z, "roll": roll, "pitch": pitch, "yaw": yaw}
    )

    return oriented_data


def handle_nan(data, frame_rate: int, time_limit: int):
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


if __name__ == "__main__":
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
    data = pd.read_csv(
        data_path,
        skiprows=7,
        usecols=range(1, len(names) + 1),
        names=names,
        header=None,
    ).iloc[:, :]

    handle_nan(data, 100, 2)
    orient_data(data, "ForwardLeftUp")
