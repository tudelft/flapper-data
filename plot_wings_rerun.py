import pandas as pd
import rerun as rr
import rerun.blueprint as rrb
import numpy as np
from scipy.spatial.transform import Rotation as R

# Local imports
from process_data import process_optitrack, handle_nan, filter_data


wing_body_distance = 0.073
axes_radius = 0.002
marker_radius = 0.02
line_radius = 0.007

flight_exp = "flight_002"

data_path = f"data/raw/{flight_exp}/{flight_exp}_optitrack.csv"

# Frame definition: x forward, y left, z up
# OptiTrack z,x,y --> x,y,z, switch also for quaternions


# def mirror_point_through_plane(P, A, B, C):
#     # P: point to mirror (np.array([x, y, z]))
#     # A, B, C: three points on the plane (np.array([x, y, z]))
#     # Compute plane normal
#     AB = B - A
#     AC = C - A
#     n = np.cross(AB, AC)
#     n = n / np.linalg.norm(n)
#     # Compute vector from A to P
#     AP = P - A
#     # Distance from P to plane
#     d = np.dot(AP, n)
#     # Mirrored point
#     P_mirror = P - 2 * d * n
#     return P_mirror

angle_values_R = []
angle_values_L = []

# Common parameters
WINDOW_SIZE = 16
TARGET_FFT_SIZE = 256
SAMPLE_RATE = 100
FREQ_RANGE = (5, 25)

def calculate_flapping_frequency(signal_window, sample_rate=100, fft_size=256, freq_range=(5, 25)):
    """Calculate dominant frequency from signal window using FFT"""
    signal = signal_window - np.mean(signal_window)
    
    # Apply zero-padding
    padded_signal = np.zeros(fft_size)
    padded_signal[:len(signal)] = signal
    
    # Compute FFT
    freqs = np.fft.fftfreq(fft_size, 1/sample_rate)
    fft_vals = np.abs(np.fft.fft(padded_signal))
    
    # Find dominant frequency in specified range
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    valid_freqs = freqs[mask]
    valid_fft = fft_vals[mask]
    
    if len(valid_freqs) > 0:
        dominant_idx = np.argmax(valid_fft)
        return valid_freqs[dominant_idx]
    return None

def calculate_dihedral_angle(forward_body, norm_dihedral):
    """Calculate dihedral angle with proper sign"""
    cross_product = np.cross(forward_body, norm_dihedral)
    angle = np.arcsin(np.linalg.norm(cross_product) / 
                     (np.linalg.norm(forward_body) * np.linalg.norm(norm_dihedral)))
    
    # Determine sign based on z-component of cross product
    return angle if cross_product[2] >= 0 else -angle


blueprint = rrb.Blueprint(
    rrb.Vertical(
        rrb.Spatial3DView(
            origin="/flapper/",
            name="flapper",
        ),
        rrb.TimeSeriesView(origin="/dihedral/", name="dihedral", visible=False),
        rrb.TimeSeriesView(origin="/rotations/", name="rotations", visible=False),
        rrb.TimeSeriesView(origin="/frequency/", name="frequency", visible=True),

    ),
    collapse_panels=False,
)


def log_body_markers(df, i, marker_radius):
    for idx in range(1, 6):
        rr.log(
            f"/flapper/fb_body_{idx}",
            rr.Points3D(
                [
                    df[f"fb{idx}z"].iloc[i],
                    df[f"fb{idx}x"].iloc[i],
                    df[f"fb{idx}y"].iloc[i],
                ],
                colors=[0, 255, 0],
                radii=[marker_radius],
            ),
        )
    rr.log(
        "/flapper/fb_body",
        rr.Points3D(
            [df["fbz"].iloc[i], df["fbx"].iloc[i], df["fby"].iloc[i]],
            colors=[0, 255, 0],
            radii=[marker_radius],
        ),
    )


def log_body_strips(df, i, line_radius):
    rr.log(
        "/flapper/fb_body_strips",
        rr.LineStrips3D(
            [
                [
                    [df["fb1z"].iloc[i], df["fb1x"].iloc[i], df["fb1y"].iloc[i]],
                    [df["fbz"].iloc[i], df["fbx"].iloc[i], df["fby"].iloc[i]],
                    [df["fb2z"].iloc[i], df["fb2x"].iloc[i], df["fb2y"].iloc[i]],
                    [df["fb3z"].iloc[i], df["fb3x"].iloc[i], df["fb3y"].iloc[i]],
                    [df["fb5z"].iloc[i], df["fb5x"].iloc[i], df["fb5y"].iloc[i]],
                    [df["fb4z"].iloc[i], df["fb4x"].iloc[i], df["fb4y"].iloc[i]],
                    [df["fb2z"].iloc[i], df["fb2x"].iloc[i], df["fb2y"].iloc[i]],
                ],
                [
                    [df["fb3z"].iloc[i], df["fb3x"].iloc[i], df["fb3y"].iloc[i]],
                    [df["fbz"].iloc[i], df["fbx"].iloc[i], df["fby"].iloc[i]],
                    [df["fb4z"].iloc[i], df["fb4x"].iloc[i], df["fb4y"].iloc[i]],
                ],
                [
                    [df["fb5z"].iloc[i], df["fb5x"].iloc[i], df["fb5y"].iloc[i]],
                    [df["fbz"].iloc[i], df["fbx"].iloc[i], df["fby"].iloc[i]],
                ],
            ],
            radii=[line_radius, line_radius],
            colors=[[0, 255, 0], [0, 255, 0]],
        ),
    )


def log_wing_markers(df, i, wing, marker_radius):
    color = [255, 0, 0] if wing == "right" else [0, 0, 255]
    rr.log(
        f"/flapper/fb_{wing}_wing",
        rr.Points3D(
            [
                df[f"fb{wing[0]}wz"].iloc[i],
                df[f"fb{wing[0]}wx"].iloc[i],
                df[f"fb{wing[0]}wy"].iloc[i],
            ],
            colors=color,
            radii=[marker_radius],
        ),
    )
    for idx in [1, 2, 3]:
        rr.log(
            f"/flapper/fb_{wing}_wing_{idx}",
            rr.Points3D(
                [
                    df[f"fb{wing[0]}w{idx}z"].iloc[i],
                    df[f"fb{wing[0]}w{idx}x"].iloc[i],
                    df[f"fb{wing[0]}w{idx}y"].iloc[i],
                ],
                colors=color,
                radii=[marker_radius],
            ),
        )


def log_wing_strips(df, i, wing, line_radius):
    color = [255, 0, 0] if wing == "right" else [0, 0, 255]
    rr.log(
        f"/flapper/fb_{wing[0]}w",
        rr.LineStrips3D(
            [
                [
                    df[f"fb{wing[0]}w1z"].iloc[i],
                    df[f"fb{wing[0]}w1x"].iloc[i],
                    df[f"fb{wing[0]}w1y"].iloc[i],
                ],
                [
                    df[f"fb{wing[0]}w2z"].iloc[i],
                    df[f"fb{wing[0]}w2x"].iloc[i],
                    df[f"fb{wing[0]}w2y"].iloc[i],
                ],
                [
                    df[f"fb{wing[0]}w3z"].iloc[i],
                    df[f"fb{wing[0]}w3x"].iloc[i],
                    df[f"fb{wing[0]}w3y"].iloc[i],
                ],
            ],
            radii=[line_radius, line_radius, line_radius],
            colors=[color, color, color],
        ),
    )


def log_body_axes(df, i, axes_radius):
    origin = np.array([df["fbz"].iloc[i], df["fbx"].iloc[i], df["fby"].iloc[i]])

    quat = np.array(
        [
            df["fbqz"].iloc[i],
            df["fbqx"].iloc[i],
            df["fbqy"].iloc[i],
            df["fbqw"].iloc[i],
        ]
    )  # z, x, y for ForwardLeftUp reference frame
    r = R.from_quat(quat)
    r = r.apply(np.eye(3))

    rr.log(
        "/flapper/axes",
        rr.Arrows3D(
            origins=origin,
            vectors=r * 0.5,
            radii=axes_radius,
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ),
    )

def log_dihedral_frequency(df, i):
    # Body orientation
    body = np.array([df["fbz"].iloc[i], df["fbx"].iloc[i], df["fby"].iloc[i]])
    top_marker = np.array([df["fb1z"].iloc[i], df["fb1x"].iloc[i], df["fb1y"].iloc[i]])

    quat_body = np.array([
        df["fbqz"].iloc[i],
        df["fbqx"].iloc[i], 
        df["fbqy"].iloc[i],
        df["fbqw"].iloc[i],
    ])

    r_body = R.from_quat(quat_body, scalar_first=False)
    forward_body = r_body.apply([0, 1, 0])  # Forward facing vector

    # RIGHT WING CALCULATIONS
    wing_rootR = np.array([df["fbrwz"].iloc[i], df["fbrwx"].iloc[i], df["fbrwy"].iloc[i]])
    wing_lastR = np.array([df["fbrw3z"].iloc[i], df["fbrw3x"].iloc[i], df["fbrw3y"].iloc[i]])

    # Calculate wing plane normal
    BA_right = body - top_marker
    BC_right = wing_rootR - body
    norm_dihedral_right = np.cross(BA_right, BC_right)

    # Calculate flapping angle
    right_wing_vector = wing_lastR - wing_rootR
    flapping_angle_R = np.arcsin(
        np.dot(right_wing_vector, norm_dihedral_right) / 
        (np.linalg.norm(right_wing_vector) * np.linalg.norm(norm_dihedral_right))
    )
    angle_values_R.append(flapping_angle_R)

    # Calculate frequency for right wing
    if len(angle_values_R) >= WINDOW_SIZE:
        dominant_freq_R = calculate_flapping_frequency(
            angle_values_R[-WINDOW_SIZE:], 
            SAMPLE_RATE, 
            TARGET_FFT_SIZE, 
            FREQ_RANGE
        )
        if dominant_freq_R is not None:
            rr.log("/frequency/frequency_right", rr.Scalars(dominant_freq_R))

    # Calculate dihedral angle for right wing
    dihedral_R = calculate_dihedral_angle(forward_body, norm_dihedral_right)

    # LEFT WING CALCULATIONS  
    wing_rootL = np.array([df["fblwz"].iloc[i], df["fblwx"].iloc[i], df["fblwy"].iloc[i]])
    wing_lastL = np.array([df["fblw3z"].iloc[i], df["fblw3x"].iloc[i], df["fblw3y"].iloc[i]])

    # Calculate wing plane normal
    BA_left = body - top_marker
    BC_left = wing_rootL - body
    norm_dihedral_left = np.cross(BA_left, BC_left)

    # Calculate flapping angle
    left_wing_vector = wing_lastL - wing_rootL
    flapping_angle_L = np.arcsin(
        np.dot(left_wing_vector, norm_dihedral_left) / 
        (np.linalg.norm(left_wing_vector) * np.linalg.norm(norm_dihedral_left))
    )
    angle_values_L.append(flapping_angle_L)

    # Calculate frequency for left wing
    if len(angle_values_L) >= WINDOW_SIZE:
        dominant_freq_L = calculate_flapping_frequency(
            angle_values_L[-WINDOW_SIZE:],
            SAMPLE_RATE,
            TARGET_FFT_SIZE,
            FREQ_RANGE
        )
        if dominant_freq_L is not None:
            rr.log("/frequency/frequency_left", rr.Scalars(dominant_freq_L))

    # Calculate dihedral angle for left wing
    dihedral_L = calculate_dihedral_angle(forward_body, norm_dihedral_left)

    rr.log("/dihedral/dihedral_L", rr.Scalars(dihedral_L))
    rr.log("/dihedral/dihedral_R", rr.Scalars(dihedral_R))


if __name__ == "__main__":
    # Load the data from a CSV file
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
    df = pd.read_csv(
        data_path,
        skiprows=7,
        usecols=range(1, len(names) + 1),
        names=names,
        header=None,
    )
    df = df.iloc[0:7500, :]

    rr.init("rerun_flapper", spawn=True)
    rr.send_blueprint(blueprint)

    df = handle_nan(df, 100, 2)
    filtered_optitrack = filter_data(df, 8, 100)
    processed_optitrack = process_optitrack(filtered_optitrack, "ForwardRightDown", np.array([0, 0, 0.6]))

    for i in range(len(df)):
        rr.set_time("time", timestamp=df["time"].iloc[i])
        log_body_markers(df, i, marker_radius)
        log_body_strips(df, i, line_radius)
        log_wing_markers(df, i, "right", marker_radius)
        log_wing_markers(df, i, "left", marker_radius)
        log_wing_strips(df, i, "right", line_radius)
        log_wing_strips(df, i, "left", line_radius)
        log_body_axes(df, i, axes_radius)
        log_dihedral_frequency(df, i)


        rr.log("/rotations/pitch", rr.Scalars(processed_optitrack["pitch"].loc[i]))
        rr.log("/rotations/roll", rr.Scalars(processed_optitrack["roll"].loc[i]))
        rr.log("/rotations/yaw", rr.Scalars(processed_optitrack["yaw"].loc[i]))



         