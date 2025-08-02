import pandas as pd
import rerun as rr
import rerun.blueprint as rrb
import numpy as np
import scipy.spatial.transform

wing_body_distance = 0.073
axes_radius = 0.002
marker_radius = 0.02
line_radius = 0.007

flight_exp = "flight_002"

data_path = f"data/{flight_exp}/{flight_exp}_optitrack.csv"

# Frame definition: x forward, y left, z up
# OptiTrack z,x,y --> x,y,z, remember to switch also for quaternions


def mirror_point_through_plane(P, A, B, C):
    # P: point to mirror (np.array([x, y, z]))
    # A, B, C: three points on the plane (np.array([x, y, z]))
    # Compute plane normal
    AB = B - A
    AC = C - A
    n = np.cross(AB, AC)
    n = n / np.linalg.norm(n)
    # Compute vector from A to P
    AP = P - A
    # Distance from P to plane
    d = np.dot(AP, n)
    # Mirrored point
    P_mirror = P - 2 * d * n
    return P_mirror


blueprint = rrb.Blueprint(
    rrb.Vertical(
        rrb.Spatial3DView(
            origin="/flapper/",
            name="flapper",
        ),
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
    ) # z, x, y for ForwardLeftUp reference frame

    r = scipy.spatial.transform.Rotation.from_quat(quat, scalar_first=False)
    r = r.apply(np.eye(3))

    rr.log(
        "/flapper/axes",
        rr.Arrows3D(
            origins=origin,
            vectors=r*0.5,
            radii=axes_radius,
            colors=[[0, 255, 0], [255, 0, 0], [0, 0, 255]],
        ),
    )


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

    df = df.iloc[60:2000, :]

    rr.init("rerun_flapper", spawn=True)
    rr.send_blueprint(blueprint)

    for i in range(len(df)):
        rr.set_time("time", timestamp=df["time"].iloc[i])
        log_body_markers(df, i, marker_radius)
        log_body_strips(df, i, line_radius)
        log_wing_markers(df, i, "right", marker_radius)
        log_wing_markers(df, i, "left", marker_radius)
        log_wing_strips(df, i, "right", line_radius)
        log_wing_strips(df, i, "left", line_radius)
        log_body_axes(df, i, axes_radius)
