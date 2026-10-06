import csv
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

_FLIGHTS_YAML = Path(__file__).resolve().parents[2] / "flights.yaml"

# Prefixes of the wing rigid bodies in the raw mocap columns, see read_mocap_markers
_WING_PREFIX = {
    "FlapperLeftWing": "fblw",
    "FlapperRightWing": "fbrw",
}


@dataclass
class Config:
    flight: str
    onboard_path: str
    mocap_path: str
    processed_path: str
    rigid_body: str
    mocap_cutoff_hz: float
    clock_scale: float | None = None  # mocap time = clock_scale * onboard time + time_offset
    time_offset: float | None = None


def _read_yaml() -> dict:
    with open(_FLIGHTS_YAML) as f:
        return yaml.safe_load(f)


def flights() -> list[str]:
    """Names of the flights listed in flights.yaml."""
    return list(_read_yaml()["flights"])


def load(flight: str) -> Config:
    """Build the Config of a flight from flights.yaml."""
    data = _read_yaml()
    if flight not in data["flights"]:
        raise KeyError(f"Flight '{flight}' is not listed in {_FLIGHTS_YAML.name}")
    settings = {**data["defaults"], **data["flights"][flight]}
    return Config(
        flight=flight,
        onboard_path=f"data/onboard/{settings['onboard']}.csv",
        mocap_path=f"data/mocap/{settings['mocap']}.csv",
        processed_path=f"data/processed/{flight}.csv",
        rigid_body=settings["rigid_body"],
        mocap_cutoff_hz=float(settings["mocap_cutoff_hz"]),
        clock_scale=settings.get("clock_scale"),
        time_offset=settings.get("time_offset"),
    )


def _read_mocap_header(path: str) -> tuple[list, float]:
    """Return the 7 header rows of a Motive CSV export and its capture frame rate."""
    with open(path) as f:
        reader = csv.reader(f)
        rows = [next(reader) for _ in range(7)]

    metadata = dict(zip(rows[0][::2], rows[0][1::2]))
    if metadata["Format Version"] != "1.23":
        print(f"Warning: {path} has format version {metadata['Format Version']}, only 1.23 is tested")
    return rows, float(metadata["Capture Frame Rate"])


def read_mocap_markers(path: str, rigid_body: str) -> tuple[pd.DataFrame, float]:
    """Read the raw rigid bodies and their markers from a Motive CSV export.

    The drone rigid body gets the prefix 'fb'; the wing rigid bodies of older
    recordings get 'fblw' and 'fbrw'. Other rigid bodies and unlabeled markers
    are ignored. Columns are named e.g. fbx (position), fbqx (quaternion) and
    fb1x (marker 1), in the Motive frame. Returns them with the time column, and
    the capture frame rate.
    """
    rows, fps = _read_mocap_header(path)
    prefixes = {rigid_body: "fb", **_WING_PREFIX}

    # Rows 2, 3, 5 and 6 hold the type, name, measure (Rotation/Position) and axis of each column
    cols = {1: "time"}
    for i, (typ, name, measure, axis) in enumerate(zip(rows[2], rows[3], rows[5], rows[6])):
        body = name.split(":")[0]
        if body not in prefixes:
            continue
        prefix = prefixes[body]
        ax = axis.lower()

        if typ == "Rigid Body":
            cols[i] = f"{prefix}q{ax}" if measure == "Rotation" else f"{prefix}{ax}"  # e.g. fbqx, fbx
        elif typ == "Rigid Body Marker":
            marker_num = int(re.search(r"(\d+)$", name).group(1))
            cols[i] = f"{prefix}{marker_num}{ax}"  # e.g. fb1x

    data = pd.read_csv(path, skiprows=7, header=None, usecols=list(cols))
    return data.rename(columns=cols), fps


def read_mocap(path: str, rigid_body: str) -> tuple[pd.DataFrame, float]:
    """Read the pose of one rigid body from a Motive CSV export.

    Returns a DataFrame with the columns time, qx, qy, qz, qw, x, y, z in the
    Motive frame (Y up), and the capture frame rate. Frames where the rigid body
    was not tracked are NaN.
    """
    rows, fps = _read_mocap_header(path)

    # Rows 3, 5 and 6 hold the name, measure (Rotation/Position) and axis of each column
    names, measures, axes = rows[3], rows[5], rows[6]
    cols = {}
    for i, (name, measure, axis) in enumerate(zip(names, measures, axes)):
        if name == rigid_body:
            prefix = "q" if measure == "Rotation" else ""
            cols[i] = f"{prefix}{axis.lower()}"
    if not cols:
        raise ValueError(f"Rigid body '{rigid_body}' not found in {path}")

    data = pd.read_csv(path, skiprows=7, header=None, usecols=[1, *cols])
    data = data.rename(columns={1: "time", **cols})
    return data[["time", "qx", "qy", "qz", "qw", "x", "y", "z"]], fps


if __name__ == "__main__":
    print(*flights())
