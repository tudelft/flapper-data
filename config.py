import csv
import re
from dataclasses import dataclass

# Mapping from OptiTrack rigid body names to short prefixes used in code.
# Add entries here if new rigid bodies appear in your recordings.
_BODY_PREFIX = {
    "FlapperBody": "fb",
    "FlapperLeftWing": "fblw",
    "FlapperRightWing": "fbrw",
}


@dataclass
class Config:
    flight_exp: str
    processed_path: str
    onboard_path: str
    optitrack_path: str
    optitrack_cols: list


def _parse_optitrack_columns(csv_path):
    """Auto-detect OptiTrack column names from the CSV header rows.

    Reads rows 2 (Type) and 3 (Name) of the OptiTrack export to discover
    rigid bodies, their markers, and the correct column ordering.
    Only 'Rigid Body' and 'Rigid Body Marker' columns are kept;
    trailing raw 'Marker' columns are ignored.
    """
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        rows = [next(reader) for _ in range(7)]

    types = rows[2][2:]  # skip Frame, Time columns
    names = rows[3][2:]
    measures = rows[5][2:]
    axes = rows[6][2:]

    cols = ["time"]

    for typ, name, measure, axis in zip(types, names, measures, axes):
        if typ == "Marker":
            continue  # skip raw unlabeled markers

        body_name = name.split(":")[0]
        prefix = _BODY_PREFIX[body_name]
        ax = axis.lower()

        if typ == "Rigid Body":
            if measure == "Rotation":
                cols.append(f"{prefix}q{ax}")        # e.g. fbqx
            else:
                cols.append(f"{prefix}{ax}")          # e.g. fbx
        elif typ == "Rigid Body Marker":
            marker_num = int(re.search(r"(\d+)$", name).group(1))
            cols.append(f"{prefix}{marker_num}{ax}")  # e.g. fb1x

    return cols


def load(flight: str) -> Config:
    """Build and return a Config for the given flight experiment."""
    optitrack_path = f"data/raw/{flight}/optitrack-{flight}.csv"
    return Config(
        flight_exp=flight,
        processed_path=f"data/processed/{flight}/",
        onboard_path=f"data/raw/{flight}/onboard-{flight}.csv",
        optitrack_path=optitrack_path,
        optitrack_cols=_parse_optitrack_columns(optitrack_path),
    )