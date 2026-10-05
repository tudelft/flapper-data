# flapper-data
Collection of scripts to process data captured with a Flapper Drone: the onboard log and the OptiTrack recording of the same flight are synced and merged into one CSV.

## Setup and usage

```bash
uv sync                               # create the environment
./run.sh process hover1               # process one flight
./run.sh process-all                  # process every flight listed in run.sh
./run.sh rerun hover1                 # replay a flight in the Rerun viewer
```

Run everything from the repository root: the data paths are relative to the current directory.

## Data layout

```
data/                                 # not tracked by git
├── raw/
│   └── <flight>/                     # e.g. hover1, climb2, flight_001
│       ├── onboard-<flight>.csv      # onboard log
│       └── optitrack-<flight>.csv    # OptiTrack (Motive) export
└── processed/
    └── <flight>-processed.csv        # written by process_data
```

- **onboard-\<flight\>.csv**: Crazyflie log at 200 Hz. The first column is the timestamp, and it must contain `acc.x/y/z` (g) and `gyro.x/y/z` (deg/s). All other columns (e.g. `controller.*`, `motor.*`) are carried through to the output.
- **optitrack-\<flight\>.csv**: Motive CSV export (tested with format version 1.23) with the rigid bodies `FlapperBody`, `FlapperLeftWing` and `FlapperRightWing`. The columns are detected from the header; unlabeled markers are ignored.

## Files

| File | What it does |
|---|---|
| `src/flapper_data/process_data.py` | Main pipeline. Low-passes and rotates the OptiTrack data to the body frame (attitude, rates, CoM velocity and acceleration) and computes the wing dihedral angles and flapping frequency. Low-passes the onboard IMU data, rotates it to the body frame and resamples it to the OptiTrack frame rate. Syncs the two by cross-correlating the body rates p, q, r, merges them, and removes gravity from the onboard accelerations using the OptiTrack attitude. Constants such as the onboard rate, filter cutoff and CoM offset are set at the bottom of the file. |
| `src/flapper_data/rerun_visuals.py` | Replays a flight in the [Rerun](https://rerun.io) viewer: body and wing markers, body axes, dihedral angle, flapping frequency and position. Uses the raw OptiTrack file, or the processed CSV with `--processed`. |
| `src/flapper_data/data_loader.py` | Builds the configuration for a flight: file paths, OptiTrack column names, and the settings from `datasets.yaml`. |
| `datasets.yaml` | Per-flight settings: `yaw_offset` to align the rigid-body x-axis with the nose, and `wing_markers` used for the dihedral angle. Flights that are not listed use the defaults. |
| `run.sh` | Shortcut for running the modules above on one or all flights. |

## Output

Each row of `<flight>-processed.csv` is one OptiTrack frame. Columns start with `onboard.` or `optitrack.`:

- Attitude, rates and their derivatives are in rad, rad/s and rad/s², velocities and accelerations in m/s and m/s², in the body frame (x forward, y right, z down).
- `onboard.controller.*` and the other passed-through log columns keep the units of the onboard log.
- `optitrack.fb*`, `optitrack.fblw*` and `optitrack.fbrw*` are the (low-passed) marker positions and quaternions of the body, left wing and right wing, in the OptiTrack frame.
- `optitrack.dihedral.*` are in rad, `optitrack.freq.*` in Hz.
