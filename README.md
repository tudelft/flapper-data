# flapper-data
Collection of scripts to process data captured with a Flapper Drone: the onboard log and the OptiTrack recording of the same flight are synced and merged into one CSV at 200 Hz.

## Setup and usage

```bash
uv sync                               # create the environment
./run.sh process lateral1_rl          # process one flight
./run.sh process-all                  # process every flight in flights.yaml
./run.sh rerun lateral1_rl            # replay the mocap of a flight in the Rerun viewer
./run.sh rerun lateral1_rl --processed  # replay the processed flight instead
```

Run everything from the repository root: the data paths are relative to the current directory.

## Data layout

```
data/                                 # not tracked by git
├── onboard/
│   └── <log>.csv                     # e.g. rllog07.csv, decoded uSD-card log
├── mocap/
│   └── <take>.csv                    # e.g. lateral1_rl.csv, OptiTrack (Motive) export
└── processed/
    └── <flight>.csv                  # written by process_data
```

- **onboard/\<log\>.csv**: Crazyflie log at a nominal 200 Hz. It must contain `timestamp` (ms), `gyro.x/y/z` (deg/s), `locSrv.x/y/z` (the position streamed to the drone) and `controller.cmd_thrust`. All columns are carried through to the output unchanged.
- **mocap/\<take\>.csv**: Motive CSV export at 360 Hz (tested with format version 1.23, Y up). Only the pose of the drone's rigid body (`flapper_rl` by default) is used.

## Files

| File | What it does |
|---|---|
| `flights.yaml` | Pairs each onboard log with the mocap take of the same flight, plus the processing settings (rigid body name, mocap low-pass). |
| `src/flapper_data/process_data.py` | Main pipeline, described below. |
| `src/flapper_data/rerun_visuals.py` | Replays a flight in the [Rerun](https://rerun.io) viewer. From the raw mocap: body markers, body axes and position, plus the wing markers, dihedral angle and flapping frequency when the wing rigid bodies were recorded (`FlapperLeftWing`/`FlapperRightWing`, older recordings only). With `--processed`: body pose and trajectory, onboard gyro vs OptiTrack body rates, attitude, position and velocity. |
| `src/flapper_data/data_loader.py` | Reads `flights.yaml`, and the rigid-body pose and markers from the Motive export. Run it to list the flights. |
| `run.sh` | Shortcut for running the modules above. |

## Processing

1. **OptiTrack, at 360 Hz.** The rigid-body pose is rotated from the Motive frame to NED (body axes FRD). Frames where the rigid body is not tracked or is mis-solved (e.g. orientation flipped for a single frame) are dropped and interpolated. Position and quaternion are low-passed (zero phase, `mocap_cutoff_hz`, 30 Hz by default, above the ~16 Hz flapping frequency), then differentiated: the velocity by central differences, the body rates from the relative rotation between neighbouring frames.
2. **Sync.** The onboard timestamps are not accurate: the onboard clock runs ~1.1% slow compared to OptiTrack. The pipeline fits `mocap time = scale * onboard time + offset`: a coarse offset from cross-correlating `locSrv` with the OptiTrack position, then scale and offset maximising the cross-correlation of the gyro with the OptiTrack body rates. It prints the result and the gyro/OptiTrack correlation while flying (low-passed at 5 Hz), which should be close to 1. If the sync fails, `clock_scale` and `time_offset` can be set per flight in `flights.yaml`.
3. **Merge, at 200 Hz.** The output covers the time both systems were recording, on a uniform 200 Hz grid. Each row takes the nearest onboard sample, unchanged (the log runs at ~199.8 Hz, so a sample is repeated every few seconds), and the OptiTrack states interpolated at that time.

## Output

Each row of `<flight>.csv` is one 200 Hz sample; `time` starts at 0 s. The other columns start with `onboard.` or `optitrack.`:

- `onboard.*`: the columns of the onboard log, unfiltered and in their original units and frames.
- `optitrack.ned.x/y/z` [m]: position of the rigid body in NED (origin at the OptiTrack origin).
- `optitrack.ned.qw/qx/qy/qz`, `optitrack.ned.roll/pitch/yaw` [rad]: attitude of the FRD body frame relative to NED, as a quaternion and as ZYX Euler angles.
- `optitrack.ned.velx/vely/velz` [m/s]: velocity in NED.
- `optitrack.frd.velx/vely/velz` [m/s]: velocity in the body frame.
- `optitrack.frd.p/q/r` [rad/s]: body rates.

Rows that fall in a mocap dropout longer than 0.05 s have NaN OptiTrack columns.
