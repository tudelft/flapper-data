# flapper-data

Collection of scripts to process data captured with a Flapper Drone.

## `convert_test_data.py`

`convert_test_data.py` is a small helper script for importing and
converting Crazyflie SD card log data into the local `data/` folder.

When run, it will:

- Create the next available folder named `data/flowdeck_cyberzoo_XXX`.
- Ask for a short description of the experiment and store it in
	`notes.txt` inside that folder.
- Copy all non‑text files from the SD card mount point
	`/media/sstroobants/FlapperSD` into the new folder.
- Rename the first `.csv` file it finds there to `opti.csv`
	(OptiTrack data).
- Find the first Crazyflie SD log binary (`log00`, `log01`, …),
	decode it using `cfusdlog`, and save all fixed‑frequency log
	signals to `sd.csv` in the same folder.

### Data on SD card
The SD card should contain 1 crazyflie log file, and the corresponding converted optitrack .csv.
To make compatible, ensure the optitrack file is named `opti.csv`. 

### Usage

1. Insert and mount the Crazyflie SD card so it appears at
	 `/media/sstroobants/FlapperSD` (or adjust the `src_folder` path
	 in the script).
2. From the repository root, run the script with Python:

	 ```bash
	 python convert_test_data.py
	 ```

3. Enter a short description when prompted. The script will then copy
	 the data and generate the `sd.csv` file in the newly created
	 `data/flowdeck_cyberzoo_XXX` directory.

## `plot_flow.py`

`plot_flow.py` loads a selected experiment from the `data/` folder and
generates a set of comparison plots between the Crazyflie state
estimator and OptiTrack measurements.

For the experiment specified in `CONFIG['test_name']` inside the
script, it will:

- Load `sd.csv` (Crazyflie SD data) and `opti.csv` (OptiTrack data).
- Align both time series based on height (z/y) by minimising the MSE
	between them.
- Derive velocities from OptiTrack positions and rotate them into the
	body frame using quaternions.
- Rotate Crazyflie estimator velocities into the body frame as well.
- Produce plots comparing:
	- flow vs gyro signals,
	- Kalman filter predicted vs measured flow,
	- body-frame velocities (estimator vs OptiTrack vs targets),
	- position traces (x, y, z),
	- attitude (roll, pitch, yaw) vs OptiTrack and command signals.

### Usage

1. Make sure you have already generated a `data/flowdeck_cyberzoo_XXX`
	folder using `convert_test_data.py` or by placing compatible
	`sd.csv` and `opti.csv` files there.
2. Open `plot_flow.py` and set `CONFIG['test_name']` to the desired
	folder name (e.g. `"flowdeck_cyberzoo_032"`).
3. From the repository root, run:

	```bash
	python plot_flow.py
	```

4. A number of matplotlib windows will open showing the different
	comparison plots.
