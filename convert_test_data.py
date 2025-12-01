# -*- coding: utf-8 -*-
"""convert_test_data
=====================

Small utility script to copy a Crazyflie SD card logging session
into the local ``data/`` folder and convert the binary log file
into a CSV file that is easier to analyse.

The script performs the following high‑level steps:

* Find the next free ``data/flowdeck_cyberzoo_XXX`` folder name.
* Ask the user for a short description and store it in ``notes.txt``.
* Copy all non‑text files from the SD card mount point into that folder.
* Rename the first ``.csv`` file it finds to ``opti.csv`` (OptiTrack data).
* Find the first ``logXX`` binary file, decode it with :mod:`cfusdlog`,
    and keep only the fixed‑frequency part of the log.
* Store all decoded log signals as columns in ``sd.csv`` inside
    the destination folder.

Run this script from the repository root while the SD card is mounted
at ``/media/sstroobants/FlapperSD`` (or adjust ``src_folder`` below).
"""
import cfusdlog
import matplotlib.pyplot as plt
import re
import argparse  # currently unused; kept for potential future CLI
import pandas as pd
import numpy as np
from scipy import stats  # currently unused; imported for historic reasons
import os
import shutil

# parser = argparse.ArgumentParser()
# parser.add_argument("--filename", type=str, default="data/")
# args = parser.parse_args()

# ---------------------------------------------------------------------------
# Determine destination folder
# ---------------------------------------------------------------------------

# Find the next available folder name of the form
#   data/flowdeck_cyberzoo_000, data/flowdeck_cyberzoo_001, ...
# by incrementing ``i`` until the path does not yet exist.
i = 0
while os.path.exists(f"data/flowdeck_cyberzoo_{i:03d}"):
    i += 1
dest_folder = f"data/flowdeck_cyberzoo_{i:03d}"
os.makedirs(dest_folder)

# ---------------------------------------------------------------------------
# Store user notes for the experiment
# ---------------------------------------------------------------------------

# Ask the user for a short description of the log/experiment and
# write it to ``notes.txt`` inside the destination folder so that
# the raw data is self‑documenting.
notes = input("Enter notes for this log: ")
with open(os.path.join(dest_folder, 'notes.txt'), 'w') as f:
    f.write(notes)

# ---------------------------------------------------------------------------
# Copy data from SD card
# ---------------------------------------------------------------------------

# Folder where the Crazyflie SD card is expected to be mounted.
# Adjust this path if your system uses a different mount point.
src_folder = "/media/sstroobants/FlapperSD"

# Copy every file from the SD card except ``*.txt`` helper files into
# the destination folder, preserving the original filenames.
for f in os.listdir(src_folder):
    src_file = os.path.join(src_folder, f)
    if os.path.isfile(src_file) and not f.endswith('.txt'):
        shutil.copy(src_file, dest_folder)

# ---------------------------------------------------------------------------
# Locate OptiTrack CSV and normalise its name
# ---------------------------------------------------------------------------

# Find the first ``.csv`` file in the destination folder and rename it to
# ``opti.csv``.  This assumes there is exactly one OptiTrack CSV per log.
for f in os.listdir(dest_folder):
    if f.endswith('.csv'):
        os.rename(os.path.join(dest_folder, f), os.path.join(dest_folder, 'opti.csv'))
        break

# ---------------------------------------------------------------------------
# Decode Crazyflie binary log into a pandas DataFrame
# ---------------------------------------------------------------------------

# Find the first file whose name matches ``log\d+`` – this is expected to be
# the Crazyflie SD log. Example filenames: ``log00``, ``log01``, ...
log_file_path = None
for f in os.listdir(dest_folder):
    if re.match(r'log\d+', f):
        log_file_path = os.path.join(dest_folder, f)
        break

if log_file_path is None:
    raise FileNotFoundError(f"No log file found in {dest_folder}")

logData = cfusdlog.decode(log_file_path)

# Only keep the regular fixed‑frequency log data. ``cfusdlog.decode`` returns
# a dictionary with different logging groups; here we use the
# ``'fixedFrequency'`` group which is most convenient for time‑series
# analysis.
logData = logData['fixedFrequency']

# Configure matplotlib default figure background to white.  The script does
# not generate plots at the moment, but this keeps behaviour consistent with
# the original code and is handy when plotting ``sd.csv`` interactively.
plt.rcParams['figure.facecolor'] = 'w'

# ``logData`` is a dictionary that maps signal names to 1‑D numpy arrays.
# Convert it into a pandas DataFrame where each key becomes a column.
data = pd.DataFrame()
for key, values in logData.items():
    data[key] = values

# Finally, write the decoded log to ``sd.csv`` in the destination folder.
# The CSV has one row per log sample and one column per signal.
data.to_csv(f'{dest_folder}/sd.csv', index=False)