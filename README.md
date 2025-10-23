# flapper-data
Collection of scripts to proces data captured with a Flapper Drone

By running process_data.py the onboard IMU data and the optitrack data are filtered, resampled down to 100 Hz, and matched by means of regression. 
All the values in the columns "onboard. ..." are in degrees, while the ones related to the optitrack, thus "optitrack. ..." are in radians.