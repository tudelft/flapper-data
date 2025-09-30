# flapper-data
Collection of scripts to proces data captured with a Flapper Drone

By running process_data.py the onboard IMU data and the optitrack data are filtered, resampled down to 100 Hz, and matched by means of regression. 
All the values in the columns "onboard. ..." are in degrees, while the ones related to the optitrack, thus "optitrack. ..." are in radians.

# Notes-to-self
Figure out whether the onboard oriented data should be returned in degrees or radians, keep in mind all the controllers on the flapper work with degrees. 

I think return everything in radians, transform in degreees inside the function in flapper_model.py in my personal repo.