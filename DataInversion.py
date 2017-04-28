import numpy as np         # Data manipulation
import matplotlib.pyplot as plt
import pandas as pd        # Use to read in csv file
import tkinter as tk       # Use to open file dialog
from tkinter import filedialog
import re                  # Regular Expressions for parsing strings

root = tk.Tk()
file_path = filedialog.askopenfilename()
root.withdraw()

"""
DataInversion.py
Author: John Vogel, john.eric.vogel@gmail.com
This file currently reads in a .csv file, parses the name, and
divides the data up by location in preparation for inversion
"""

# -----------------------------------------------------------------
# Parse the filename to get dimensions of matrix
# -----------------------------------------------------------------
fp_len = file_path.count('/')               # Number of splits in filepath at /
fp_split = file_path.split('/', fp_len)     # Split up filepath into strings
fp_name = fp_split[-1]                      # Grab just the filename
fp_len = fp_name.count('_')                 # New number of splits at _
fp_split = fp_name.split('_', fp_len)       # Split at _ characters
fp_name2 = fp_split[1]                      # Grab just the measurement charac
fp_len = fp_name2.count('x')                # Number of splits at x
fp_split = fp_name2.split('x', fp_len)      # split at x characters

row = fp_split[0]       # get number of rows
col = fp_split[1]       # get number of columns
samp = fp_split[2]      # get number of samples per location
spac = re.search(r'\d*\.*\d*', fp_split[3]).group()      # get spacing

data = pd.read_csv(file_path, delimiter=",").values  # Get values out of csv

# -----------------------------------------------------------------
# Grab data and organize by location, this allows you to invert
# one location at a time.
# -----------------------------------------------------------------
data[:, 9] = data[:, 9]/1000    # Normalize to units of mm
pastLoc = data[0, 9]            # Grab the current location value
loc = 0                         # Location Counter
locInd = 0                      # Current location index
locData = []
dataTemp = np.zeros((int(samp)*12-1, 30))
for ind in range(0, len(data)):     # Iterate through All Data
    currLoc = data[ind, 9]          # Current location
    if currLoc != pastLoc:          # If we've switched location
        loc = loc + 1               # increment location counter
        pastLoc = currLoc           # Set past location
        locInd = ind                # Record where we switched locations
        locData.append(dataTemp)    # Store values in a list of matrices
        dataTemp = np.zeros((int(samp)*12, 30))  # Reinitialize matrix
    dataTemp[ind-locInd, :] = data[ind, :]       # Measurements for location
locData.append(dataTemp)            # Drop in the last set of values



plt.plot(data[:, 9])
