import numpy as np         # Data manipulation
import matplotlib.pyplot as plt
import pandas as pd        # Use to read in csv file
import tkinter as tk       # Use to open file dialog
from tkinter import filedialog
import re                  # Regular Expressions for parsing strings
from InversionFunctions import precompute
from InversionFunctions import cond
import time

root = tk.Tk()
file_path = filedialog.askopenfilename()
root.withdraw()

# -----------------------------------------------------------------
# Parse the filename to get dimensions of matrix
# -----------------------------------------------------------------
startT = time.time()
start = time.time()
fp_len = file_path.count('/')               # Number of splits in filepath at /
fp_split = file_path.split('/', fp_len)     # Split up filepath into strings
fp_name = fp_split[-1]                      # Grab just the filename
fp_len = fp_name.count('_')                 # New number of splits at _
fp_split = fp_name.split('_', fp_len)       # Split at _ characters
fp_name2 = fp_split[1]                      # Grab just the measurement charac
fp_len = fp_name2.count('x')                # Number of splits at x
fp_split = fp_name2.split('x', fp_len)      # split at x characters

row = int(fp_split[0])       # get number of rows
col = int(fp_split[1])       # get number of columns
samp = int(fp_split[2])      # get number of samples per location
spac = int(re.search(r'\d*\.*\d*', fp_split[3]).group())      # get spacing
# T = input('Enter film thickness (without current collector): ')
T = 60

data = pd.read_csv(file_path, delimiter=",",
                   header=None).values  # Get values out of csv

# -----------------------------------------------------------------
# Grab data and organize by location, this allows you to invert
# one location at a time.
# -----------------------------------------------------------------
data[:, 9] = data[:, 9]/1000    # Normalize to units of mm
pastLoc = data[0, 9]            # Grab the current location value
loc = 0                         # Location Counter
locInd = 0                      # Current location index
locData = []                    # List of data by location
locErr = []                     # List of Error flags by location
dataTemp = np.zeros((samp*12, 30))
for ind in range(0, len(data)):     # Iterate through All Data
    currLoc = data[ind, 9]          # Current location
    if currLoc != pastLoc:          # If we've switched location
        loc = loc + 1               # increment location counter
        pastLoc = currLoc           # Set past location
        locInd = ind                # Record where we switched locations
        locErr.append(np.zeros((samp*12, 1)))   # Initialize errors to 0
        locData.append(dataTemp)    # Store values in a list of matrices
        dataTemp = np.zeros((samp*12, 30))      # Reinitialize matrix
    dataTemp[ind-locInd, :] = data[ind, :]        # Measurements for location
locData.append(dataTemp)            # Drop in the last set of values
locErr.append(np.zeros((samp*12, 1)))

# -----------------------------------------------------------------
# If resistance data has been taken then this section sets
# error flags based on measured resistance data
# -----------------------------------------------------------------
if data.shape[1] > 20:              # If the measurements include resistance
    for ind in range(len(locData)):
        dataTemp = locData[ind]
        LCC = dataTemp[:, 24:29]   # Line to current Collector
        CL = dataTemp[:, [13, 15, 17, 19, 21, 23]]  # Across Line
        IL = dataTemp[:, [14, 16, 18, 20, 22]]      # Inter-line
        for ind2 in range(len(LCC)):                # Check Line-CC
            if max(LCC[ind2, :]) > 1e6:
                locErr[ind][ind2, :] = 1
        for ind2 in range(len(CL)):                 # Check line begin-end
            for ind3 in range(0, 6):
                if CL[ind2, ind3] > 1e10:
                    locErr[ind][ind2, :] = 1
        for ind2 in range(len(IL)):                 # Check Inter-line
            for ind3 in range(0, 5):
                if IL[ind2, ind3] < 1e10:
                    locErr[ind][ind2, :] = 1
end = time.time()
print('Pre-Inversion Error Checking Complete (',
      round(end-start, 5), 'seconds elapsed )')
# Note: This should all be changed to add the fact that not all
# experiments have necessarily failed if one line shorts or one
# line breaks... later

# -----------------------------------------------------------------
# Data inversion, and precomputation of some harder stuff
# -----------------------------------------------------------------
start = time.time()
precompute()
end = time.time()
print('Precomputations complete (', round(end-start, 5), 'seconds elapsed )')
print('Beginning Inversion...')
start = time.time()
for ind in range(0, len(locData)):
    cond(T, locData[ind], ind)
end = time.time()
print('Inversion Complete (', round(end-start, 5), 'seconds elapsed )')
print('Plotting Results...')
endT = time.time()
print('Complete, Total Time:', round(endT-startT, 5), 'seconds')

# plt.plot(data[:, 9])
