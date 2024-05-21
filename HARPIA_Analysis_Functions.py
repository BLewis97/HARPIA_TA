import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

def load_TA(fname):
    """Loads the data from TA Carpetview-exported file.

    Args:
        fname (str): Filename or filepath to the TA data file

    Returns:
        _type_: 3 arrays of wavelengths, timepoints, and intensities
    """
    data = np.genfromtxt(fname,skip_header=2)
    wavelengths = data[0,1:]
    timepoints = data[1:,0]
    intensities = data[1:,1:]
    return wavelengths, timepoints, intensities


def plot_spectrum(fname, times = [1], mOD = False, wls = [500,800]):
    """Create a spectrum plot from TA data.

    Args:
        fname (str): Path or name of Carpetview-adjusted TA data file.
        times (list, optional): Spectrum as specified time. Defaults to [1].
        mOD (bool, optional): Carpetview is initially in mOD - change if you have converted to DT/T. Defaults to False.
        wls (list, optional): Wavelgnth range. Defaults to [500,800].
    """
    wavelengths, timepoints, intensities = load_TA(fname)
    #find wavelength index in wavelengths closest to wls
    wls = [np.argmin(np.abs(wavelengths - wl)) for wl in wls]
    #find time index in timepoints closest to time in times
    times = [np.argmin(np.abs(timepoints - time)) for time in times]
    for i in times:
        plt.plot(wavelengths[wls[0]:wls[1]], intensities[i][wls[0]:wls[1]], label = int(timepoints[i]))
    plt.legend(title = 'Time (ns)')
    plt.xlabel('Wavelength (nm)')
    if mOD:
        plt.ylabel('mOD')
    else:
        plt.ylabel('${{\Delta}}$ T/T')
    plt.axhline(0, color = 'black', linewidth = 0.5)

def plot_bleach(fname, wls = [760,800], normalise = False, mOD = False,label = None,from_max=True, constant = 0):
    """
    
    Creates an average of the selected wavelength range and plots it as a function of time.

    Args:
        fname (str): Path or name of Carpetview-adjusted TA data file.
        wls (list, optional): Wavelengths desired to be averaged. Defaults to [760,800].
        normalise (bool, optional): Normalise the data. Defaults to False.
        mOD (bool, optional): Converts mOD to DT/T. Defaults to False.
        label (_type_, optional): Label of trace. Defaults to None.
        from_max (bool, optional): Should the data be plotted with pre-maximum (excitation) values. Defaults to True.
        constant (float, optional): Constant to adjust the data to make it all the same sign. Defaults to None.
    
    Returns:
        array: Signal as a function of time.
        
    """
    wavelengths, timepoints, intensities = load_TA(fname)
    wls = [np.argmin(np.abs(wavelengths - wl)) for wl in wls]
    signal = intensities[:,wls[0]:wls[1]].mean(axis=1)
    signal_max = signal.argmin()
    if not mOD:
        signal = -2.28*signal +constant
    else:
        signal = signal-constant
    if from_max:
        timepoints = timepoints[signal_max:] - timepoints[signal_max]
        signal = signal[signal_max:]
    if normalise:
        signal = signal/signal.max()
    plt.plot(timepoints, signal,label = label)
    plt.xlabel('Time (ns)')
    if mOD:
        plt.ylabel('$\Delta$mOD')
    else:
        plt.ylabel('${{\Delta}}$ T/T')
    plt.yscale('log')
    plt.legend()
    return np.array([timepoints,signal])
    
def extract_pump_values(file_path):
    """
    
    Extract pump values from a HARPIA output file - BEFORE CARPETVIEW
    
    Returns an array of photodiode measured values while the shutter is open (average per measurement)
    
    """
    pump_values = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r'Pump=(\d+\.\d+e[+-]?\d+)', line)
            if match:
                pump_value = float(match.group(1))
                pump_values.append(pump_value)
    pump_values = np.array(pump_values)
    
    for i, val in enumerate(pump_values):
        plt.plot(i, val, 'r.',alpha=0.4)
        plt.ylabel('Picolo Pump Power on Photodiode per spectrum')
        plt.xlabel('Experimental Points Measured')
    plt.show()

    percentage_deviation = [((pump_values.mean()-value )/ pump_values.mean()) * 100 for value in pump_values]
    percentile_95 = np.percentile(percentage_deviation, 95)
    percentile_5 = np.percentile(percentage_deviation, 5)

    plt.hist(percentage_deviation, bins=100, alpha=0.5, color='b')
    plt.axvline(percentile_95, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(percentile_5, color='r', linestyle='dashed', linewidth=1)
    plt.xlabel('Percentage Deviation from Mean')
    plt.show()