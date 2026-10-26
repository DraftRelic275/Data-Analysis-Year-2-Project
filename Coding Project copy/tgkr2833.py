#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 20:10:49 2026

@author: davidrimmer
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import root_scalar

###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#===== Task 1

def read_data(filename):
    """Reads the given data file from a dat file and return the time and channel values.

    Args:
        filename (str):
            The name and path of the .dat file to read.

    Returns:
        left_data (1D array of float):
            Time values recorded at the left channel.
        right_data (1D array of float):
            Time values recorded at the right channel.
    """
    
    if "practice_data" in filename:
        data = np.genfromtxt(filename, skip_header=15)
    elif "assessment_data" in filename:
        data = np.genfromtxt(filename, skip_header=13)
    else:
        raise Exception("file is not of the correct type")
    
    left_data = [line[0] for line in data if line[1] == 1]
    right_data = [line[0] for line in data if line[1] == 2]
    
    
    return left_data, right_data

def plot_histograms(filename):
    """Plots histograms for the left and right channels.
    
    Args:
        filename:
            The name and path of a .dat file to read.
        
    Returns:
        Histogram plots of the left and right detectors.
    """
    
    data = read_data(filename)
    left_data = data[0]
    right_data = data[1]
    
    left_bins = round(np.sqrt(len(left_data)))
    right_bins = round(np.sqrt(len(right_data)))
    
    plt.figure()
    plt.hist(left_data, bins=left_bins, color='000000')
    plt.title('tgkr2833 - Channel 1, 10keV')
    plt.xlabel('Time (μs)')
    plt.ylabel('Counts')
    plt.show()
    
    plt.figure()
    plt.hist(right_data, bins=right_bins, color='000000')
    plt.title('tgkr2833 - Channel 2, 10keV')
    plt.xlabel('Time (μs)')
    plt.ylabel('Counts')
    plt.show()
    
###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#===== Task 2

def asymmetry_eqtn(Time, Beta, Magnetic_Field, Time_Damping,Gamma=851.616E6):
    """A function to calculate the asymmetry using equation 2.6.
    
    Args:
        Beta (float):
           The size of the angle covered by each detector. 
        Gamma (float):
            The gyeomagnetic ratio.
        Magnetic_Field (float):
            The value of the magnetic field strength.
        Time (1D array of float):
            An array of times.
        Time_Damping (float):
            A value for the time damping of muons.
    
    Returns:
        (1D array of float):
            An array of values for the asymmetry.
    """
    
    Omega = Gamma*Magnetic_Field
    T = np.cos(Omega*Time)
    A = np.sin(Beta)/(3*Beta)
    B = np.exp(-(Time/Time_Damping))
    Asymmetry = A*T*B
    
    return Asymmetry
    
def Magnetic_Field_Estimation(Asymmetry, times_asymmetry):
    """Estimates a value for the Magnetic Field strength by finding the time period of the asymmetry data.
    
    Args:
        Asymmetry (1D array of float):
            List of values for the asymmetry.
        times_asymmetry (1D array of float):
            List of time values in microseconds.
    Returns:
        B_est (float):
            Estimate for the magnetic field.
    """
    
    Times_cross = []
    Indexes = []
    
    # Loop through 1 less than the length to not go out of bounds.
    
    for i in range(0, 100):
        y1, y2 = Asymmetry[i], Asymmetry[i+1]
        
        # Check if the y value changes sign between indexes.
        
        if y1 > 0 and y2 < 0:
            t1, t2 = times_asymmetry[i], times_asymmetry[i+1]
                 
            # use linear interpolation to estimate the actual time value when crossing 0.
                
            t_zero = t1 - y1*(t2-t1)/(y2-y1)
            
            # Append the interpolated value to a list
            Indexes.append(i)
            Times_cross.append(t_zero)
                
    # Calculates the time period by finding the difference between the i and i+1 elements in the list passed through.
    T_values = np.diff(Times_cross)
    
    # Calculates the mean value from the list of time periods.
    Period = np.mean(T_values)
    
    # Calculates the Magnetic Field Strength
    Gamma = 851.616E6
    
    B_Est = (2*np.pi)/(Period*Gamma)
    
    return B_Est

def fitting(estimates, Asymmetry, Upper_Bounds, Lower_Bounds, Asymmetry_Error, Time):
    """fits equation 2.6 to the data and returns parameter values matching the asymmetry data.
    
    Args:
        estimates (1D array of float):
            array of estimates for the equations parameters.
        Asymmetry (1D array of float):
            List of values for the asymmetry.
        Upper_Bounds (1D array of float):
            Upper bounds of the range the parameters exist between.
        Lower_Bounds (1D of float):
            Lower bounds of the range the parameters exist between.
        Asymmetry_Error (1D array of float):
            Uncertainty values for each Asymmetry value.
        Time (1D array of float):
            List of time values in microseconds.
    Returns:
        (2D array of float):
            values of the parameters and their correesponding uncertainty.
    """
    
    Bounds = (Lower_Bounds, Upper_Bounds)
    
    # Variables to recieve the outputs from curve_fit, which give the estimated paramater values.
    p_optimal, covariance = curve_fit(asymmetry_eqtn, Time, Asymmetry, p0=estimates, sigma = Asymmetry_Error, absolute_sigma=True, bounds=Bounds)
    
    # Calculates the uncertainties in the estimated values
    p_error=np.sqrt(np.diag(covariance))
    
    # Setting variables to the estimated values of parameters
    Beta_est = p_optimal[0]
    Beta_est_error = p_error[0]
    
    Mag_field_est = p_optimal[1]
    Mag_field_est_error = p_error[1]
    
    Time_damp_est = p_optimal[2]
    Time_damp_est_error = p_error[2]
        
    return [[Beta_est, Beta_est_error], [Mag_field_est, Mag_field_est_error], [Time_damp_est, Time_damp_est_error]]

def amplitude_eqtn(Beta, Amplitude):
    """Defines the time independant equation which influences the amplitude in equation 2.6.
    
    Args:
        Beta (float):
            The detector angle.
        Asymmetry (1D array of float):
            Array of values for the asymmetry.
        
    Returns:
        C (float):
            Result of subbing arguments into the equation.
    """
        
    A = np.sin(Beta)/Beta
    B = 3*Amplitude
    C = A-B
    return C

def Beta_estimate(Amplitude, est_range):
    """Finds the  root of the function 'amplitude_equation' to estimate a value of beta.
    
    Args:
        Amplitude (1D array of float):
            Array of values for the asymmetry.
    Returns:
        Beta (float):
            value of the detector angle beta.
    """
    
    Beta_root = root_scalar(amplitude_eqtn, args=Amplitude, method="brentq", bracket=est_range)
    Beta = Beta_root.root
    return Beta
        
def asymmetry(filename):
    """Calculates the asymmetry of a datafile.
    
    Args:
        filename (str):
            The name and path of a .dat file to read.
    
    Returns:
        Scatter Graph:
            A scatter graph of the mixed signals of both detectors and a graph of the asymmetry.
        Beta (float):
            Value for the detector angle in radians.
        Beta_error (float):
            Uncertainty associated with beta.
        Mag_field (float):
            Value for the Magnetic Field in Tesla.
        Mag_field_error (float):
            Uncertainty associated with the Magnetic Field.
        Tau_damping (float):
            Time damping constant in seconds.
        Tau_damping_error (float):
            Uncertainty associated with the Tau damping constant.
    """
    
    # Reads data file ans seperates data into the left and right channels.
    data = read_data(filename)
    left_data = data[0]
    right_data = data[1]
    
    left_bins = round(np.sqrt(len(left_data)))
    right_bins = round(np.sqrt(len(right_data)))
    bins_t = left_bins if left_bins < right_bins else right_bins
    
    
    # Retreives the counts for the left detector from the .hist function.
    counts_L, bins_L, bars_L = plt.hist(left_data, bins=bins_t, color='000000')
    plt.close()
    
    
    # Retreives the counts for the right detector from the .hist function.
    counts_R, bins_R, bars_R = plt.hist(right_data, bins=bins_L, color='000000')
    plt.close()
    
    
    # adjusts times so they start from the middle of the bins and not the bin edges.
    time = (0.5 * (bins_L[:-1] + bins_L[1:])) * 1e-6
    
    
    # Calculates the Asymmetry.
    Asymmetry = ((counts_L-counts_R)/(counts_L+counts_R))
    
    
    # Calculates the total Counts.
    Counts_total = counts_L + counts_R
    
    
    # Calculates the Error bars for Asymmetry_0 with equation 8.1.
    Asymmetry_Error = 2*np.sqrt((counts_L*counts_R)/((Counts_total)**3))
    
    
    # Estimation for the value of the magnetic field.
    B_Est = Magnetic_Field_Estimation(Asymmetry, time)
     
    
    # Estimate the value of Beta.
    tau_damping_estimate = 6e-6
    
    # remove time damping from asymetry data
    Corrected_Asymmetry = Asymmetry * np.exp(time/tau_damping_estimate)
    
    # finds the amplitude by taking the max value of the first peak
    first_peak = Corrected_Asymmetry[0:20]
    amplitude_0 = np.max(first_peak)
    
    # creates a list of values for beta
    beta_vals = np.linspace(0.5, 1.5, 100)
    
    # creates a list of values for the function below using the beta values above
    func_Vals = (np.sin(beta_vals)) - 3*beta_vals*amplitude_0
    lower = 0
    upper = 0
    
    # goes through the values of func_Vals and finds where the sign changes 0, and saves these as bounds to estimate where beta will be between
    for i in range(0,100):
        if func_Vals[i] > 0 and func_Vals[i+1] < 0:
            lower = beta_vals[i]
            upper = beta_vals[i+1]
    est_range = [lower, upper]
    
    # estimates a value for beta
    Beta_Est = Beta_estimate(amplitude_0, est_range)
    
    
    # Estimates to pass into the curve_fit finction as a starting point and bounds to limit fitting range.
    estimates = [Beta_Est, B_Est, tau_damping_estimate]
    Lower_Bounds = [0.5, 0.01, 2e-6]
    Upper_Bounds = [1.5, 0.03, 10e-6]
    
    
    # Calls the fitting functiion to run the curve_fit function with the data and estimates.
    results = fitting(estimates, Asymmetry, Upper_Bounds, Lower_Bounds, Asymmetry_Error, time)
    
    Beta = results[0][0]
    Beta_error = results[0][1]
    
    
    Mag_field = results[1][0]
    Mag_field_error = results[1][1]
    
    
    Tau_damping = results[2][0]
    Tau_damping_error = results[2][1]
    
    # plots the graph as required by task 3 of the project.
    plt.figure()
    plt.title('Measured and Fitted Asymmetry, 10keV - tgkr2833')
    plt.xlabel('time (µs)')
    plt.ylabel('Asymmetry ')
    plt.errorbar(time, Asymmetry, color='red', fmt='.', yerr=Asymmetry_Error, ecolor='blue', capsize=2, elinewidth=0.5, linewidth=1.5, label = 'Measured Asymmetry Data')
    plt.plot(time, asymmetry_eqtn(time, Beta, Mag_field, Tau_damping), color='black', linewidth=3, label = 'Fitted Data')
    plt.legend(loc='lower left')
    plt.show()
    
    return Beta, Beta_error, Mag_field, Mag_field_error, Tau_damping, Tau_damping_error
        
def ProcessData(filename):
    """Plots histograms for the left and right channels, plots the asymmetry data and asymmetry data on the same graph, returns estimated value for Beta, Magnetic Field and Tau.
    
    Args:
        filename (str):
            The name and path of a .dat file to read.
    
    Returns:
        Histogram plot:
            Plot of the left and right channels.
        Asymmetry plot:
            Plot of the asymmetry data and asymmetry function (equation 2.6).
        results (dict):
            Contains the values for the magnetic field, Beta and Tau damping.
    """
            
    plot_histograms(filename)
    Beta, Beta_error, Magnetic_Field, Magnetic_Field_error, Tau_Damping, Tau_Damping_error = asymmetry(filename)
    
    results={"10keV_B":Magnetic_Field, #this would be the magnetic field for 10keV data (T)
             "10keV_B_error":Magnetic_Field_error, # the error in the magnetic field (T)
             "beta": Beta, #Detector angle in radians
             "beta_error": Beta_error, #uncertainity in detector angle (rad)
             "10keV_tau_damp": Tau_Damping, #Damping time for 10keV (s)
             "10keV_tau_damp_error": Tau_Damping_error, #and error (s)
             "B(Energy)_coeffs":(None,None,None), #tuple of a,b,c for quadratic,linear and constant terms
                                                  #for fitting B dependence on energy
                                                  #(T/keV^2,T/keV,T)
             "B(Energy)_coeffs_errors":(None,None,None), # Errors in above in same order with the same units
             }
    
    return results



###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
if __name__ == "__main__":
    file = '/Users/davidrimmer/Desktop/Uni Work/Programming/2nd year/Coding Project/assessment_data_tgkr2833.dat'
    # I have changed this file
# =============================================================================
#     file = '/Users/davidrimmer/Desktop/Uni Work/Programming/2nd year/Coding Project/practice_data_tgkr2833(2).dat'
# =============================================================================
    
# =============================================================================
#     file = '/Users/davidrimmer/Desktop/Uni Work/Programming/2nd year/Coding Project/practice_data_tgkr2833(3).dat'
# =============================================================================
    
    print(ProcessData(file))
    
    
    
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
