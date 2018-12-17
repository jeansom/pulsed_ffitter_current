import sys, os
import numpy as np
from scipy.stats import linregress
from fitFunctions import *

def correct2Pi(times, phi_arr, phi_err, freq, N):
    fit_freq = np.zeros(len(phi_arr)-1)
    fit_err = np.zeros(len(phi_arr)-1)
    NCum=0
    fit_freq_temp = 0
    phi_err_temp = 0
    for i in range(1, len(phi_arr)):
        fit_freq[i-1], fit_err[i-1], add = find2PiN(times[:i+1], phi_arr[:i+1], phi_err[:i+1], NCum, N, freq)
        phi_arr[i] = phi_arr[i] + 2*np.pi*(NCum+add)
        NCum = NCum + add
    return phi_arr, fit_freq, fit_err

def find2PiN(times, phi_arr, phi_err, NCum, N, freq):
    resmin = np.inf
    for i0 in np.arange(-1, 2):
        phi_arrT = phi_arr.copy()
        phi_arrT[-1] = phi_arrT[-1] + 2*np.pi*(NCum+N+i0)
        phi_fit = linregress( times, phi_arrT )
        resminT = np.sum((phi_arrT - (phi_fit[1] + times*phi_fit[0]))**2)
        if phi_fit[0] > 0 and resminT < resmin and np.abs((phi_fit[0]-freq)/freq) < 0.02: 
            phi_fit, phi_cov = curve_fit( lambda x, a, b: a*x+b, times, phi_arrT, phi_err, [phi_fit[0], phi_fit[1]], [])
            fit_err_temp = phi_cov[0]
            fit_freq_temp = phi_fit[0]
            add = N+i0
            resmin = resminT
    return fit_freq_temp, fit_err_temp, add

def correctPhases(times, paraOut_arr, errtout_arr):
    # ing phases by 2*pi*N
    phi_arr = paraOut_arr[:,2].copy()
    phi_err = errtout_arr[:,2].copy()
    freq = 2*np.pi*np.mean(paraOut_arr[:,1])

    N = np.floor(freq*(times[1]-times[0])/(2*np.pi)) # Number of integer cycles for phase1 in 1 subsection of the data

    phi_arr, fit_freq, fit_err = correct2Pi(times, phi_arr, phi_err, freq, N)
    return np.array(phi_arr), np.array(fit_freq), np.array(fit_err)