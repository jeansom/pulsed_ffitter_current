import sys, os
import numpy as np
import scipy.optimize as optimize
from chi2 import *

def corrphase(times, phi1_arr, phi2_arr, phi1_err, phi2_err, r, absSigma):
    
    corrphase_err = np.sqrt(phi1_err**2 + (phi2_err*r)**2)
    corrphase = phi1_arr - r*phi2_arr
    
    corrphase_fit, corrphase_cov = optimize.curve_fit( lambda x, a, b: a*x+b, times, corrphase, p0=[1e-10, 0.1], sigma=corrphase_err, absolute_sigma=absSigma)
    corrphase_fit_err = np.sqrt(np.diag(corrphase_cov))
    corrphase_fit_results = corrphase_fit[1] + times*corrphase_fit[0]
    chisq = chi2( corrphase_fit_results, corrphase, corrphase_err, 2 )
    
    return np.array(corrphase), np.array(corrphase_err), corrphase_fit_results, np.array(corrphase_fit), np.array(corrphase_fit_err), chisq
