import sys, os
import numpy as np
import scipy.optimize as optimize
from scipy.stats import linregress
from chi2 import *

def fitPhases( times, phi1_arr, phi1_err, phi2_arr, phi2_err, absSigma ):
    phi1_fit = linregress( times, phi1_arr )
    
    phi1_fit, phi1_cov = optimize.curve_fit( lambda x, a, b: a*x+b, times, phi1_arr, p0=[phi1_fit[0], phi1_fit[1]], sigma=phi1_err, bounds=([.5*phi1_fit[0], .5*phi1_fit[1]], [1.5*phi1_fit[0], 1.5*phi1_fit[1]]), absolute_sigma=absSigma )
    phi1_fit_err = np.sqrt(np.diag(phi1_cov))
    
    phi2_fit = linregress( times, phi2_arr )
    
    phi2_fit, phi2_cov = optimize.curve_fit( lambda x, a, b: a*x+b, times, phi2_arr, p0=[phi2_fit[0], phi2_fit[1]], sigma=phi2_err, absolute_sigma=absSigma )
    
    phi2_fit_err = np.sqrt(np.diag(phi2_cov))
    
    phi1_fit_results = phi1_fit[1] + times*phi1_fit[0]
    phi2_fit_results = phi2_fit[1] + times*phi2_fit[0]
    
    chi2_1 = chi2( phi1_fit_results, phi1_arr, phi1_err, 2)
    chi2_2 = chi2( phi2_fit_results, phi2_arr, phi2_err, 2)
    
    return np.array(phi1_fit_results), np.array(phi1_fit), np.array(phi1_fit_err), chi2_1, np.array(phi2_fit_results), np.array(phi2_fit), np.array(phi2_fit_err), chi2_2