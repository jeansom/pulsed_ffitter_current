import sys, os
import numpy as np
import scipy.optimize as optimize
from scipy.stats import linregress
from chi2 import *
from fitFunctions import *

def fitPhases( times, phi_arr, phi_err ):
    phi_fit = linregress( times, phi_arr )
    
    phi_fit, phi_fit_err = curve_fit( lambda x, a, b: a*x+b, times, phi_arr, phi_err, [phi_fit[0], phi_fit[1]], [] )
    
    phi_fit_results = phi_fit[1] + times*phi_fit[0]
    chi2_1 = chi2( phi_fit_results, phi_arr, phi_err, 2)
    
    return np.array(phi_fit_results), np.array(phi_fit), np.array(phi_fit_err), chi2_1