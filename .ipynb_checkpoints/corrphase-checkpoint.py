import sys, os
import numpy as np
import scipy.optimize as optimize
from chi2 import *
from fitFunctions import *

def corrphase(times, phi1_arr, phi2_arr, phi1_err, phi2_err, r, absSigma):
    
    cp_err = np.sqrt(phi1_err**2 + (phi2_err*r)**2)
    cp = phi1_arr - r*phi2_arr
    
    fitResult = optimize.minimize( lambda args: least_sq(cp, args[0]*times+args[1])/1e4, [1e-10,0.1], method="SLSQP", tol=1e-10, options={'eps': 1e-12, 'maxiter': 10000, 'ftol':1e-10} )
    cp_fit, cp_fiterr = curve_fit(lambda x, *args: args[0]*x+args[1], times, cp, np.ones(len(cp)), fitResult.x, [] )
    cp_fitval = cp_fit[1] + times*cp_fit[0]
    chisq = chi2( cp_fitval, cp, cp_err, 2 )
    
    return np.array(cp), np.array(cp_err), cp_fitval, np.array(cp_fit), np.array(cp_fiterr), chisq
