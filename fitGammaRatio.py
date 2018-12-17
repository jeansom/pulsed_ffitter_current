import sys, os
import numpy as np
import scipy.optimize as optimize

def fitGammaRatio( phi1_arr, phi2_arr, p0 ):
    r_fit, r_cov = optimize.leastsq( lambda p: phi1_arr-p[0]*phi2_arr-p[1], [p0, 0.01] )
    return r_fit[0]