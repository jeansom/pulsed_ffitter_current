import sys, os
import numpy as np
import scipy.optimize as optimize
from scipy.stats import linregress
from chi2 import *
from fitFunctions import *
from fitGammaRatio import *

gHe = 20378.9
gNe = 2*np.pi*336.1

def fitPhases( saveDirName, k_ind, absSigma, times=[], phi1_arr=[], phi1_err=[], phi2_arr=[], phi2_err=[] ):
    try: 
        save_arrs = np.load(saveDirName+"/initialfitting_"+str(k_ind)+".npz")
        times = save_arrs['times']
        errtout_arr = save_arrs['errtout_arr']
        phi1_err = errtout_arr[:,5]; phi2_err = errtout_arr[:,2]
        save_arrs = np.load(saveDirName+"/correctphases_"+str(k_ind)+".npz")
        phi1_arr = save_arrs['phi1_arr']
        phi2_arr = save_arrs['phi2_arr']
    except:
        if not(len(phi1_arr)>0 and len(phi1_err)>0 and len(phi2_arr)>0 and len(phi2_err)>0):
            print("RUN INITIALFITTING AND CORRECTPHASES FIRST OR PROVIDE ARRAYS!!!")
            sys.exit()
    phi1_fit = linregress( times, phi1_arr )
    
    phi1_fit, phi1_fit_err = curve_fit( lambda x, a, b: a*x+b, times, phi1_arr, phi1_err, [phi1_fit[0], phi1_fit[1]], [] )
    
    phi2_fit = linregress( times, phi2_arr )
    
    phi2_fit, phi2_fit_err = curve_fit( lambda x, a, b: a*x+b, times, phi2_arr, phi2_err, [phi2_fit[0], phi2_fit[1]], [] )
    
    phi1_fit_results = phi1_fit[1] + times*phi1_fit[0]
    phi2_fit_results = phi2_fit[1] + times*phi2_fit[0]
    
    chi2_1 = chi2( phi1_fit_results, phi1_arr, phi1_err, 2)
    chi2_2 = chi2( phi2_fit_results, phi2_arr, phi2_err, 2)
    
    r = fitGammaRatio(phi1_arr, phi2_arr, gHe/gNe)
    
    np.savez(os.path.join(saveDirName, 'fitPhases_'+str(k_ind)+'.npz'), phi1_fit_results=np.array(phi1_fit_results), phi1_fit=np.array(phi1_fit), phi1_fit_err=np.array(phi1_fit_err),  phi2_fit_results=np.array(phi2_fit_results), phi2_fit=np.array(phi2_fit), phi2_fit_err=np.array(phi2_fit_err), r_chi21_chi22=np.array([r, chi2_1, chi2_2]) )