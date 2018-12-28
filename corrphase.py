import sys, os
import numpy as np
import scipy.optimize as optimize
from chi2 import *
from fitFunctions import *

def corrphase(saveDirName, k_ind, absSigma, phi1_arr=[], phi2_arr=[], phi1_err=[], phi2_err=[], r=[]):
    try: 
        save_arrs = np.load(saveDirName+"/initialfitting_"+str(k_ind)+".npz")
        times = save_arrs['times']
        errtout_arr = save_arrs['errtout_arr']
        phi1_err = errtout_arr[:,5]; phi2_err = errtout_arr[:,2]
        save_arrs = np.load(saveDirName+"/correctphases_"+str(k_ind)+".npz")
        phi1_arr = save_arrs['phi1_arr']
        phi2_arr = save_arrs['phi2_arr']
        r_arrs = [f for f in os.listdir(saveDirName) if f.endswith('.npz')]
        r = []
        for fi in r_arrs:
            if 'fitPhases' in fi:
                r.append(np.load(saveDirName+"/"+fi)['r_chi21_chi22'][0])
        r = np.array(r)
        r = np.mean(r[np.abs(r-r[0])<0.00005])
        print(r)
    except:
        if not(len(phi1_arr)>0 and len(phi1_err)>0 and len(phi2_arr)>0 and len(phi2_err)>0):
            print("RUN INITIALFITTING, CORRECTPHASES, AND FITPHASES FIRST OR PROVIDE ARRAYS!!!")
            sys.exit()
    cp_err = np.sqrt(phi1_err**2 + (phi2_err*r)**2)
    cp = phi1_arr - r*phi2_arr
    
    fitResult = optimize.minimize( lambda args: least_sq(cp, args[0]*times+args[1])/1e4, [1e-10,0.1], method="SLSQP", tol=1e-10, options={'eps': 1e-12, 'maxiter': 10000, 'ftol':1e-10} )
    cp_fit, cp_fiterr = curve_fit(lambda x, *args: args[0]*x+args[1], times, cp, cp_err, fitResult.x, [] )
    cp_fitval = cp_fit[1] + times*cp_fit[0]
    chisq = chi2( cp_fitval, cp, cp_err, 2 )
    
    np.savez(os.path.join(saveDirName, 'corrphase_'+str(k_ind)+'.npz'), cp_fit=np.array(cp_fit), cp_fiterr=np.array(cp_fiterr), r=[r])