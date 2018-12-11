import sys, os
import numpy as np
import scipy.optimize as optimize
import scipy.fftpack as fft
import scipy.linalg as linalg
from scipy.stats import chisquare
from scipy.stats import linregress

from BandPass2 import *
from fitFunctions import *
from partialfitting import *
from tqdm import *
def partialfitting(k_ind, k, n_ave, dirName, files, dt, Fs, D1Start, D1EndTimes, D2Start, D2EndTimes, curve_fit, calcJac, delta_t):
    
    bounds_fixfreq = ( ( -3, 3), (-1.5,1.5), (-1,1), (-1,1), (-1, 1), (-1, 1) ) # Bounds for fit if frequency is fixed, for scipy minimize
    bounds = ( ( -3, 3), (-1.5,1.5), (0, 3), (-1,1), (-1,1), (10,20), (-1, 1), (-1, 1) ) # Bounds for fit if frequency is not fixed, for scipy minimize

    bounds_fixfreq_cf = [ [ -3, -1.5, -1, -1, -1, -1, -1 ], [ 3, 1.5, 1, 1, 1, 1 ] ] # Bounds for fit if frequency is fixed, for curve_fit
    bounds_cf = [ [ -3, -1.5, 0, -1, -1, 10, -1, -1 ], [ 3, 1.5, 3, 1, 1, 20, 1, 1 ] ] # Bounds for fit if frequency is not fixed, for curve_fit

    paraOut = np.array([1.1, 1.1, 0.5, 0.5, 1e-10, 1e-10 ]) # Initial parameters for fit
    data = np.loadtxt(dirName+"/"+files[k_ind])
    data = data[:int(np.floor(len(data)/n_ave)*n_ave)]
    data = np.mean( data.reshape(-1, n_ave), axis=1 ) # Average data
    times = np.arange( 1, len(data)+1, 1 )*dt
    DStart = D1Start
    DEnd = D1EndTimes[np.mod(k_ind-1,len(D2EndTimes))]
    ind_DStart = np.argmin( np.abs(times - DStart) )+1 # Index of start time
    ind_DEnd = np.argmin( np.abs(times - DEnd) ) # Index of end time
    datafit = np.array([ times[ind_DStart:ind_DEnd] - times[ind_DEnd-1], data[ind_DStart:ind_DEnd] - np.mean(data[ind_DStart:ind_DEnd]) ])
    # Run fit a few times to get initial parameters
    paraOut = optimize.minimize( lambda para: least_sq(datafit[1], fitSine2Slope([para[0], para[1], 1.5554, para[2], para[3], 15.008, paraOut[4], paraOut[5]], datafit[0])), [ paraOut[0], paraOut[1], paraOut[2], paraOut[3], paraOut[4], paraOut[5] ], method="SLSQP", bounds=bounds_fixfreq, options={'eps': 1e-12} ).x
    paraOut = optimize.minimize( lambda para: least_sq(datafit[1], fitSine2Slope(para, datafit[0])), [ paraOut[0], paraOut[1], 1.5554, paraOut[2], paraOut[3], 15.008, paraOut[4], paraOut[5] ], method="SLSQP", bounds=bounds, options={'eps': 1e-12} ).x
    datafit = BandPassFilter(datafit, dt, paraOut, 200, 1) # Bandpass filter
    paraOut = optimize.minimize( lambda para: least_sq(datafit[1], fitSine2Slope(para, datafit[0])), paraOut, method="SLSQP", bounds=bounds, options={'eps': 1e-12} ).x # Run fit again
    ## Partial Fitting
    NPoints = len(datafit[0])
    nPoints = int(np.round(delta_t/dt))
    n_subs = int(np.floor(NPoints/nPoints))
    dataPar_x = datafit[0][:nPoints*n_subs]
    dataPar_y = datafit[1][:nPoints*n_subs]
    dataPar_x = np.reshape(dataPar_x, [n_subs, nPoints])
    dataPar_y = np.reshape(dataPar_y, [n_subs, nPoints])
    paraOut_arr = []
    errtout_arr = []
    errtout_arr2= []
    errchi2_arr = []
    phi1_arr_all = []
    phi2_arr_all = []
    datafit2 = []
    for i in range(n_subs):
        dataPar_x0 = dataPar_x[i] - min(dataPar_x[i]) # Set 0 to beginning of subsection of data
        # Run fit a few times with minimize to get good initial parameters
        fitResult = optimize.minimize( lambda para: least_sq(dataPar_y[i], fitSine2Slope([para[0], para[1], 1.5554, para[2], para[3], 15.008, para[4], para[5]], dataPar_x0)), [ paraOut[0], paraOut[1], paraOut[3], paraOut[4], paraOut[5], paraOut[6]], method="SLSQP", bounds=bounds_fixfreq, options={'eps': 1e-12, 'maxiter': 10000} )
        fitResult = optimize.minimize( lambda para: least_sq(dataPar_y[i], fitSine2Slope(para, dataPar_x0)), [ fitResult.x[0], fitResult.x[1], 1.5554, fitResult.x[3], fitResult.x[3], 15.008, fitResult.x[4], fitResult.x[5]], method="SLSQP", bounds=bounds, options={'eps': 1e-12, 'maxiter': 10000} )

        if curve_fit: # Run fit using curve fit
            fitResult = optimize.curve_fit( lambda xdata, *para: fitSine2Slope(para, xdata), dataPar_x0, dataPar_y[i], fitResult.x, bounds=bounds_cf, gtol=2.220446049250313e-16, ftol=2.220446049250313e-16, xtol=2.220446049250313e-16)
            #fitResult = optimize.least_squares( lambda xdata, *para: fitSine2Slope(para, xdata), dataPar_x[i], dataPar_y[i], x0=fitResult.x, bounds=bounds, gtol=2.220446049250313e-16, ftol=2.220446049250313e-16, xtol=2.220446049250313e-16)
            paraOut = fitResult[0]
            datafit2.extend(fitSine2Slope(paraOut, dataPar_x0))

            errtout = np.sqrt(np.diag(fitResult[1]))
        else: # Run fit using minimize
            fitResult = optimize.minimize( lambda para: least_sq(dataPar_y[i], fitSine2Slope(para, dataPar_x0)), fitResult.x, method="SLSQP", bounds=bounds, options={'eps': 1e-12} )
            paraOut = fitResult.x
            datafit2.extend(fitSine2Slope(paraOut, dataPar_x0))

        if calcJac or (not curve_fit): # Calculate jacobian by hand if desired
            if (not curve_fit) and (not calcJac): J = np.array([fitResult.jac])
            else: J = np.array([np.sum(np.array(Jacobian( paraOut, dataPar_x0 )).T, axis=0)])
            rechi2 = np.sum( (fitSine2Slope(paraOut, dataPar_x0) - dataPar_y[i])**2/(len(dataPar_x) - len(paraOut)))
            resigma = np.sqrt(rechi2)
            alpha = np.matmul( J.T, J )/resigma**2
            try:
                erroralpha = np.linalg.inv(alpha.T)
                errtout = np.sqrt(np.diag(erroralpha))
            except:
                errtout = np.zeros(len(paraOut))
        paraOut_arr.append([ 
            np.sqrt(paraOut[0]**2 + paraOut[1]**2), paraOut[2], np.arctan2(paraOut[1], paraOut[0]), # Convert linearized parameters to normal
            np.sqrt(paraOut[3]**2 + paraOut[4]**2), paraOut[5], np.arctan2(paraOut[4], paraOut[3]),
            paraOut[6], paraOut[7]
        ])
        errtout_arr.append([
            paraOut_arr[-1][0]**(-1) * np.sqrt( (errtout[0]/paraOut[0])**2 + (errtout[1]/paraOut[1])**2 ), # Convert linearized parameter errors to normal
            errtout[2], 
            np.abs(paraOut[0]*paraOut[1])/paraOut_arr[-1][0]**2 * np.sqrt( (errtout[0]/paraOut[0])**2 + (errtout[1]/paraOut[1])**2 ),
            paraOut_arr[-1][3]**(-1) * np.sqrt( (errtout[3]/paraOut[3])**2 + (errtout[4]/paraOut[4])**2 ), 
            errtout[5], 
            np.abs(paraOut[3]*paraOut[4])/paraOut_arr[-1][3]**2 * np.sqrt( (errtout[3]/paraOut[3])**2 + (errtout[4]/paraOut[4])**2 ),
            errtout[6], errtout[7]
        ])
    return np.array(paraOut_arr), np.array(errtout_arr), np.array(dataPar_x), np.array(dataPar_y), np.array(datafit2)

def correctPhases(paraOut_arr, errtout_arr, dataPar_x, dataPar_y):
    # Correcting phases by 2*pi*N
    n = 2*np.pi
    phi1_arr = paraOut_arr[:,5].copy() # phase 1
    phi2_arr = paraOut_arr[:,2].copy() # phase 2
    freq1 = 2*np.pi*np.mean(paraOut_arr[:,4])
    freq2 = 2*np.pi*np.mean(paraOut_arr[:,1])
    N1 = np.floor(freq1*(dataPar_x[1,0]-dataPar_x[0,0])/n) # Number of integer cycles for phase1 in 1 subsection of the data
    N2 = np.floor(freq2*(dataPar_x[1,0]-dataPar_x[0,0])/n) # Number of integer cycles for phase2 in 1 subsection of the data
    test_arr = []
    fit_freq1 = []
    fit_err1 = []
    fit_freq2 = []
    fit_err2 = []
    vald = -1 # How much to let N1, N2 vary by, minimum
    valu = 1 # How much to let N1, N2 vary by, maximum
    NCum1=0
    NCum2=0
    add1=0
    add2=0
    fit_freq1_temp = 0
    fit_freq2_temp = 0
    phi1_err_temp = 0
    phi2_err_temp = 0

    # Correct the phases so they fall on the same line
    for i in range(len(phi1_arr)):
        if i == 0: continue
        ind = i
        resmin1 = np.inf
        resmin2 = np.inf
        for i0 in np.arange(vald,valu+1):
            phi1_arrT = phi1_arr[:ind+1].copy()
            phi1_arrT[-1] = phi1_arrT[-1] + n*(NCum1+N1+i0)
            phi1_fit = linregress( np.mean(dataPar_x, axis=1)[:ind+1], phi1_arrT )
            resmin = np.sum((phi1_arrT - (phi1_fit[1] + np.mean(dataPar_x, axis=1)[:ind+1]*phi1_fit[0]))**2)
            if phi1_fit[0] > 0 and resmin < resmin1 and np.abs((phi1_fit[0]-freq1)/freq1) < 0.02: 
                phi1_fit, phi1_cov = optimize.curve_fit( lambda x, a, b: a*x+b, np.mean(dataPar_x, axis=1)[:ind+1], phi1_arrT, p0=[phi1_fit[0], phi1_fit[1]], sigma=errtout_arr[:ind+1,5] )
                phi1_err_temp = np.sqrt(np.diag(phi1_cov))[0]
                fit_freq1_temp = phi1_fit[0]
                add1 = N1+i0
                resmin1 = resmin

        for i0 in np.arange(vald,valu+1):
            phi2_arrT = phi2_arr[:ind+1].copy()
            phi2_arrT[-1] = phi2_arrT[-1] + n*(NCum2+N2+i0)
            phi2_fit = linregress( np.mean(dataPar_x, axis=1)[:ind+1], phi2_arrT )
            resmin = np.sum((phi2_arrT - (phi2_fit[1] + np.mean(dataPar_x, axis=1)[:ind+1]*phi2_fit[0]))**2)
            if phi2_fit[0] > 0 and resmin < resmin2 and np.abs((phi2_fit[0]-freq2)/freq2) < 0.02:
                phi2_fit, phi2_cov = optimize.curve_fit( lambda x, a, b: a*x+b, np.mean(dataPar_x, axis=1)[:ind+1], phi2_arrT, p0=[phi2_fit[0], phi2_fit[1]], sigma=errtout_arr[:ind+1,2] )
                phi2_err_temp = np.sqrt(np.diag(phi2_cov))[0]
                fit_freq2_temp = phi2_fit[0]
                add2 = N2+i0
                resmin2 = resmin

        fit_freq1.append(fit_freq1_temp)
        fit_freq2.append(fit_freq2_temp)
        fit_err1.append(phi1_err_temp)
        fit_err2.append(phi2_err_temp)
        phi1_arr[ind] = phi1_arr[ind] + n*(NCum1+add1)
        NCum1 = NCum1 + add1
        phi2_arr[ind] = phi2_arr[ind] + n*(NCum2+add2)
        NCum2 = NCum2 + add2
    return np.array(phi1_arr), np.array(phi2_arr), np.array(fit_freq1), np.array(fit_freq2), np.array(fit_err1), np.array(fit_err2), 

