import sys, os
import numpy as np
import scipy.optimize as optimize
import scipy.signal as signal

from BandPass2 import *
from fitFunctions import *
from partialfitting import *

bounds_fixfreq = ( ( -3, 3), (-1.5,1.5), (-1,1), (-1,1), (-1, 1), (-1, 1) ) # Bounds for fit if frequency is fixed, for scipy minimize
bounds = ( ( -3, 3), (-1.5,1.5), (0, 3), (-1,1), (-1,1), (10,20), (-1, 1), (-1, 1) ) # Bounds for fit if frequency is not fixed, for scipy minimize

bounds_fixfreq_cf = [ [ -3, -1.5, -1, -1, -1, -1, -1 ], [ 3, 1.5, 1, 1, 1, 1 ] ] # Bounds for fit if frequency is fixed, for curve_fit
bounds_cf = [ [ -3, -1.5, 0, -1, -1, 10, -1, -1 ], [ 3, 1.5, 3, 1, 1, 20, 1, 1 ] ] # Bounds for fit if frequency is not fixed, for curve_fit

def averageData(data, n_ave):
    data = data[:int(np.floor(len(data)/n_ave)*n_ave)]
    data = np.mean( data.reshape(-1, n_ave), axis=1 ) # Average data
    return data

def initialfitting(saveDirName, k_ind, k, n_ave, dirName, files, dt, Fs, D1Start, D1EndTimes, D2Start, D2EndTimes, delta_t, disp=False, width=3):
    
    paraOut = np.array([1.1, 1.1, 0.5, 0.5, 1e-10, 1e-10 ]) # Initial parameters for fit
    
    data = np.loadtxt(dirName+"/"+files[k_ind])
    if hasattr(data[0], "__len__"): data=data[:,1]
    data = averageData( data, n_ave )
    times = np.arange( 1, len(data)+1, 1 )*dt
    
    DStart = D1Start
    DEnd = D1EndTimes[np.mod(k_ind-1,len(D2EndTimes))]
    ind_DStart = np.argmin( np.abs(times - DStart) )+1 # Index of start time
    ind_DEnd = np.argmin( np.abs(times - DEnd) ) # Index of end time
    
    datafit = np.array([ times[ind_DStart:ind_DEnd] - times[ind_DEnd-1], data[ind_DStart:ind_DEnd] ])
    # Run fit a few times to get initial parameters
    eps = 1e-12
    datafit = BandPassFilter(datafit, dt, paraOut, 500, 500, width) # Bandpass filter
    datafit = np.array(datafit)
    paraOut = optimize.minimize( lambda para: least_sq(datafit[1], fitSine2Slope([para[0], para[1], 1.5554, para[2], para[3], 15.008, para[4], para[5]], datafit[0]))/1e6, [ paraOut[0], paraOut[1], paraOut[2], paraOut[3], paraOut[4], paraOut[5] ], method="SLSQP", bounds=bounds_fixfreq, tol=1e-15, options={'eps': eps, 'disp': disp, 'ftol':1e-15, 'maxiter':5000} ).x
    paraOut = optimize.minimize( lambda para: least_sq(datafit[1], fitSine2Slope(para, datafit[0]))/1e5, [ paraOut[0], paraOut[1], 1.5554, paraOut[2], paraOut[3], 15.008, paraOut[4], paraOut[5] ], method="SLSQP", bounds=bounds, tol=1e-30, options={'eps': 1e-12, 'disp': disp, 'ftol':1e-15, 'maxiter':5000} ).x
    paraOut = optimize.minimize( lambda para: least_sq(datafit[1], fitSine2Slope(para, datafit[0])), paraOut, method="SLSQP", bounds=bounds, options={'eps': 1e-12} ).x # Run fit again

    ## Partial Fitting
    NPoints = len(datafit[0])
    nPoints = int(np.round(delta_t/dt))
    n_subs = int(np.floor(NPoints/nPoints))
    dataPar_x = datafit[0][:nPoints*n_subs]
    dataPar_y = datafit[1][:nPoints*n_subs]
    dataPar_x = np.reshape(dataPar_x, [n_subs, nPoints])
    dataPar_y = np.reshape(dataPar_y, [n_subs, nPoints])
    sigma = getSigma(dataPar_y.flatten(), dt)
    print("Sigma:", sigma)
    paraOut_arr = np.empty((n_subs, 8))
    errtout_arr = np.empty((n_subs, 8))
    datafit2 = []
    for i in range(n_subs):
        if i==0: 
            paraOut_arr[i], errtout_arr[i], datafit2_i = fitSubsec( dataPar_x[i], dataPar_y[i], paraOut, disp )
        else: 
            paraOut_arr[i], errtout_arr[i], datafit2_i = fitSubsec( dataPar_x[i], dataPar_y[i], paraOut, disp )
        if errtout_arr[i][0] > 1e3 or np.sum((datafit2_i-dataPar_y[i])**2) > 1e3:
            print(dataPar_x[i][0])
            paraOut_arr[i], errtout_arr[i], datafit2_i = fitSubsec( dataPar_x[i], dataPar_y[i], paraOut, disp )
        datafit2.extend(datafit2_i)
        N = len(datafit2_i)
        A1 = paraOut_arr[i,0]
        A2 = paraOut_arr[i,3]
        errtout_arr[i] = [
            np.sqrt(sigma**2 / (2*N)),
            np.sqrt(6*sigma**2 / ((np.pi*dt*A1)**2 * N**3)),
            np.sqrt(8*sigma**2 / (A1**2 * N)),
            np.sqrt(sigma**2 / (2*N)),
            np.sqrt(6*sigma**2 / ((np.pi*dt*A2)**2 * N**3)),
            np.sqrt(8*sigma**2 / (A2**2 * N)),
            sigma/np.sqrt(N), np.sqrt(2*sigma**2/(dt*N**2))
        ]
    np.savez(os.path.join(saveDirName, 'initialfitting_'+str(k_ind)+'.npz'), paraOut_arr=np.array(paraOut_arr), errtout_arr=np.array(errtout_arr), dataPara_x=np.array(dataPar_x), dataPara_y=np.array(dataPar_y), datafit2=np.array(datafit2), times=np.mean(dataPar_x, axis=1))

def fitSubsec( dataPar_x, dataPar_y, paraOut, disp=False ):
    eps = 1e-12
    dataPar_x0 = dataPar_x - min(dataPar_x) # Set 0 to beginning of subsection of data
    # Run fit a few times with minimize to get good initial parameters
    fitResult = optimize.minimize( lambda para: least_sq(dataPar_y, fitSine2Slope([para[0], para[1], paraOut[2], para[2], para[3], paraOut[5], para[4], para[5]], dataPar_x0))/1e4, [ paraOut[0], paraOut[1], paraOut[3], paraOut[4], paraOut[6], paraOut[7]], method="SLSQP", bounds=bounds_fixfreq, tol=1e-10, options={'eps': eps, 'maxiter': 10000, 'disp':disp, 'ftol':1e-15} )
    fitResult = optimize.minimize( lambda para: least_sq(dataPar_y, fitSine2Slope(para, dataPar_x0))/1e4, [ fitResult.x[0], fitResult.x[1], paraOut[2], fitResult.x[2], fitResult.x[3], paraOut[5], fitResult.x[4], fitResult.x[5]], method="SLSQP", tol=1e-15, bounds=bounds, options={'eps': 1e-12, 'maxiter': 10000, 'ftol':1e-15} )

    # Run fit using curve fit
    paraOut, errtout = curve_fit( lambda xdata, *para: fitSine2Slope([para[0], para[1], fitResult.x[2], para[2], para[3], fitResult.x[5], para[4], para[5]], xdata), dataPar_x0, dataPar_y, np.ones(len(dataPar_y)), [ fitResult.x[0], fitResult.x[1], fitResult.x[3], fitResult.x[4], fitResult.x[6], fitResult.x[7]], [] )
    #paraOut, errtout = curve_fit( lambda xdata, *para: fitSine2Slope(para, xdata), dataPar_x0, dataPar_y, np.ones(len(dataPar_y)), [ paraOut[0], paraOut[1], fitResult.x[2], paraOut[2], paraOut[3], fitResult.x[5], paraOut[4], paraOut[5]], [] )
    paraOut = [ paraOut[0], paraOut[1], fitResult.x[2], paraOut[2], paraOut[3], fitResult.x[5], paraOut[4], paraOut[5] ]
    #paraOut = [ paraOut[0], paraOut[1], paraOut[2], paraOut[3], paraOut[4], paraOut[5], paraOut[6], paraOut[7] ]
    errtout = [ errtout[0], errtout[1], 0, errtout[2], errtout[3], 0, errtout[4], errtout[5] ]
    paraNorm, errNorm = linearizedSine2Norm( paraOut, errtout )
    return paraNorm, errNorm, fitSine2Slope(paraOut, dataPar_x0)

def linearizedSine2Norm( paraOut, errtout ):
    paraNorm = [np.sqrt(paraOut[0]**2 + paraOut[1]**2), paraOut[2], np.arctan2(paraOut[1], paraOut[0]), # Convert linearized parameters to normal
            np.sqrt(paraOut[3]**2 + paraOut[4]**2), paraOut[5], np.arctan2(paraOut[4], paraOut[3]),
            paraOut[6], paraOut[7]]
    errNorm = [paraNorm[0]**(-1) * np.sqrt( (errtout[0]*paraOut[0])**2 + (errtout[1]*paraOut[1])**2 ), # Convert linearized parameter errors to normal
            errtout[2], 
            np.abs(paraOut[0]*paraOut[1])/paraNorm[0]**2 * np.sqrt( (errtout[0]/paraOut[0])**2 + (errtout[1]/paraOut[1])**2 ),
            paraNorm[3]**(-1) * np.sqrt( (errtout[3]*paraOut[3])**2 + (errtout[4]*paraOut[4])**2 ), 
            errtout[5], 
            np.abs(paraOut[3]*paraOut[4])/paraNorm[3]**2 * np.sqrt( (errtout[3]/paraOut[3])**2 + (errtout[4]/paraOut[4])**2 ),
            errtout[6], errtout[7]]
    return paraNorm, errNorm

def getSigma( data, dt ):
    f, P_den = signal.welch(data, 1/dt, nperseg=2**15)
    sigma = np.sqrt(np.mean(P_den[np.argmin(np.abs(f-14.3)):np.argmin(np.abs(f-14.5))]*(1/(2*dt))))
    return sigma