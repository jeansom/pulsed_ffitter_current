import sys, os
import numpy as np
import scipy.optimize as optimize

from BandPass2 import *
from fitFunctions import *
from partialfitting import *

bounds_fixfreq = ( ( -3, 3), (-1.5,1.5), (-1, 1), (-1, 1) ) # Bounds for fit if frequency is fixed, for scipy minimize
bounds = ( ( -3, 3), (-1.5,1.5), (0, 3), (-1, 1), (-1, 1) ) # Bounds for fit if frequency is not fixed, for scipy minimize

bounds_fixfreq_cf = [ [ -3, -1.5, -1, -1 ], [ 3, 1.5, 1, 1 ] ] # Bounds for fit if frequency is fixed, for curve_fit
bounds_cf = [ [ -3, -1.5, 0, -1, -1 ], [ 3, 1.5, 3, 1, 1 ] ] # Bounds for fit if frequency is not fixed, for curve_fit

def averageData(data, n_ave):
    data = data[:int(np.floor(len(data)/n_ave)*n_ave)]
    data = np.mean( data.reshape(-1, n_ave), axis=1 ) # Average data
    return data

def initialfitting(k_ind, k, n_ave, dirName, files, dt, Fs, D1Start, D1EndTimes, D2Start, D2EndTimes, delta_t, sigma=1e-5):
    
    paraOut = np.array([1.1, 1.1, 1e-10, 1e-10 ]) # Initial parameters for fit
    data = np.loadtxt(dirName+"/"+files[k_ind])
    data = averageData( data, n_ave )
    times = np.arange( 1, len(data)+1, 1 )*dt
    
    DStart = D1Start
    DEnd = D1EndTimes[np.mod(k_ind-1,len(D2EndTimes))]
    ind_DStart = np.argmin( np.abs(times - DStart) )+1 # Index of start time
    ind_DEnd = np.argmin( np.abs(times - DEnd) ) # Index of end time
    
    datafit = np.array([ times[ind_DStart:ind_DEnd] - times[ind_DEnd-1], data[ind_DStart:ind_DEnd] ])
    
    # Run fit a few times to get initial parameters
    eps = 1e-12
    paraOut = optimize.minimize( lambda para: least_sq(datafit[1], fitSine2Slope([para[0], para[1], 1.554, para[2], para[3]], datafit[0]))/1e6, [ paraOut[0], paraOut[1], paraOut[2], paraOut[3] ], method="SLSQP", bounds=bounds_fixfreq, tol=1e-15, options={'eps': eps, 'ftol':1e-15, 'maxiter':5000} ).x
    paraOut = optimize.minimize( lambda para: least_sq(datafit[1], fitSine2Slope(para, datafit[0]))/1e5, [ paraOut[0], paraOut[1], 1.554, paraOut[2], paraOut[3] ], method="SLSQP", bounds=bounds, tol=1e-15, options={'eps': 1e-12, 'ftol':1e-15, 'maxiter':5000} ).x
    #datafit, errfac = BandPassFilter(datafit, dt, 200, 1) # Bandpass filter
    paraOut = optimize.minimize( lambda para: least_sq(datafit[1], fitSine2Slope(para, datafit[0])), paraOut, method="SLSQP", bounds=bounds, options={'eps': 1e-12} ).x # Run fit again

    ## Partial Fitting
    NPoints = len(datafit[0])
    nPoints = int(np.round(delta_t/dt))
    n_subs = int(np.floor(NPoints/nPoints))
    dataPar_x = datafit[0][:nPoints*n_subs]
    dataPar_y = datafit[1][:nPoints*n_subs]
    dataPar_x = np.reshape(dataPar_x, [n_subs, nPoints])
    dataPar_y = np.reshape(dataPar_y, [n_subs, nPoints]) 
    sigma = (np.sqrt(getSigma(datafit[1], dt)))
    paraOut_arr = np.empty((n_subs, 5))
    errtout_arr = np.empty((n_subs, 5))
    datafit2 = []
    for i in range(n_subs):
        paraOut_arr[i], errtout_arr[i], datafit2_i = fitSubsec( dataPar_x[i], dataPar_y[i], paraOut )
        errtout_arr[i] = [
            np.sqrt(sigma**2/len(datafit2_i)),
            np.sqrt(3*sigma**2/(np.pi**2*dt**2*paraOut_arr[i,0]**2*len(datafit2_i)*(len(datafit2_i)**2-1))),
            np.sqrt(2*sigma**2*(2*len(datafit2_i)-1)/(paraOut_arr[i,0]**2*len(datafit2_i)*(len(datafit2_i)+1))),
            np.sqrt(2*sigma**2/(dt*len(datafit2_i)*(len(datafit2_i)-1))), 
            sigma/np.sqrt(len(datafit2_i))
        ]
        datafit2.extend(datafit2_i)
        
    return np.array(paraOut_arr), np.array(errtout_arr), np.array(dataPar_x), np.array(dataPar_y), np.array(datafit2)

def fitSubsec( dataPar_x, dataPar_y, paraOut ):
    eps = 1e-12
    dataPar_x0 = dataPar_x - min(dataPar_x) # Set 0 to beginning of subsection of data
    # Run fit a few times with minimize to get good initial parameters
    fitResult = optimize.minimize( lambda para: least_sq(dataPar_y, fitSine2Slope([para[0], para[1], 1.554, para[2], para[3]], dataPar_x0))/1e4, [ paraOut[0], paraOut[1], paraOut[3], paraOut[4]], method="SLSQP", bounds=bounds_fixfreq, tol=1e-15, options={'eps': eps, 'maxiter': 10000, 'ftol':1e-15} )
    fitResult = optimize.minimize( lambda para: least_sq(dataPar_y, fitSine2Slope(para, dataPar_x0))/1e4, [ fitResult.x[0], fitResult.x[1], 1.554, fitResult.x[2], fitResult.x[3]], method="SLSQP", tol=1e-15, bounds=bounds, options={'eps': 1e-12, 'maxiter': 10000, 'ftol':1e-15} )

    # Run fit using curve fit
    paraOut, errtout = curve_fit( lambda xdata, *para: fitSine2Slope([para[0], para[1], fitResult.x[2], para[2], para[3]], xdata), dataPar_x0, dataPar_y, np.ones(len(dataPar_y)), [ fitResult.x[0], fitResult.x[1], fitResult.x[3], fitResult.x[4]], [], 2.0 )
    paraOut = [ paraOut[0], paraOut[1], fitResult.x[2], paraOut[2], paraOut[3] ]
    errtout = [ errtout[0], errtout[1], 0, errtout[2], errtout[3] ]
    paraNorm, errNorm = linearizedSine2Norm( paraOut, errtout )
    return paraNorm, errNorm, fitSine2Slope(paraOut, dataPar_x0)

def linearizedSine2Norm( paraOut, errtout ):
    paraNorm = [np.sqrt(paraOut[0]**2 + paraOut[1]**2), paraOut[2], np.arctan2(paraOut[1], paraOut[0]), # Convert linearized parameters to normal
                paraOut[3], paraOut[4]]
    errNorm = [paraNorm[0]**(-1) * np.sqrt( (errtout[0]*paraOut[0])**2 + (errtout[1]*paraOut[1])**2 ), # Convert linearized parameter errors to normal
            errtout[2], 
            np.abs(paraOut[0]*paraOut[1])/paraNorm[0]**2 * np.sqrt( (errtout[0]/paraOut[0])**2 + (errtout[1]/paraOut[1])**2 ),
            errtout[3], errtout[4]]
    return paraNorm, errNorm

def getSigma( data, dt ):
    f, P_den = signal.periodogram(data, 1/dt)
    sigma = np.sqrt(np.mean( P_den[np.argmin(np.abs(f-400)):np.argmin(np.abs(f-450))] )*(1/(2*dt)))
    return sigma