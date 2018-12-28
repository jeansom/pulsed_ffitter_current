import sys, os
import numpy as np
import scipy.optimize
import scipy.integrate as integrate
import scipy.fftpack as fft
import scipy.signal as signal
from chi2 import *

def BandPassFilter(datafit, dt, paraOut, tCut=100, tEnd=100):
    nyq = 1/dt/2
    #Combine into a bandpass filter
    width = 3.0/nyq
    ripple_db = 100.0
    N, beta = signal.kaiserord(ripple_db, width)
    d = signal.firwin(N, [ 1e-15/nyq, 1/nyq, 2/nyq, 14/nyq, 16/nyq ], window=('kaiser', beta))
    freqz = signal.freqz(d, worN=8000)[1]
    d = d/max(np.abs(freqz))
    fftd = getfft(d, dt) 
    filtered_sig = signal.convolve(datafit[1], d, mode='same', method='auto')
    return np.array(datafit[0][tCut:-tEnd]), np.array(filtered_sig[tCut:-tEnd])