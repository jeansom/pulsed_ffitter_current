import sys, os
import numpy as np
import scipy.optimize
import scipy.fftpack as fft
import scipy.signal as signal
import scipy.integrate as integrate
from chi2 import *

def BandPassFilter(datafit, dt, tCut=100, tEnd=100):
    nyq = 1/dt/2
    #Combine into a bandpass filter
    d = signal.firwin(101, [0.5, 2.5], nyq=nyq, window='hamming', pass_zero=False)
    filtered_sig = signal.convolve(datafit[1], d, mode='same', method='auto')
    fftd = getfft(d, dt)
    errfac = integrate.simps(np.abs(fftd[0]/max(fftd[0]))**2, fftd[1]) /(2*np.pi)   
    return [ datafit[0][tCut:-tEnd], filtered_sig[tCut:-tEnd] ], errfac