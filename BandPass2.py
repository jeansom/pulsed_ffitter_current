import sys, os
import numpy as np
import scipy.optimize
import scipy.fftpack as fft
import scipy.signal as signal

def BandPassFilter(datafit, dt, paraOut, tCut=100, tEnd=100):
    nyq = 1/dt/2
    #Combine into a bandpass filter
    d = signal.firwin(101, np.array([0.5, 2.5, 14, 16]), nyq=nyq, window='hann', pass_zero=False)
    filtered_sig = signal.lfilter(d, 1.0, datafit[1])
    return [ datafit[0][tCut:-tEnd], filtered_sig[tCut:-tEnd] ]