import numpy as np
import scipy.fftpack as fft

def chi2(expected, observed, error, ddof):
    return np.sum((expected-observed)**2 / error**2 / (len(expected)-ddof))

def getfft(arr, dt):
    L = len(arr)
    Fs = 1/dt
    f = Fs*np.linspace(0, 1, L)
    fftarr = fft.fft(arr)/L
    return fftarr, f