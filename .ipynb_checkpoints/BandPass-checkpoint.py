import sys, os
import numpy as np
import scipy.optimize
import scipy.fftpack as fft
import fitFunctions
from fitFunctions import *

def BandPassFilter(data, dt, para, tExt1 = 0, rmSec = 2):
    tAdd = np.linspace(1, np.round(tExt1/dt), round(tExt1/dt))*dt
    t_BF = np.flip(-tAdd) + data[0,0]
    
    # Frequencies              Xe_low_cutoff   Xe_low_trans_width   Xe_high_cutoff   Xe_high_trans_width   He_low_cutoff   He_low_trans_width    He_high_cutoff   He_high_trans_width
    XeHe_cutoff_trans_freq = [       0.5,               0.2,                2.5,            0.2,               14,                 0.2,                16,               0.2 ]
    data_filt = np.array([ np.concatenate( (t_BF, data[0], tAdd) ), np.concatenate( (fitSine2Slope(para, t_BF), data[1], fitSine2Slope(para, tAdd)) )])
    Lfilt = len(data_filt[1])
    Fs = 1/dt
    f = Fs*np.linspace(0, 1, Lfilt)
    fftdata = np.zeros((2,Lfilt))
    fftdata[0] = f
    fftdata[1] = fft.fft(data_filt[1])/Lfilt
    fftdata[1] = Fermi_Dirac_dig_filt( fftdata[0], fftdata[1], *XeHe_cutoff_trans_freq )
    data_filt[1] = Lfilt*np.real(np.fft.ifft(fftdata[1]))
    
    len1 = len(tAdd)+1
    len2 = len(data[1]) + len(tAdd)
    
    datafit = data
    datafit[1] = data_filt[1, len1:len2+1]
    nlast = int(rmSec/dt)
    stopR = len(datafit[0])-nlast
    startR = nlast
    
    datafit = datafit[:, startR:stopR+1]
    return datafit

def Fermi_Dirac_dig_filt(freq, F, *args ):
    count = len(F)
    proc_F = F
    f_max = max(freq)
    Xe_low_cutoff_freq, Xe_low_trans_freq_width, Xe_high_cutoff_freq, Xe_high_trans_freq_width, He_low_cutoff_freq, He_low_trans_freq_width, He_high_cutoff_freq, He_high_trans_freq_width = args
    
    for n in np.arange(0, count, 1):
        proc_F[n] = F[n]*(1/(np.exp((freq[n]-Xe_high_cutoff_freq)/Xe_high_trans_freq_width)+1))*(1/(np.exp(-(freq[n]-Xe_low_cutoff_freq)/Xe_low_trans_freq_width)+1)) \
        + F[n]*(1/(np.exp(-(freq[n]-f_max + Xe_high_cutoff_freq)/Xe_high_trans_freq_width)+1))*(1/(np.exp((freq[n]-f_max + Xe_low_cutoff_freq)/Xe_low_trans_freq_width)+1)) \
        + F[n]*(1/(np.exp((freq[n]-He_high_cutoff_freq)/He_high_trans_freq_width)+1))*(1/(np.exp(-(freq[n]-He_low_cutoff_freq)/He_low_trans_freq_width)+1)) \
        + F[n]*(1/(np.exp(-(freq[n]-f_max + He_high_cutoff_freq)/He_high_trans_freq_width)+1))*(1/(np.exp((freq[n]-f_max + He_low_cutoff_freq)/He_low_trans_freq_width)+1));
    return proc_F