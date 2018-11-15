import sys, os
import numpy as np
import scipy.optimize as optimize
import scipy.fftpack as fft

def fitSine2Slope( para, t, t0=-999 ):
    if t0 == -999: t0 = 0
    a11, a12, f1, a21, a22, f2 = para
    return a11*np.sin(2*np.pi*f1*(t-t0)) + a12*np.cos(2*np.pi*f1*(t-t0)) + a21*np.sin(2*np.pi*f2*(t-t0)) + a22*np.cos(2*np.pi*f2*(t-t0))

def least_sq( y, yfit ):
    return sum( (y - yfit)**2 )

def Jacobian( para, t, t0=-999 ):
    if t0 == -999: t0 = 0
    a11, a12, f1, a21, a22, f2 = para
    return [
        np.sin(2*np.pi*f1*(t-t0)),
        np.cos(2*np.pi*f1*(t-t0)),
        2*np.pi*(t-t0)*(a11*np.sin(2*np.pi*f1*(t-t0)) + a12*np.cos(2*np.pi*f1*(t-t0))),
        np.sin(2*np.pi*f2*(t-t0)),
        np.cos(2*np.pi*f2*(t-t0)),
        2*np.pi*(t-t0)*(a21*np.sin(2*np.pi*f2*(t-t0)) + a22*np.cos(2*np.pi*f2*(t-t0)))
    ]

def Jacobian2( para, t, t0=-999 ):
    if t0 == -999: t0 = 0
    a1, f1, phi1, a2, f2, phi2 = para
    return [
        np.sin(2*np.pi*f1*(t-t0) + phi1),
        a1*2*np.pi*(t-t0)*np.cos(2*np.pi*f1*(t-t0) + phi1),
        a1*np.cos(2*np.pi*f1*(t-t0) + phi1),
        np.sin(2*np.pi*f2*(t-t0) + phi2),
        a2*2*np.pi*(t-t0)*np.cos(2*np.pi*f2*(t-t0) + phi2),
        a2*np.cos(2*np.pi*f2*(t-t0) + phi2)
    ]