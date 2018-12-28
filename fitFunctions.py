import sys, os
import numpy as np
import scipy.optimize as optimize
import scipy.fftpack as fft
from scipy.linalg import svd

def least_sq( y, yfit ):
    return np.sum( (y - yfit)**2 )

def curve_fit( func, x, y, sigma, p0, bounds ):
    if bounds == []:
        fit = optimize.leastsq( lambda args: (func(x, *args)-y)/sigma, x0=p0, full_output=1, ftol=1e-15, gtol=1e-15, xtol=1e-15 )
        cov = fit[1]
        fit = fit[0]
        fitval = func(x, *fit)
        s_sq = np.sum( ((fitval-y)/sigma)**2 )/(len(x) - 2)
        if(len(x)==2):
            s_sq = np.sum( ((fitval-y)/sigma)**2 )/len(x)
        if cov is not None:
            cov = cov * s_sq
            fiterr = np.sqrt(np.diag(cov))
        else:
            print("NONE")
            fiterr = np.ones(len(fit))*1e9
    else:
        fit = optimize.least_squares( lambda args: (func(x, *args)-y)/sigma, x0=p0, bounds=bounds, ftol=1e-15, gtol=1e-15, xtol=1e-15 )
        jac = fit['jac']
        fit = fit['x']
        _, s, VT = svd(jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        cov = np.dot(VT.T / s**2, VT)
        fiterr = np.sqrt(np.diag(cov))
    return fit, fiterr

def fitSine2Slope( para, t, t0=-999 ):
    if t0 == -999: t0 = 0
    a11, a12, f1, a21, a22, f2, a, b = para
    return a11*np.sin(2*np.pi*f1*(t-t0)) + a12*np.cos(2*np.pi*f1*(t-t0)) + a21*np.sin(2*np.pi*f2*(t-t0)) + a22*np.cos(2*np.pi*f2*(t-t0)) + b + a*(t-t0)

def fitSine2Slope2( para, t, t0=-999 ):
    if t0 == -999: t0 = 0
    a1, f1, phi1, a2, f2, phi2, a, b = para
    return a1*np.sin(2*np.pi*f1*(t-t0) + phi1) + a2*np.sin(2*np.pi*f2*(t-t0) + phi2) + b + a*(t-t0)

def Jacobian( para, t, t0=-999 ):
    if t0 == -999: t0 = 0
    a11, a12, f1, a21, a22, f2, a, b = para
    return [
        np.sin(2*np.pi*f1*(t-t0)),
        np.cos(2*np.pi*f1*(t-t0)),
        2*np.pi*(t-t0)*(a11*np.cos(2*np.pi*f1*(t-t0)) - a12*np.sin(2*np.pi*f1*(t-t0))),
        np.sin(2*np.pi*f2*(t-t0)),
        np.cos(2*np.pi*f2*(t-t0)),
        2*np.pi*(t-t0)*(a21*np.cos(2*np.pi*f2*(t-t0)) - a22*np.sin(2*np.pi*f2*(t-t0))),
        (t-t0),
        1+0*(t-t0)
    ]

def Jacobian_nofreq( para, t, t0=-999 ):
    if t0 == -999: t0 = 0
    a11, a12, f1, a21, a22, f2, a, b = para
    return [
        np.sin(2*np.pi*f1*(t-t0)),
        np.cos(2*np.pi*f1*(t-t0)),
        np.sin(2*np.pi*f2*(t-t0)),
        np.cos(2*np.pi*f2*(t-t0)),
        (t-t0),
        1
    ]

def Jacobian2( para, t, t0=-999 ):
    if t0 == -999: t0 = 0
    a1, f1, phi1, a2, f2, phi2, a, b = para
    return [
        np.sin(2*np.pi*f1*(t-t0) + phi1),
        a1*2*np.pi*(t-t0)*np.cos(2*np.pi*f1*(t-t0) + phi1),
        a1*np.cos(2*np.pi*f1*(t-t0) + phi1),
        np.sin(2*np.pi*f2*(t-t0) + phi2),
        a2*2*np.pi*(t-t0)*np.cos(2*np.pi*f2*(t-t0) + phi2),
        a2*np.cos(2*np.pi*f2*(t-t0) + phi2)
    ]