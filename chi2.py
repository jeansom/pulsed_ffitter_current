import numpy as np

def chi2(expected, observed, error, ddof):
    return np.sum((expected-observed)**2 / error**2 / (len(expected)-ddof))