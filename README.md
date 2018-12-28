pulsed_fitter_current.ipynb
  Runs analysis over data files. Saves output into a .gz file
  Functions
    N/A
  Calls
    partialfitting.py
  Called By
    N/A
partialfitting.py
  Runs partial fitting
  Functions
    averageData: averages data points in groups of length n_ave
    initialfitting: loads data from averages, filters, and runs partial fitting. Saves fit parameters, errors, and a time array
    fitSubsec: fits the data points (dataPar_x, dataPar_y) with two sine waves using paraOut as the initial point
    linearizedSine2Norm: Converts linearized sine parameters to a non-linearized sine parameters (ie, phase)
    getSigma: uses signal.welch to extract the noise level from data
  Calls
    partialfitting.py
    BandPass2.py
    fitfunctions.py
  Called By
    pulsed_fitter_current.ipynb

    
      
        
  
