pulsed_fitter_current.ipynb
  
  * Runs analysis over data files. Saves output into a .gz file
  * Functions
    * N/A
  * Calls
    * partialfitting.py
    * corr2Pi.py
    * fitPhases.py
    * corrphase.py
  * Called By
    * N/A

partialfitting.py
  * Runs partial fitting
  * Functions
    * averageData: averages data points in groups of length n_ave
    * initialfitting: loads data from averages, filters, and runs partial fitting. Saves fit parameters, errors, and a time array
    * fitSubsec: fits the data points (dataPar_x, dataPar_y) with two sine waves using paraOut as the initial point
    * linearizedSine2Norm: Converts linearized sine parameters to a non-linearized sine parameters (ie, phase)
    * getSigma: uses signal.welch to extract the noise level from data
  * Calls
    * partialfitting.py
    * BandPass2.py
    * fitfunctions.py
  * Called By
    * pulsed_fitter_current.ipynb

corr2Pi.py
  * Corrects phase by 2Pi
  * correct2Pi: loops over an array of phases and calls find2PiN to find the correct number of 2Pi's to add to each phase. * * Returns the phase array corrected by 2Pis
  * find2PiN: uses linregress to find the correct number of 2Pi's (\pm1, 0) to add to the last element of an array of phases
  * correctPhases: loads phases created by partialfitting.py or phases given in arrays, corrects them by 2Pi, and saves in  saveDirName
  * Calls
    * corr2Pi.py
    * fitfunctions.py
  * Called By
    * pulsed_fitter_current.ipynb
        
  
