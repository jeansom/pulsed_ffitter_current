import sys, os, copy
import random
import numpy as np

batch='''#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=12:00:00
#SBATCH --mem=2GB
'''

zpol_arr = [ 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 ]
mag_arr = [ 5, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0 ]
dur_dark_arr = [ 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120 ]
g_arr = [ 0.1, 1, 10, 100, 500, 1000, 1500, 2000, 2500, 3000 ]
B0 = 5
runs = []

for zpol in zpol_arr:
        for mag in mag_arr:
                runs.append([ mag, 1500, B0, 30, zpol ])
for zpol in zpol_arr:
        for dur_dark in dur_dark_arr:
                runs.append([ 5.5, 1500, B0, dur_dark, zpol ])
for zpol in zpol_arr:
        for g in g_arr:
                runs.append([ 5, g, B0, 30, zpol ])

for run in runs:
        mag, g, B0, dur_dark, zpol = run
        randseed = random.randint(1, 2**(32)-1)

        options = str(mag) + " " + str(g) + " " + str(B0) + " " + str(dur_dark) + " " + str(zpol)
        tag = 'mag-' + str(mag) + '_g-' + str(g) + '_B0-' + str(B0) + '_dark-' + str(dur_dark) + '_pz-' + str(zpol)
        
        batchn = copy.copy(batch)
        batchn += "#SBATCH --output=slurm/BFieldNoise_"+tag+".out\n"
        batchn += "cd /tigress/somalwar/RomalisResearch/MATLAB/\n"
        
        batchn += "python BFieldNoise_ONEPHASE.py " + options + "\n"
        fname = "batch/BFieldNoise_"+tag+".sh"
        f=open(fname, "w")
        f.write(batchn)
        f.close()
        os.system("chmod +x " + fname);
        os.system("sbatch " + fname);
        break
