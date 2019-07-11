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
magx_arr = [ 5, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6  ]
dur_dark_arr = [ 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120 ]
g_arr = [ 0.1, 1, 10, 100, 500, 1000, 1500, 2000, 2500, 3000 ]
B0fac_arr = [ 0.01, 0.5, 1 ]

runs = []

#for zpol in zpol_arr:
#        for magx in magx_arr:
#                runs.append([ 1, 1, 0, magx, magx, 1, 1, zpol, 150, 1500, 30 ])
#for zpol in zpol_arr:
#        for dur_dark in dur_dark_arr:
#                runs.append([ 1, 1, 0, 5.5, 5.5, 5, 1, zpol, 150, 1500, dur_dark ])
for g1 in [100, 200, 300]:
        for g2 in [ 1000, 2000, 3000 ]:
                runs.append([ 1, 1, 0, 5.5, 5.5, 5, 1, 0, g1, g2, 30 ])
                runs.append([ 1, 1, 0, 5.5, 5.5, 5, 1, 0.03, g1, g2, 30 ])
                runs.append([ 1, 1, 0, 5.5, 5.5, 5, 1, 0.06, g1, g2, 30 ])
                runs.append([ 1, 1, 0, 5.5, 5.5, 5, 1, 0.1, g1, g2, 30 ])

for B0fac in B0fac_arr:
        runs.append([ 1, 1, 0, 5.5, 5.5, 5, B0fac, 0, 150, 1500, 30 ])
        runs.append([ 1, 1, 0, 5.5, 5.5, 5, B0fac, 0.1, 150, 1500, 30 ])

#runs.append([ 0, 1, 0, 5.5, 5.5, 5, 0, 0, 1500, 30 ])
for run in runs:
        noisex, noisey, noisez, magx, magy, magz, B0fac, zpol, g1, g2, dur_dark = run
        randseed = random.randint(1, 2**(32)-1)

        options = str(dur_dark) + ", " + str(noisex) + ", " + str(noisey) + ", " + str(noisez) + ", " + str(magx) + ", " + str(magy) + ", " + str(magz) + ", " + str(zpol) + ", " + str(B0fac) + ", " + str(g1) + ", " + str(g2) + ", " + str(randseed) 
        tag = 'dark-' + str(dur_dark) + '_x-' + str(noisex) + '_y-' + str(noisey) + '_z-' + str(noisez) + '_magx-' + str(magx) + '_magy-' + str(magy) + '_magz-' + str(magz)  + '_zpol-' + str(zpol) + '_B0fac-' + str(B0fac) + '_g1-' + str(g1) + '_g2-' + str(g2)
        
        batchn = copy.copy(batch)
        batchn += "#SBATCH --output=slurm/BFieldNoise_"+tag+".out\n"
        batchn += "cd /tigress/somalwar/RomalisResearch/MATLAB/\n"
        
        batchn += "/usr/licensed/bin/matlab -singleCompThread -nodisplay -nosplash -nojvm -r 'BFieldNoise_TWOPHASE_ODE5(" + options + " )'\n"
        fname = "batch/BFieldNoise_"+tag+".sh"
        f=open(fname, "w")
        f.write(batchn)
        f.close()
        os.system("chmod +x " + fname);
        os.system("sbatch " + fname);
