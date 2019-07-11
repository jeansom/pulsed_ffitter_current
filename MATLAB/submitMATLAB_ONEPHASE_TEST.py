import sys, os, copy
import random
import numpy as np

batch='''#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=5:00:00
#SBATCH --mem=2GB
'''

zpol_arr = [ 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 ]
magx_arr = [ 5, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6  ]
dur_dark_arr = [ 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120 ]
g_arr = [ 0.1, 1, 10, 100, 500, 1000, 1500, 2000, 2500, 3000 ]
B0fac_arr = np.linspace(0,1,20) #[ 0.01, 0.1, 0.2, 0.5, 0.7, 0.9, 1 ]

runs = []

#for zpol in zpol_arr:
#        for B0fac in B0fac_arr:
#                runs.append([ 1, 1, 0, 5.5, 5.5, 5, B0fac, zpol, 1500, 30 ])
#for zpol in zpol_arr:
#        for magx in magx_arr:
#                runs.append([ 1, 1, 0, magx, magx, 1, 0, zpol, 1500, 30 ])
#                runs.append([ 1, 1, 0, magx, magx, 1, 1, zpol, 1500, 30 ])
#for zpol in zpol_arr:
#        for dur_dark in dur_dark_arr:
#                runs.append([ 1, 1, 0, 5.5, 5.5, 5, 0, zpol, 1500, dur_dark ])
#                runs.append([ 1, 1, 0, 5.5, 5.5, 5, 1, zpol, 1500, dur_dark ])
#for zpol in zpol_arr:
#        for g in g_arr:
#                runs.append([ 1, 1, 0, 5.5, 5.5, 5, 0, zpol, g, 30 ])
#                runs.append([ 1, 1, 0, 5.5, 5.5, 5, 1, zpol, g, 30 ])
#for magx in magx_arr:
#        for magy in magx_arr:
#                if 10*(magx-5)%2 < 1e-10 and 10*(magy-5)%2 < 1e-10 and magx != magy:
#                        print(magx, magy)
#                        runs.append([ 1, 1, 0, magx, magy, 1, 1, 0, 1500, 10 ])
#                        runs.append([ 1, 1, 0, magx, magy, 1, 0, 0.05, 1500, 20 ])
#                        runs.append([ 1, 1, 0, magx, magy, 1, 0, 1, 1500, 20 ])

runs.append([ 1, 1, 0, 5.5, 5.5, 5, 0, 0, 1500, 30 ])
runs.append([ 1, 1, 0, 5.5, 5.5, 5, 0, 0.1, 1500, 30 ])
for run in runs:
        noisex, noisey, noisez, magx, magy, magz, B0fac, zpol, g, dur_dark = run
        randseed = random.randint(1, 2**(32)-1)

        options = str(dur_dark) + ", " + str(noisex) + ", " + str(noisey) + ", " + str(noisez) + ", " + str(magx) + ", " + str(magy) + ", " + str(magz) + ", " + str(zpol) + ", " + str(B0fac) + ", " + str(g) + ", " + str(randseed) 
        tag = 'dark-' + str(dur_dark) + '_x-' + str(noisex) + '_y-' + str(noisey) + '_z-' + str(noisez) + '_magx-' + str(magx) + '_magy-' + str(magy) + '_magz-' + str(magz)  + '_zpol-' + str(zpol) + '_B0fac-' + str(B0fac) + '_g-' + str(g)
        
        batchn = copy.copy(batch)
        batchn += "#SBATCH --output=slurm/BFieldNoise_"+tag+".out\n"
        batchn += "cd /tigress/somalwar/RomalisResearch/MATLAB/\n"
        
        batchn += "/usr/licensed/bin/matlab -singleCompThread -nodisplay -nosplash -nojvm -r 'BFieldNoise_ONEPHASE_ODE5_SAVEALL(" + options + " )'\n"
        fname = "batch/BFieldNoise_"+tag+".sh"
        f=open(fname, "w")
        f.write(batchn)
        f.close()
        os.system("chmod +x " + fname);
        os.system("sbatch " + fname);
