#!/usr/bin/env python3
# import os; os.environ['OMP_NUM_THREADS']='1'
from analysis_tools_cython import *
from functools import reduce
import os
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Analyse target lightcurve.")
parser.add_argument(help="Target lightcurve file", nargs=1, dest="fits_file")
parser.add_argument("-f", help="Choose flux", default='PDCSAP_FLUX',dest='f',nargs=1)
parser.add_argument("-n", help="No graphical output", action="store_true")
parser.add_argument(
    "-q", help="Keep only points with SAP_QUALITY=0", action="store_true"
)
parser.add_argument("-c", help="Choose sigma clip threshold for XRP lightcurves. Default 4.", default=4,dest='c')

args = parser.parse_args()

if (os.path.split(args.fits_file[0])[1].startswith('kplr')) or (os.path.split(args.fits_file[0])[1].startswith("hlsp_tess") and os.path.split(args.fits_file[0])[1].endswith("fits") or os.path.split(args.fits_file[0])[1].startswith("tess") and os.path.split(args.fits_file[0])[1].endswith("fits")):
    table = import_lightcurve(args.fits_file[0])
    t, flux, quality, real = clean_data(table)

else:
    table,lc_info = ( 
        import_XRPlightcurve(args.fits_file[0],sector=6,clip=args.c,drop_bad_points=True)[0],
        import_XRPlightcurve(args.fits_file[0],sector=6,clip=args.c,drop_bad_points=True)[1],    
    )
    table = table["time", args.f, "quality"]
    t, flux, quality, real = clean_data(to_clean) # only kept because t is needed for plotting and transit shape purposes...

N = len(t)
ones = np.ones(N)
timestep = calculate_timestep(table)

Tm_info,params = processing(table,args.fits_file[0],single_analysis=True)

## Getting transit information
print("Timestep of lightcurve: ", round(timestep * 1440,3), "minutes.")
print("Maximum transit chance:")
print("   Time =", round(Tm_info[3], 2), "days.")
print("   Duration =", round(Tm_info[4], 2), "days.")
print("   T =", round(Tm_info[2], 1))
print("   T/sigma =", round(Tm_info[2] / Tm_info[-1], 1))
print("Transit depth =", round(Tm_info[5], 6))

m = Tm_info[0]
n = Tm_info[1]

## Transit shape calculation
if n - 3 * m >= 0 and n + 3 * m < N:  # m: width of point(s) in lc. first part: 3 transit widths away from first data point. last part: not more than 3 transit widths away.
    t2, x2, y2, w2, q2 = transit_shape(table,m,n,N,params) 

    scores = [score_fit(x2, fit) for fit in [y2, w2]]
    print("Asym score:", round(scores[0] / scores[1], 4))
    qual_flags = reduce(lambda a, b: a or b, q2) # reduces to single value of quality flags
    print("Quality flags:", qual_flags)

## Classify events
asym, _, _ = calc_shape(m, n, t, flux)
print(classify(m, n, real, asym))

## Skip plotting if no graphical output set
if args.n:
    sys.exit()

## plots
fig1, axarr = plt.subplots(4,figsize=(8,7))
axarr[0].plot(params[0])  # fourier plot
axarr[0].title.set_text("Fourier plot")
axarr[1].plot(t, params[2] + ones) # the lightcurve
axarr[1].plot(t, params[1] + ones,c='orange')  # the periodic noise
axarr[1].title.set_text("Periodic noise plot")
axarr[2].plot(t, params[3] + ones)  # lomb-scargle plot
axarr[2].title.set_text("Lomb-Scargle plot")
cax = axarr[3].imshow(params[4])
axarr[3].set_ylabel('Normalized flux + offset')
axarr[3].set_xlabel('BJD - 2457000')
axarr[3].set_ylabel('Transit width in timesteps')
axarr[3].set_aspect("auto")
fig1.colorbar(cax)
fig1.tight_layout()

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

try:
    ax2.plot(t2, x2, t2, y2, t2, w2)
    ax2.set_title('Transit shape in box')
except:
    pass

plt.show()