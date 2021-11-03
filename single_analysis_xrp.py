#!/usr/bin/env python3
# import os; os.environ['OMP_NUM_THREADS']='1'
from analysis_tools_cython import *
from functools import reduce
import data
import loaders
import pandas as pd
import sys
import matplotlib.pyplot as plt
import argparse
import glob
from tess_tools import *

parser = argparse.ArgumentParser(description="Analyse target lightcurve.")
parser.add_argument(help="Target lightcurve file", nargs=1, dest="fits_file")
parser.add_argument("-n", help="No graphical output", action="store_true")
parser.add_argument(
    "-q", help="Keep only points with SAP_QUALITY=0", action="store_true"
)


args = parser.parse_args()

# If TESS lightcurve, apply MAD. If Kepler lightcurve, skip to timestep
if os.path.split(args.fits_file[0])[1][:4] == "kplr":
    table = import_lightcurve(args.fits_file[0])
    t, flux, quality, real = clean_data(table)
else:

    table, lc_info = (
        import_XRPlightcurve(args.fits_file[0])[0],
        import_XRPlightcurve(args.fits_file[0])[1],
    )
    table["normalised PCA"] = normalise_lc(table["PCA flux"])
    bad_times = data.load_bad_times()
    bad_times = bad_times - 2457000
    mad_df = data.load_mad()

    sec = int(
        input("Sector? ")
    )  # int(os.path.basename(args.fits_file[0]).split('_')[2]) # eleanor lightcurve gives sector number in filename
    cam = lc_info[4]
    mad_arr = mad_df.loc[: len(table) - 1, f"{sec}-{cam}"]
    mad_cut = mad_arr.values < (np.nanmedian(mad_arr) + 10 * np.std(mad_arr[900:950]))
    mask = np.ones_like(table["time"], dtype=bool)
    for i in bad_times:
        newchunk = (table["time"] < i[0]) | (table["time"] > i[1])
        mask = mask & newchunk

    new_lc = table[
        (table["quality"] == 0) & mask & mad_cut
    ]  # applying mad cut to lightcurve
    to_clean = remove_zeros(new_lc)  # removing any zero points
    to_clean = to_clean["time", "PCA flux", "quality"]
    t, flux, quality, real = clean_data(to_clean)
timestep = calculate_timestep(table)

N = len(t)
ones = np.ones(N)

flux = normalise_flux(flux)

# filteredflux = fourier_filter(flux, 8) # returns smooth lc
A_mag = np.abs(np.fft.rfft(flux))
# periodicnoise = flux - filteredflux


sigma = flux.std()

flux_ls = np.copy(flux)
lombscargle_filter(t, flux_ls, real, 0.05)  # happens in place
periodicnoise_ls = flux - flux_ls
flux_ls = flux_ls * real

# T1 = test_statistic_array(filteredflux, 60)
T = test_statistic_array(flux_ls, 60)
data = nonzero(T)

# Find minimum test statistic value, and its location.
m, n = np.unravel_index(T.argmin(), T.shape)
minT = T[m, n]
minT_time = t[n]
minT_duration = m * timestep
print("Maximum transit chance:")
print("   Time =", round(minT_time, 2), "days.")
print("   Duration =", round(minT_duration, 2), "days.")
print("   T =", round(minT, 1))
print("   T/sigma =", round(minT / data.std(), 1))

trans_start = n - math.floor((m - 1) / 2)
trans_end = trans_start + m
print("Transit depth =", round(flux[trans_start:trans_end].mean(), 6))

# Transit shape calculation
if n - 3 * m >= 0 and n + 3 * m < N:
    t2 = t[n - 3 * m : n + 3 * m]
    x2 = flux_ls[n - 3 * m : n + 3 * m]
    q2 = quality[n - 3 * m : n + 3 * m]
    background = (sum(x2[: 1 * m]) + sum(x2[5 * m :])) / (2 * m)
    x2 -= background
    paramsgauss = single_gaussian_curve_fit(t2, -x2)
    y2 = -gauss(t2, *paramsgauss)
    paramscomet = comet_curve_fit(t2, -x2)
    w2 = -comet_curve(t2, *paramscomet)

    scores = [score_fit(x2, fit) for fit in [y2, w2]]
    print(scores)
    print("Asym score:", round(scores[0] / scores[1], 4))

    qual_flags = reduce(lambda a, b: a or b, q2)
    print("Quality flags:", qual_flags)

# Classify events
asym, _, _ = calc_shape(m, n, t, flux)
print(classify(m, n, real, asym))

# Skip plotting if no graphical output set
if args.n:
    sys.exit()

# plt.xkcd()
fig1, axarr = plt.subplots(4)
axarr[0].plot(A_mag)  # fourier plot
axarr[0].title.set_text("Fourier plot")
axarr[1].plot(t, flux + ones, t, periodicnoise_ls + ones)  #
axarr[2].plot(t, flux_ls + ones)  # lomb-scargle plot
axarr[2].title.set_text("Lomb-Scargle plot")
cax = axarr[3].imshow(T)
axarr[3].set_aspect("auto")
fig1.colorbar(cax)

# params = double_gaussian_curve_fit(T)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
# T_test_nonzero = np.array(data)
# _,bins,_ = ax2.hist(T_test_nonzero,bins=100,log=True)
# y = np.maximum(bimodal(bins,*params),10)
# ax2.plot(bins,y)
try:
    ax2.plot(t2, x2, t2, y2, t2, w2)
except:
    pass

plt.show()
