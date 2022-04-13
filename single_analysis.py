#!/usr/bin/env python3
# import os; os.environ['OMP_NUM_THREADS']='1'
from analysis_tools_cython import *
from functools import reduce
from astropy.table import Table, unique
from astropy.stats import sigma_clip, sigma_clipped_stats
import data
import os
import loaders
import pandas as pd
import sys
import matplotlib.pyplot as plt
import argparse
import glob


parser = argparse.ArgumentParser(description="Analyse target lightcurve.")
parser.add_argument(help="Target lightcurve file", nargs=1, dest="fits_file")
parser.add_argument("-n", help="No graphical output", action="store_true")
parser.add_argument(
    "-q", help="Keep only points with SAP_QUALITY=0", action="store_true"
)


args = parser.parse_args()

# If TESS lightcurve, apply MAD. If Kepler lightcurve, skip to timestep
if (os.path.split(args.fits_file[0])[1].startswith("kplr")) or (
    os.path.split(args.fits_file[0])[1].startswith("hlsp_tess")
    and os.path.split(args.fits_file[0])[1].endswith("fits")
    or os.path.split(args.fits_file[0])[1].startswith("tess")
    and os.path.split(args.fits_file[0])[1].endswith("fits")
):
    table = import_lightcurve(args.fits_file[0])
    t, flux, quality, real = clean_data(table)

else:
    table, lc_info = (
        import_XRPlightcurve(args.fits_file[0], sector=6, clip=4, drop_bad_points=True)
    )
    fig, axarr = plt.subplots(1)
    print(len(table["time"]), ": length of lightcurve")
    print(unique(table, "quality"))
    to_clean = table["time", "corrected flux", "quality"]
    plt.scatter(table["time"], normalise_lc(table["corrected flux"]), s=5)
    plt.savefig(f"figs_tess/lightcurve {lc_info[0]} at import")
    t, flux, quality, real = clean_data(to_clean)


timestep = calculate_timestep(table)

# The default is a 30-minute cadence.
factor = (1 / 48) / timestep

N = len(t)
ones = np.ones(N)

flux = normalise_flux(flux)

# filteredflux = fourier_filter(flux, 8) # returns smooth lc
A_mag = np.abs(np.fft.rfft(flux))
# periodicnoise = flux - filteredflux

sigma = flux.std()

flux_ls = np.copy(flux)
lombscargle_filter(t, flux_ls, real, 0.05)  # happens in-place. 0.05 is minimum score
periodicnoise_ls = flux - flux_ls
flux_ls = flux_ls * real

# T1 = test_statistic_array(filteredflux, 60)
T = test_statistic_array(flux_ls, 60 * factor)
data_nonzeroT = nonzero(T)

# Find minimum test statistic value (m), and its location (n).
m, n = np.unravel_index(
    T.argmin(), T.shape
)  # T.argmin(): location of  T.shape: 2D array with x,y points in that dimension
# unravel_index: return values tell us what should have been the indices of the array if it was *not* flattened.
minT = T[m, n]
minT_time = t[n]
minT_duration = m * timestep
print("Timestep of lightcurve: ", round(timestep * 1440, 3), "minutes.")
print("Maximum transit chance:")
print("   Time =", round(minT_time, 2), "days.")
print("   Duration =", round(minT_duration, 2), "days.")
print("   T =", round(minT, 1))
print("   T/sigma =", round(minT / data_nonzeroT.std(), 1))

trans_start = n - math.floor((m - 1) / 2)
trans_end = trans_start + m
print("Transit depth =", round(flux[trans_start:trans_end].mean(), 6))

# Transit shape calculation
if (
    n - 3 * m >= 0 and n + 3 * m < N
):  # m: width of point(s) in lc. first part: 3 transit widths away from first data point. last part: not more than 3 transit widths away.
    t2 = t[n - 3 * m : n + 3 * m]
    x2 = flux_ls[n - 3 * m : n + 3 * m]
    q2 = quality[
        n - 3 * m : n + 3 * m
    ]  # quality points from three transit widths to other edge of three transit widths.
    background = (sum(x2[: 1 * m]) + sum(x2[5 * m :])) / (2 * m)
    x2 -= background
    paramsgauss = single_gaussian_curve_fit(t2, -x2)
    y2 = -gauss(t2, *paramsgauss)
    paramscomet = comet_curve_fit(t2, -x2)
    w2 = -comet_curve(t2, *paramscomet)

    scores = [score_fit(x2, fit) for fit in [y2, w2]]
    print("Asym score:", round(scores[0] / scores[1], 4))

    qual_flags = reduce(
        lambda a, b: a or b, q2
    )  # reduces to single value of quality flags
    print("Quality flags:", qual_flags)

# Classify events
asym, _, _ = calc_shape(m, n, t, flux)
print(classify(m, n, real, asym))

# Skip plotting if no graphical output set
if args.n:
    sys.exit()

# plt.xkcd()
fig1, axarr = plt.subplots(4, figsize=(18, 10))
plt.rcParams['font.size'] = '16'
plt.rcParams['axes.labelsize'] = '16'
axarr[0].plot(A_mag)  # fourier plot
axarr[0].title.set_text("Fourier plot")
axarr[1].plot(t, flux + ones, label="flux")
axarr[1].plot(t, periodicnoise_ls + ones, label="periodic noise")
axarr[1].title.set_text("Raw lightcurve and the periodic noise")
axarr[1].set_xlabel("Days in BTJD")
axarr[1].set_ylabel("Normalised flux")
axarr[1].legend(loc='lower left')
axarr[2].plot(t, flux_ls + ones)  # lomb-scargle plot
axarr[2].title.set_text("Noise-removed lightcurve")
axarr[2].set_xlabel("Days in BTJD")
axarr[2].set_ylabel("Normalised flux")
im = axarr[3].imshow(
    T,
    origin="bottom",
    extent=axarr[1].get_xlim() + (0, 2.5),
    aspect="auto",
    cmap="rainbow",
)
cax = fig1.add_axes([1.02, 0.01, 0.05, 0.25])
axarr[3].title.set_text("An image of the lightcurve")
axarr[3].set_xlabel("Days in BTJD")
axarr[3].set_ylabel("Transit width in days")
axarr[3].set_aspect("auto")
fig1.colorbar(im, cax=cax)
fig1.tight_layout()

# params = double_gaussian_curve_fit(T)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
# T_test_nonzero = np.array(data)
# _,bins,_ = ax2.hist(T_test_nonzero,bins=100,log=True)
# y = np.maximum(bimodal(bins,*params),10)
# ax2.plot(bins,y)
try:
    ax2.plot(t2, x2 + 1, label="flux")
    ax2.plot(t2, w2 + 1, label="comet curve", color="k")
    ax2.plot(t2, y2 + 1, label="gaussian", color="r")
    ax2.legend()
    ax2.set_title("Transit shape")
    ax2.set_xlabel("Days in BTJD")
    ax2.set_ylabel("Normalised flux")
except:
    pass


plt.show()
try:
    os.makedirs("figs_tess")  # make directory plot if it doesn't exist
except FileExistsError:
    pass

fig1.savefig("figs_tess/fourier plots", dpi=300)
