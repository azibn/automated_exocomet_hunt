#!/usr/bin/env python3
# import os; os.environ['OMP_NUM_THREADS']='1'
from analysis_tools_cython import *
from functools import reduce
from astropy.table import Table, unique
from astropy.stats import sigma_clip, sigma_clipped_stats
import scipy.signal as signal
import data
import os
import loaders
import pandas as pd
import sys
import matplotlib.pyplot as plt
import argparse
import glob
import math


parser = argparse.ArgumentParser(description="Analyse target lightcurve.")
parser.add_argument(help="Target lightcurve file", nargs=1, dest="file")
parser.add_argument("-n", help="No graphical output", action="store_true")
parser.add_argument(
    "-q", help="Keep only points with SAP_QUALITY=0", action="store_true"
)
parser.add_argument("-s", help="TESS sector for XRP lightcurves",dest="sector",default=6)


args = parser.parse_args()

# If XRP TESS lightcurve, apply MAD. If Kepler lightcurve, skip to timestep
if (os.path.split(args.file[0])[1].startswith("kplr")) or (
    os.path.split(args.file[0])[1].startswith("hlsp_tess")
    and os.path.split(args.file[0])[1].endswith("fits")
    or "qlp" in os.path.split(args.file[0])[1]
):
    # or os.path.split(args.file[0])[1].startswith("tess")
    # and os.path.split(args.file[0])[1].endswith("fits")
    # ):
    table = import_lightcurve(args.file[0])
    t, flux, quality, real = clean_data(table)


elif "tasoc" in os.path.split(args.file[0])[1]:
    table = import_tasoclightcurve(args.file[0])
    plt.plot(table["TIME"], table["FLUX_CORR"])
    t, flux, quality, real = clean_data(table)

# elif os.path.split(args.file[0])[1].endswith(".csv"):
#     table = import_eleanor(args.file[0], args.sector, 1, 4, drop_bad_points=True)
#     t, flux, quality, real = clean_data(table)
else:
    table, lc_info = import_XRPlightcurve(
        args.file[0], sector=args.sector, clip=4, drop_bad_points=True
    )
    to_clean = table["time", "PCA flux", "quality"]
    t, flux, quality, real = clean_data(to_clean)


timestep = calculate_timestep(table)

# The default is a 30-minute cadence.
factor = (1 / 48) / timestep

N = len(t)
ones = np.ones(N)

flux = normalise_flux(flux)

# Fourier Transform
A_mag = np.abs(np.fft.rfft(flux))

# Lomb-Scargle

flux_ls = np.copy(flux)

# freq, powers = lombscargle_plotting(t,flux_ls,real,0.1)
lombscargle_filter(t, flux_ls, real, 0.08)  # happens in-place. 0.05 is minimum score

periodicnoise_ls = flux - flux_ls
flux_ls = flux_ls * real


freq, powers = lombscargle_plotting(t, flux_ls, real, 0.08)
# T1 = test_statistic_array(filteredflux, 60)
T = test_statistic_array(flux_ls, 60 * factor)
data_nonzeroT = nonzero(T)

# Find minimum test statistic value (m), and its location (n).
m, n = np.unravel_index(
    T.argmin(), T.shape
)  # T.argmin(): location of  T.shape: 2D array with x,y points in that dimension

# minT = T[m, n]
# minT_time = t[n]
# minT_flux = flux[n]
# minT_duration = m * timestep

masked_flux = np.copy(flux)
masked_flux[n - 2*math.ceil(n*timestep) : n + 2*math.ceil(n*timestep)] = 0  # 2x width of dip

original_masked_flux = np.copy(masked_flux)
lombscargle_filter(t, masked_flux, real, 0.08)
periodicnoise_ls2 = original_masked_flux - masked_flux
masked_flux = masked_flux * real
final_flux = flux - periodicnoise_ls2
final_flux = final_flux * real

T_new = test_statistic_array(final_flux, 60 * factor)
data_nonzeroT = nonzero(T_new)

m2, n2 = np.unravel_index(T_new.argmin(), T_new.shape)

minT = T_new[m2, n2]
minT_time = t[n2]
minT_flux = flux[n2]
minT_duration = m2 * timestep

print("Timestep of lightcurve: ", round(timestep * 1440, 3), "minutes.")
print("Maximum transit chance:")
print("   Time =", round(minT_time, 2), "days.")
print("   Duration =", round(minT_duration, 2), "days.")
print("   T =", round(minT, 1))
print("   T/sigma =", round(minT / data_nonzeroT.std(), 1))

trans_start = n2 - math.floor((m2 - 1) / 2)
trans_end = trans_start + m2
print("Transit depth =", round(flux[trans_start:trans_end].mean(), 6))

# Transit shape calculation
if (
    n2 - 3 * m2 >= 0 and n2 + 3 * m2 < N
):  # m: width of point(s) in lc. first part: 3 transit widths away from first data point. last part: not more than 3 transit widths away.
    t2 = t[n2 - 3 * m2 : n2 + 3 * m2]
    x2 = final_flux[n2 - 3 * m2 : n2 + 3 * m2]
    q2 = quality[
        n2 - 3 * m2 : n2 + 3 * m2
    ]  # quality points from three transit widths to other edge of three transit widths.
    background = (sum(x2[: 1 * m2]) + sum(x2[5 * m2 :])) / (2 * m2)
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
asym= calc_shape(m2, n2, t, quality, flux)
print(classify(m2, n2, real, asym))



fig1, axarr = plt.subplots(8, figsize=(13, 22))
# plt.rcParams['font.size'] = '16'
# plt.rcParams['axes.labelsize'] = '16'
axarr[0].plot(A_mag)  # fourier plot
axarr[0].title.set_text("Fourier plot")
axarr[0].set_xlabel("frequency")
axarr[0].set_ylabel("power")
axarr[1].plot(freq, powers)
axarr[1].title.set_text("Lomb-Scargle plot")
axarr[1].set_xlabel("frequency")
axarr[1].set_ylabel("Lomb-Scargle power")
axarr[2].plot(t, flux + ones, label="flux")
axarr[2].plot(t, periodicnoise_ls + ones, label="periodic noise")
axarr[2].title.set_text("Original lightcurve and the periodic noise")
axarr[2].set_xlabel("Days in BTJD")
axarr[2].set_ylabel("Normalised flux")
axarr[2].legend(loc="lower left")
axarr[3].plot(t, flux_ls + ones)  # lomb-scargle plot
# axarr[3].plot(m,marker='o')
axarr[3].title.set_text("First noise-removed lightcurve (first Lomb-Scargle)")
axarr[3].set_xlabel("Days in BTJD")
axarr[3].set_ylabel("Normalised flux")

im = axarr[4].imshow(
    T,
    origin="bottom",
    extent=axarr[3].get_xlim() + (0, 2.5),
    aspect="auto",
    cmap="rainbow",
)

cax = fig1.add_axes([1.02, 0.09, 0.05, 0.25])
axarr[4].title.set_text("An image of the lightcurve (first Lomb Scargle)")
axarr[4].set_xlabel("Days in BTJD")
axarr[4].set_ylabel("Transit width in days")
axarr[4].set_aspect("auto")
axarr[5].plot(t, original_masked_flux + ones, label="masked flux")
axarr[5].plot(t, periodicnoise_ls2 + ones, label="periodic noise")
axarr[5].legend()
axarr[5].title.set_text("Lomb-Scargle applied on masked flux")
axarr[6].plot(t, final_flux + ones, label="final flux")
axarr[6].plot(t, flux + ones - 0.001, color="red", label="original flux")
axarr[6].legend()
axarr[6].title.set_text("Cleaned flux")

im = axarr[7].imshow(
    T_new,
    origin="bottom",
    extent=axarr[3].get_xlim() + (0, 2.5),
    aspect="auto",
    cmap="rainbow",
)
axarr[7].title.set_text("An image of the final lightcurve (second Lomb Scargle)")
axarr[7].set_xlabel("Days in BTJD")
axarr[7].set_ylabel("Transit width in days")

fig1.colorbar(im, cax=cax)
fig1.tight_layout()

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

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


# Skip showing plots if no graphical output set
if args.n:
    sys.exit()
plt.show()

try:
    os.makedirs("figs_single_analysis")  # make directory plot if it doesn't exist
except FileExistsError:
    pass

try:
    fig1.savefig(f"figs_single_analysis/plots_TIC{lc_info[0]}", dpi=300)
except: 
    fig1.savefig(f"figs_single_analysis/plots_{args.file[0].split('/')[-1].split('.fits')[0]}", dpi=300)
