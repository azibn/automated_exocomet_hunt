#!/usr/bin/env python3
from analysis_tools_cython import *
import sys
import os
import matplotlib.pyplot as plt

if len(sys.argv) > 1:
    fits_file = sys.argv[1]
    table=import_lightcurve(fits_file)
else:
    print("Missing argument.")
    print("Usage:",sys.argv[0],"[FILENAME]")
    sys.exit()


timestep = calculate_timestep(table)
t,flux,real = clean_data(table)
N = len(t)
ones = np.ones(N)

flux = normalise_flux(flux)

filteredflux = fourier_filter(flux,8)
A_mag = np.abs(np.fft.rfft(flux))
periodicnoise = flux-filteredflux
sigma = flux.std()

flux_ls = np.copy(flux)
lombscargle_filter(t,flux_ls,real,0.05)
periodicnoise_ls = flux - flux_ls
flux_ls = flux_ls * real

T1 = test_statistic_array(filteredflux,60)
T = test_statistic_array(flux_ls,60)
data = nonzero(T)

# Find minimum test statistic value, and its location.
m,n = np.unravel_index(T.argmin(),T.shape)
minT = T[m,n]
minT_time = t[n]
minT_duration = 2*m*timestep
print("Maximum transit chance:")
print("   Time =",round(minT_time,2),"days.")
print("   Duration =",round(minT_duration,2),"days.")
print("   T =",round(minT,1))
print("Transit depth =",round(flux[n-m:n+m].mean(),6))


fig1,axarr = plt.subplots(4)
axarr[0].plot(A_mag)
axarr[1].plot(t,flux+ones,t,periodicnoise_ls+ones)
axarr[2].plot(t,flux_ls+ones)
cax = axarr[3].imshow(T)
axarr[3].set_aspect('auto')
fig1.colorbar(cax)

#params = double_gaussian_curve_fit(T)
#fig2 = plt.figure()
#ax2 = fig2.add_subplot(111)
#T_test_nonzero = np.array(data)
#_,bins,_ = ax2.hist(T_test_nonzero,bins=100,log=True)
#y = np.maximum(bimodal(bins,*params),10)
#ax2.plot(bins,y)

plt.show()
