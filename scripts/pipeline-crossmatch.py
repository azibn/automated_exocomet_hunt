#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
from analysis_tools_cython import import_lightcurve, processing
import eleanor
import lightkurve as lk
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


df = pd.read_csv('candidates/272-candidates.csv')
data = df[df.tags == 'red'].reset_index(drop=True)
print("read in dataframe.")


os.makedirs('eleanor-plots', exist_ok=True)

print("starting plots...")
for i in tqdm(data.index):
    # try:
    filepath = data.iloc[i].abs_path
    lc, lc_info = import_lightcurve(filepath)

    tic_id = data.iloc[i].TIC_ID
    sector = data.iloc[i].Sector
    transit_time = data.iloc[i].time

    # Lightkurve TESS cutout
    tpf = lk.search_tesscut(f'TIC {tic_id}', sector=sector).download_all(cutout_size=(15, 15))
    
    ## higher sigma cut
    tpf = tpf[0]
    custom_mask = tpf.create_threshold_mask(threshold=10)
    lkcurve = tpf.to_lightcurve(aperture_mask=custom_mask)
    lkcurveq = lkcurve.quality == 0

    ## lower sigma cut
    #tpf2 = tpf[0]
    custom_mask2 = tpf.create_threshold_mask(threshold=3)
    lkcurve2 = tpf.to_lightcurve(aperture_mask=custom_mask2)
    lkcurveq2 = lkcurve2.quality == 0

    # Lightkurve QLP
    qlp = lk.search_lightcurve(f"TIC {tic_id}", sector=sector, author='QLP', exptime=1800)

    # SPOC
    spoc = lk.search_lightcurve(f"TIC {tic_id}", sector=sector, author='TESS-SPOC')
    
    # Check if any data products are available
    if len(qlp) == 0:
        qlp = None
    else:
        qlp = qlp.download()
        q = qlp.quality == 0

    if len(spoc) == 0:
        spoc = None
    else:
        spoc = spoc.download()
        qu = spoc.quality == 0

    # Plotting
    fig, axs = plt.subplots(5, 2, figsize=(22, 13))

    # Plotting the eleanor-lite ones
    axs[0, 0].plot(lc['TIME'], lc['RAW_FLUX'] / np.nanmedian(lc['RAW_FLUX']))
    axs[0, 0].axvline(transit_time, linestyle='--', linewidth=2, color='red', zorder=1)
    axs[0, 0].set_title('RAW FLUX', fontsize=14)

    axs[0, 1].plot(lc['TIME'], lc['PCA_FLUX'] / np.nanmedian(lc['PCA_FLUX']))
    axs[0, 1].axvline(transit_time, linestyle='--', linewidth=2, color='red', zorder=1)
    axs[0, 1].set_title('PCA FLUX', fontsize=14)

    # Check if QLP lightcurve exists
    if qlp:
        axs[1, 0].plot(qlp.time.value[q], qlp.flux[q]) # sap flux
        axs[1, 0].axvline(transit_time, linestyle='--', linewidth=2, color='red', zorder=1)
        axs[1, 0].set_title('QLP (SAP) FLUX', fontsize=14)

    axs[1, 1].plot(lc['TIME'], lc['FLUX_BKG'])
    axs[1, 1].axvline(transit_time, linestyle='--', linewidth=2, color='red', zorder=1)
    axs[1, 1].set_title('FLUX BACKGROUND', fontsize=14)


    if spoc:
        axs[2,0].plot(spoc.time.value[qu], spoc.pdcsap_flux[qu])
        axs[2, 0].axvline(transit_time, linestyle='--', linewidth=2, color='red', zorder=1)
        axs[2, 0].set_title('SPOC PDCSAP', fontsize=14)

    # Background subtracted
    regressors = tpf.flux[:, ~custom_mask]
    bkg = np.median(regressors, axis=1)
    bkg -= np.percentile(bkg, 5)

    npix = custom_mask.sum()
    median_subtracted_lc = lkcurve - npix * bkg
    median_subtracted_lc.plot(ax=axs[2,1], normalize=True, label="Median Subtracted")
    axs[2, 1].axvline(transit_time, linestyle='--', linewidth=2, color='red', zorder=1)
    axs[2,1].set_title('RECONSTRUCTED LIGHTCURVE - 10 SIGMA, BACKGROUND SUBTRACTED', fontsize=14)


    # Plot the TPF in the first subplot
    tpf.plot(ax=axs[3, 0], aperture_mask=custom_mask)
    lkcurve[lkcurveq].plot(ax=axs[3, 1])
    axs[3, 1].axvline(transit_time, linestyle='--', linewidth=2, color='red', zorder=1)
    axs[3, 1].set_title('RECONSTRUCTED LIGHTCURVE - 10 SIGMA', fontsize=14)

    # lower sigma cut TPF
    tpf.plot(ax=axs[4, 0], aperture_mask=custom_mask2)
    lkcurve2[lkcurveq2].plot(ax=axs[4, 1])
    axs[4, 1].axvline(transit_time, linestyle='--', linewidth=2, color='red', zorder=1)
    axs[4, 1].set_title('RECONSTRUCTED LIGHTCURVE - 3 SIGMA', fontsize=14)

    plt.subplots_adjust(hspace=0.5)
    fig.suptitle(f'TIC {tic_id}', fontsize=18)

    # Construct the filename
    filename = f'compare-plots/TIC{tic_id}.png'
    
    # Check if the file already exists
    counter = 1
    while os.path.exists(filename):
        # If the file exists, increment the counter and modify the filename
        counter += 1
        filename = f'compare-plots/TIC{tic_id}_{counter}.png'

    # Save the figure with the updated filename
    fig.savefig(filename, dpi=300, bbox_inches=None)
    plt.close()  # Close the current figure to avoid overlapping in the next iteration
# except AttributeError:
    #     print(f"No data found for target {data.iloc[i].TIC_ID}. Only printing available plots")
print("plots complete.")

# # ## Finding Spectral Types

# # In[9]:


# from astroquery.simbad import Simbad
# import pandas as pd

# custom_simbad = Simbad()
# custom_simbad.add_votable_fields('sptype')

# def get_spectral_type(tic_id):
#     try:
#         tic = "TIC " + str(tic_id)
#         result = custom_simbad.query_object(tic)
#         sptype = result['SP_TYPE'][0] if 'SP_TYPE' in result else None
#         main_id = result['MAIN_ID'][0]
#         return sptype, main_id
#     except Exception as e:
#         print(f"Error querying TIC {tic_id}: {e}")
#         return None, None  # Return a tuple with two None values in case of an error

# # Apply the function and unpack the tuple into two columns
# data[['sptype', 'star_name']] = data['TIC_ID'].apply(lambda ticid: pd.Series(get_spectral_type(ticid)))
# data.head()


# # In[10]:


# data[data['sptype'].notna()]


# # In[142]:


# data.sptype.unique()


# # In[128]:


# custom_simbad = Simbad()
# custom_simbad.add_votable_fields('sptype')
# tic = "TIC " + str(270577175)
# result = custom_simbad.query_object(tic)


# # In[132]:


# result['SP_TYPE'][0] if 'SP_TYPE' in result else None


# # ## RA DEC of Candidates

# # In[ ]:


# ra = data['RA'].values * u.degree
# dec = data['DEC'].values * u.degree

# # Convert RA and DEC to SkyCoord object
# sky_coords = SkyCoord(ra=ra, dec=dec, frame='icrs')

# # Wrap RA values at 180 degrees
# wrapped_coords = SkyCoord(ra=sky_coords.ra.wrap_at(180 * u.deg), dec=sky_coords.dec, frame='icrs')

# # Create a scatter plot with Mollweide projection
# plt.figure(figsize=(10, 6))
# plt.subplot(111, projection="mollweide")
# plt.scatter(sky_coords.ra.wrap_at('180d').radian, sky_coords.dec.radian, s=40, linewidth=0.5, marker='o', label='Exocomet Candidates')
# plt.xlabel('RA')
# plt.ylabel('DEC')

# # Add grid in Mollweide projection
# plt.savefig('../figs/RA-DEC.png',dpi=300,bbox_inches=None)
# plt.grid(True)

