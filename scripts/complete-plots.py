import os
import pandas as pd
from analysis_tools_cython import import_lightcurve, processing, mad_cuts
import eleanor
import lightkurve as lk
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from astropy.table import Table
import warnings
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
import astropy.units as u
from astroquery.vizier import Vizier
import sys
from astropy.wcs import WCS
warnings.filterwarnings("ignore")


def get_tpf(tic,sector,h,w):
    tpf = lk.search_tesscut(f'TIC {tic}',sector=sector).download(cutout_size=(h,w))
    return tpf

def add_gaia_elements(df,tic,sector,fig=None,magnitude_limit=19,ax=None,h=5,w=5):
    tpf = get_tpf(tic,sector,h,w)
    c1 = SkyCoord(df.RA,df.DEC,frame='icrs', unit='deg')
    pix_scale = 21.0
    # We are querying with a diameter as the radius, overfilling by 2x.
    from astroquery.vizier import Vizier
    Vizier.ROW_LIMIT = -1
    result = Vizier.query_region(c1, catalog=["I/350/gaiaedr3"],
                                    radius=Angle(np.max(tpf.shape[1:]) * pix_scale, "arcsec"))
    no_targets_found_message = ValueError('Either no sources were found in the query region '
                                            'or Vizier is unavailable')
    too_few_found_message = ValueError('No sources found brighter than {:0.1f}'.format(magnitude_limit))
    if result is None:
        raise no_targets_found_message
    elif len(result) == 0:
        raise too_few_found_message
    result = result["I/350/gaiaedr3"].to_pandas()
    result = result[result.Gmag < magnitude_limit]

    if len(result) == 0:
        raise no_targets_found_message
    radecs = np.vstack([result['RA_ICRS'], result['DE_ICRS']]).T
    coords = tpf.wcs.all_world2pix(radecs, 0)
    try:
        year = ((tpf.time[0].jd - 2457206.375) * u.day).to(u.year)
    except:
        year = ((tpf.astropy_time[0].jd - 2457206.375) * u.day).to(u.year)
    pmra = ((np.nan_to_num(np.asarray(result.pmRA)) * u.milliarcsecond/u.year) * year).to(u.arcsec).value
    pmdec = ((np.nan_to_num(np.asarray(result.pmDE)) * u.milliarcsecond/u.year) * year).to(u.arcsec).value
    result.RA_ICRS += pmra
    result.DE_ICRS += pmdec

    # Gently size the points by their Gaia magnitude
    sizes = 10000.0 / 2**(result['Gmag']/2)

    target = tpf.wcs.world_to_pixel(c1)
    ax.scatter(target[0], target[1], s=50, zorder=1000, c='k', marker='x')
    ax.scatter(coords[:, 0], coords[:, 1], c='firebrick', alpha=0.5, edgecolors='r', s=sizes)
    ax.scatter(coords[:, 0], coords[:, 1], c='None', edgecolors='r', s=sizes)
    ax.set_xlim([0,tpf.shape[1]-1])
    ax.set_ylim([0,tpf.shape[2]-1])

    return fig, ax

df = pd.read_csv('candidates/272-candidates.csv')
data = df[df.tags == 'red'].reset_index(drop=True)
print("read in dataframe.")


os.makedirs('complete-plots', exist_ok=True)

failed_ids = []
for i in tqdm(data.index):
    try:
        #### eleanor-lite
        filepath = data.iloc[i].abs_path
        lc, lc_info = import_lightcurve(filepath)

        #### eleanor-lite tpf
        file = fits.open(filepath)  
        tpf = file[2].data['TPF']
        #aperture = file[2].data['1p25_circle_center']
        aperture = file[2].data.names[0]
        aperture = file[2].data[aperture]  
        tic_id = data.iloc[i].TIC_ID
        sector = data.iloc[i].Sector
        transit_time = data.iloc[i].time

        #### eleanor-api
        star = eleanor.Source(tic=tic_id, sector=int(sector))
        eleanor_data = eleanor.TargetData(star, height=5, width=5, bkg_size=11,do_pca=True)
        q = eleanor_data.quality == 0

        #### qlp
        qlp = lk.search_lightcurve(f"TIC {tic_id}", sector=sector, author='QLP', exptime=1800)

        #### spoc
        spoc = lk.search_lightcurve(f"TIC {tic_id}", sector=sector, author='TESS-SPOC')

        # Check if any data products are available
        if len(qlp) == 0:
            qlp = None
        else:
            qlp = qlp.download()
            qlpq = qlp.quality == 0

        if len(spoc) == 0:
            spoc = None
        else:
            spoc = spoc.download()
            qu = spoc.quality == 0

        ### make a 3x3 matplotlib grid
        fig, ax = plt.subplots(3,3,figsize=(22,13)) # gridspec_kw={'width_ratios':[1,3]}

        #### eleanor-lite

        common_min = np.nanmin(tpf)
        common_max = np.nanmax(tpf)

        ##### TPF of eleanor-lite
        im_eleanorlite = ax[0,0].imshow(tpf,vmin=common_min,vmax=common_max)

        f = lambda x,y: aperture[int(y),int(x) ]
        g = np.vectorize(f)

        x = np.linspace(0,aperture.shape[1], aperture.shape[1]*100)
        y = np.linspace(0,aperture.shape[0], aperture.shape[0]*100)
        X, Y= np.meshgrid(x[:-1],y[:-1])
        Z = g(X[:-1],Y[:-1])

        ax[0,0].contour(Z, [0.05], colors='w', linewidths=3,
                    extent=[0-0.5, x[:-1].max()-0.5,0-0.5, y[:-1].max()-0.5])

        

        #ax[0,0].imshow(aperture)
        ax[0,0].set_title('ELEANOR-LITE TPF (15X15 PIXEL)')
        #add_gaia_elements(data, tic=tic_id, sector=sector,fig=fig, magnitude_limit=18, ax=ax[0, 0],h=15,w=15)

        ax[0,1].plot(lc['TIME'],lc['RAW_FLUX']/np.nanmedian(lc['RAW_FLUX']))
        ax[0,1].axvline(transit_time, linestyle='--', linewidth=2, color='red', zorder=1)
        ax[0,1].set_title('RAW FLUX', fontsize=14)
        ax[0,2].plot(lc['TIME'],lc['PCA_FLUX']/np.nanmedian(lc['PCA_FLUX']))
        ax[0,2].axvline(transit_time, linestyle='--', linewidth=2, color='red', zorder=1) 
        ax[0,2].set_title('PCA FLUX', fontsize=14)


        ## eleanor APIS
        time = eleanor_data.time
        flux = eleanor_data.corr_flux
        quality = eleanor_data.quality

        ### making aperture vis
        eleanor_aperture = eleanor_data.aperture


        f2 = lambda x,y: eleanor_aperture[int(y),int(x) ]
        g2 = np.vectorize(f2)

        x2 = np.linspace(0,eleanor_aperture.shape[1], eleanor_aperture.shape[1]*100)
        y2 = np.linspace(0,eleanor_aperture.shape[0], eleanor_aperture.shape[0]*100)
        X2, Y2= np.meshgrid(x2[:-1],y2[:-1])
        Z2 = g2(X2[:-1],Y2[:-1])

        #add_gaia_figure_elements(RA,DEC,eleanor_data.tpf, ax[1,0], magnitude_limit=18)

        im_eleanorapi = ax[1,0].imshow(eleanor_data.tpf[0],vmin=common_min,vmax=common_max)
        ax[1,0].contour(Z2, [0.05], colors='w', linewidths=3,
                    extent=[0-0.5, x2[:-1].max()-0.5,0-0.5, y2[:-1].max()-0.5])
        ax[1,0].set_title('ELEANOR TPF (5X5 PIXEL)')
        #plt.colorbar(im_eleanorapi,ax=ax[1,0])
        #add_gaia_elements(data, tic=tic_id, sector=sector,fig=fig, magnitude_limit=18, ax=ax[1, 0])

        #sc = ax[1,0].collections[0]
        plt.colorbar(im_eleanorapi,ax=ax[0,0])
        plt.colorbar(im_eleanorapi, ax=ax[1,0])


        table = Table([time, flux, quality], names=['TIME', 'FLUX', 'QUALITY'])
        table = mad_cuts(table,info=lc_info)
        tableq = table['QUALITY'] == 0

        ax[1,1].plot(table['TIME'][tableq], table['FLUX'][tableq]/np.nanmedian(table['FLUX'][tableq]), 'r')
        ax[1,1].axvline(transit_time, linestyle='--', linewidth=2, color='red', zorder=1) 
        ax[1,1].set_title('ELEANOR CORRECTED FLUX', fontsize=14)

        ax[1,2].plot(eleanor_data.time[q], eleanor_data.flux_bkg[q], c='orange', label='1D postcard', linewidth=3)
        ax[1,2].plot(eleanor_data.time[q], eleanor_data.tpf_flux_bkg[q], c='blue',linestyle='--', label='1D TPF', linewidth=3)
        ax[1,2].axvline(transit_time, linestyle='--', linewidth=3, color='k', zorder=1) 
        ax[1,2].plot(lc['TIME'],lc['FLUX_BKG'],linewidth=3,label='ELEANOR-LITE',c='red')
        ax[1,2].set_title('FLUX BKG', fontsize=14)

        ### qlp, spoc and 

        if qlp:
            ax[2,0].plot(qlp.time.value[qlpq], qlp.flux[qlpq]) # sap flux
            ax[2,0].axvline(transit_time, linestyle='--', linewidth=2, color='red', zorder=1)
            ax[2,0].set_title('QLP (SAP) FLUX', fontsize=14)

        if spoc:
            ax[2,1].plot(spoc.time.value[qu], spoc.pdcsap_flux[qu]/np.nanmedian(spoc.pdcsap_flux[qu]))
            ax[2,1].axvline(transit_time, linestyle='--', linewidth=2, color='red', zorder=1)
            ax[2,1].set_title('SPOC PDCSAP', fontsize=14)

            ax[2,2].plot(spoc.time.value[qu], spoc.sap_flux[qu]/np.nanmedian(spoc.pdcsap_flux[qu]))
            ax[2,2].axvline(transit_time, linestyle='--', linewidth=2, color='red', zorder=1)
            ax[2,2].set_title('SPOC SAP', fontsize=14)


        fig.legend(loc='center right')
        plt.subplots_adjust(hspace=0.5)
        fig.suptitle(f'TIC {tic_id}, SECTOR {sector}', fontsize=18)

        # Construct the filename
        filename = f'complete-plots/TIC{tic_id}.png'
        
        # Check if the file already exists
        counter = 1
        while os.path.exists(filename):
            # If the file exists, increment the counter and modify the filename
            counter += 1
            filename = f'complete-plots/TIC{tic_id}_{counter}.png'

        # Save the figure with the updated filename
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()  # Close the current figure to avoid overlapping in the next iteration
    
    except OSError:
        failed_ids.append(tic_id)
        pass
    except AttributeError:
        print(f"No data found for target {data.iloc[i].TIC_ID}. Only printing available plots")
        
    except IndexError:
        print(f"Did not work for {tic_id} in sector {sector} due to an index error.")
        failed_ids.append(tic_id)
        pass
    except eleanor.utils.SearchError:
        print(f"No data found for target {tic_id}.")
        failed_ids.append(tic_id)
        pass


print("plots complete.")
print(f"TIC IDs that failed: {failed_ids}")

continue_or_not = input('exit? (y or n): ')
if continue_or_not == 'y':
    sys.exit()

print("Now trying a manual extraction from lightkurve for failed TIC IDs")
for i in failed_ids:

    filepath = data[data.TIC_ID == i].abs_path.values[0]
    lc, lc_info = import_lightcurve(filepath)

    tic_id = i
    sector = int(data[data.TIC_ID == i].Sector.values[0])
    transit_time = data[data.TIC_ID == i].time.values[0]

    #### eleanor-lite tpf
    file = fits.open(filepath)  
    tpf = file[2].data['TPF']
    aperture = file[2].data.names[0]
    aperture = file[2].data[aperture]  

    common_min = np.nanmin(tpf)
    common_max = np.nanmax(tpf)

    fig, ax = plt.subplots(2,3,figsize=(22,13)) # gridspec_kw={'width_ratios':[1,3]}
    ##### TPF of eleanor-lite
    im_eleanorlite = ax[0,0].imshow(tpf,vmin=common_min,vmax=common_max)

    f = lambda x,y: aperture[int(y),int(x)]
    g = np.vectorize(f)

    x = np.linspace(0,aperture.shape[1], aperture.shape[1]*100)
    y = np.linspace(0,aperture.shape[0], aperture.shape[0]*100)
    X, Y= np.meshgrid(x[:-1],y[:-1])
    Z = g(X[:-1],Y[:-1])

    ax[0,0].contour(Z, [0.05], colors='w', linewidths=3,
                extent=[0-0.5, x[:-1].max()-0.5,0-0.5, y[:-1].max()-0.5])

    plt.colorbar(im_eleanorlite,ax=ax[0,0])
    ax[0,0].set_title('ELEANOR-LITE TPF (15X15 PIXEL)')
    
    #add_gaia_elements(data, tic=tic_id, sector=sector,fig=fig, magnitude_limit=18, ax=ax[0, 0])

    ax[0,1].plot(lc['TIME'],lc['RAW_FLUX']/np.nanmedian(lc['RAW_FLUX']))
    ax[0,1].axvline(transit_time, linestyle='--', linewidth=2, color='red', zorder=1)
    ax[0,1].set_title('RAW FLUX', fontsize=14)
    ax[0,2].plot(lc['TIME'],lc['PCA_FLUX']/np.nanmedian(lc['PCA_FLUX']))
    ax[0,2].axvline(transit_time, linestyle='--', linewidth=2, color='red', zorder=1) 
    ax[0,2].set_title('PCA FLUX', fontsize=14)

    
    ### manual reconstruction
    try:
        star = eleanor.Source(tic=tic_id, sector=sector,tc=True)
        data = eleanor.TargetData(star, height=5, width=5, bkg_size=11)
        tpf = data.tpf[0]

        ### making aperture vis
        eleanor_aperture = eleanor_data.aperture


        f2 = lambda x,y: eleanor_aperture[int(y),int(x) ]
        g2 = np.vectorize(f2)

        x2 = np.linspace(0,eleanor_aperture.shape[1], eleanor_aperture.shape[1]*100)
        y2 = np.linspace(0,eleanor_aperture.shape[0], eleanor_aperture.shape[0]*100)
        X2, Y2= np.meshgrid(x2[:-1],y2[:-1])
        Z2 = g2(X2[:-1],Y2[:-1])

        im_eleanorapi = ax[1,0].imshow(eleanor_data.tpf[0],vmin=common_min,vmax=common_max)
        ax[1,0].contour(Z2, [0.05], colors='w', linewidths=3,
                    extent=[0-0.5, x2[:-1].max()-0.5,0-0.5, y2[:-1].max()-0.5])
        ax[1,0].set_title('ELEANOR TPF (5X5 PIXEL)')
        #plt.colorbar(im_eleanorapi,ax=ax[1,0])


    except:
        print(f"TIC {tic_id} not found with eleanor. Reconstructing with lightkurve.")
        tpf = lk.search_tesscut(f'TIC {tic_id}',sector=sector).download_all(cutout_size=(15,15))
        tpf = tpf[0]
        custom_mask = tpf[0].create_threshold_mask(threshold=3)
        tpf.plot(aperture_mask=custom_mask,ax=ax[1,0])
        lkcurve = tpf.to_lightcurve(aperture_mask=custom_mask)
        lkcurve.plot(ax=ax[1,1],normalize=True)

        regressors = tpf.flux[:, ~custom_mask]
        bkg = np.median(regressors, axis=1)
        bkg -= np.percentile(bkg, 5)

        npix = custom_mask.sum()
        median_subtracted_lc = lkcurve - npix * bkg
        median_subtracted_lc.plot(ax=ax[1,2], normalize=True, label="Median Subtracted")
        median_subtracted_lc.flatten().plot(ax=ax[1,2], normalize=True, label="+Flattened")

    #add_gaia_elements(data, tic=tic_id, sector=sector,fig=fig, magnitude_limit=18, ax=ax[1, 0])


        fig.legend(loc='center right')
        plt.subplots_adjust(hspace=0.5)
        fig.suptitle(f'TIC {tic_id}, SECTOR {sector}', fontsize=18)

            # Construct the filename
        filename = f'complete-plots/TIC{tic_id}.png'
        
        # Check if the file already exists
        counter = 1
        while os.path.exists(filename):
            # If the file exists, increment the counter and modify the filename
            counter += 1
            filename = f'complete-plots/TIC{tic_id}_{counter}.png'

        # Save the figure with the updated filename
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()  #

print("plots complete.")