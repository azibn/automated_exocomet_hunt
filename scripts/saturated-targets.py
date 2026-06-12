import lightkurve as lk
import os
import pandas as pd
import time
import multiprocessing
import numpy as np

os.makedirs("saturated-targets-data/", exist_ok=True)


def download_lightcurve(tic_id, sector):
    """
    Function: Downloads lightcurve from MAST.

    Args:
        tic_id (str): TIC ID.
        sector (int): Sector number.

    Returns:
        str: Path to FITS file.
    """
    try:
        print(f"Downloading TIC {tic_id} from MAST")
        tpf = lk.search_tesscut(f"TIC {tic_id}", sector=sector).download_all(cutout_size=(15,15))
        tpf = tpf[0]
        custom_mask = tpf[0].create_threshold_mask(threshold=3)

        raw_lc = tpf.to_lightcurve()
        lc = tpf.to_lightcurve(aperture_mask=custom_mask).fill_gaps()

        regressors = tpf.flux[:, ~custom_mask]
        regressors.shape
        bkg = np.median(regressors, axis=1)
        bkg -= np.percentile(bkg, 5)

        npix = custom_mask.sum()
        median_subtracted_lc = lc - npix * bkg


        median_subtracted_lc.to_fits(f"saturated-targets-data/TIC_{tic_id}_s{sector}_lc.fits")
        return f"TIC_{tic_id}_s{sector}_lc.fits"
    except ValueError as e:
        print(f"Error for {tic_id}: {e}")
        pass


def worker(tic_id, sector):
    for _ in range(5):  # Retry up to 5 times
        try:
            download_lightcurve(tic_id, sector)
            break  # If successful, break the retry loop
        except:
            time.sleep(5)  # Wait for 5 seconds before retrying

if __name__ == "__main__":
    data = pd.read_csv('saturated-targets.csv')

    tic_ids = data['TIC_ID'].values.astype(int)
    sectors = data['Sector'].values.astype(int)

for tic_id, sector in zip(tic_ids, sectors):
    download_lightcurve(tic_id, sector=sector)



    # with multiprocessing.Pool(processes=35) as pool:
    #     pool.starmap(worker, zip(tic_ids, sectors))