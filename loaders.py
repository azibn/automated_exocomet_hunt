import lightkurve as lk
from multiprocessing import Pool
import numpy as np
import pickle
import pandas as pd
import os 
import data


class LightCurveCollection:
    def __init__(
            self,
            sec: list,
            cam: list,
            files: list,
            factor: int = 10,
    ):
        self.sector = sec
        self.camera = cam
        self.files = files
        self.ref = pd.DataFrame(
            data={"sector": sec, "camera": cam, "file": files}
            )
        self.factor = factor
        self.useCpus = 1
        self.mad_table = data.load_mad()
        self.__load_scmad()
        self.__calc_cutmask()
        self.__calc_log_weight()
        self.loaded = False
        self.__make_summary()
        return

    def __make_summary(self):
        self.summary = {
            "Sector": self.sector,
            "Camera": self.camera,
            "Count": len(self.files),
            "Loaded": self.loaded
        }
        return self.summary

    def __load_scmad(self):
        sc_mad = dict()
        if "mad_table" not in locals():
            mad_table = self.mad_table
        sectors = np.unique(self.sector)
        cams = np.unique(self.camera)
        for sector in sectors:
            for cam in cams:
                subref = self.ref[(self.ref.sector == sector)
                                  & (self.ref.camera == cam)]
                if len(subref) == 0:
                    pass
                else:
                    f0 = subref.file.values[0]
                    lc0 = load_lc(f0)
                    sc_mad[f'{sector}-{cam}'] = mad_table.loc[
                        :len(lc0)-1,
                        f"{sector}-{cam}"
                    ]
        self.sc_mad = sc_mad
        return sc_mad

    def __calc_cutmask(self):
        self.cutmask = dict()

        sectors = np.unique(self.sector)
        cams = np.unique(self.camera)
        for sector in sectors:
            for cam in cams:
                subref = self.ref[(self.ref.sector == sector)
                                  & (self.ref.camera == cam)]
                if len(subref) == 0:
                    pass
                else:
                    sc = f'{sector}-{cam}'
                    cut = (np.mean(self.sc_mad[sc])
                           + (self.factor
                              * np.std(self.sc_mad[sc]))
                           )
                    self.cutmask[sc] = self.sc_mad[sc] < cut
        return self.cutmask

    def __calc_log_weight(self):
        # lookup MAD values for appropriate sector/camera combination
        self.mad_scaler = dict()

        sectors = np.unique(self.sector)
        cams = np.unique(self.camera)
        for sector in sectors:
            for cam in cams:
                subref = self.ref[(self.ref.sector == sector)
                                  & (self.ref.camera == cam)]
                if len(subref) == 0:
                    pass
                else:
                    sc = f'{sector}-{cam}'
                    # create the scaling array, log scaled in this instance,
                    # inverted so large MAD cadences are suppressed
                    sc_mad_loginv = -np.log(self.sc_mad[sc])
                    self.mad_scaler[sc] = np.array(
                        (sc_mad_loginv-sc_mad_loginv.min())
                        / (sc_mad_loginv.max() - sc_mad_loginv.min())
                    )

        return self.mad_scaler

    def load_cut_lc(self, miniref):
        """Load a masked light curve.
        args:
            miniref - any python object with 'file', 'sector', and 'camera' attributes
        """
        f = miniref.file
        sec = miniref.sector
        cam = miniref.camera

        # ! Not masked initially so that it lcs have consistent length
        # ! with the cutmask from the MAD array
        lc = load_lc(f)
        sc = f"{sec}-{cam}"
        # all the masks
        lc = lc[self.cutmask[sc].values & (lc.quality == 0)].remove_nans()
        if np.nanmedian(lc.flux) < 0:
            # A negative median results from bad background subtraction
            # Normalizing (w/o inverting) makes the median -1, so we translate
            # the light curve up to make the median positive.
            lc.flux += 2*abs(np.nanmedian(lc.flux))

        nfluxes = np.array(lc.flux/abs(np.nanmedian(lc.flux)))
        lc.flux = nfluxes
        lc_copy = lc[lc.flux >= 0]  # one final mask

        assert len(lc_copy) > 100, f"Check TIC {lc_copy.targetid}"

        return lc_copy.remove_outliers()

    def load_weighted_lc(self, miniref):
        f = miniref.file
        sec = miniref.sector
        cam = miniref.camera

        # import all light curves for the sector/camera combination
        lc = load_lc(f)
        sc = f"{sec}-{cam}"
        nfluxes = np.array(lc.flux/np.nanmedian(lc.flux))
        nfluxes = (nfluxes-1)*self.mad_scaler[sc]+1  # scaled fluxes
        lc.flux = nfluxes
        mask = (lc.quality == 0) & (lc.flux >= 0)
        lc = lc[mask].remove_nans()
        return lc

    def load_all_lcs(self, method: str = 'cut'):
        """Load all light curves specified in self.files
        """
        if self.useCpus == 1:
            if method == "cut":
                self.lcs = [self.load_cut_lc(self.ref.iloc[i])
                            for i in range(len(self.ref))]
            elif method == "log_w":
                self.lcs = [self.load_weighted_lc(self.ref.iloc[i])
                            for i in range(len(self.ref))]
        else:
            # todo: update for weighted
            with Pool(self.useCpus) as p:
                minirefs = [self.ref.iloc[i] for i in range(len(self.ref))]
                self.lcs = p.map(self.load_cut_lc, minirefs, chunksize=500)
        # with Pool(useCpus) as p:
        #    lcs = p.map(load_weighted_lc, files, chunksize=100)
        self.loaded = True
        return


def load_lc(fp, fluxtype="PDC", mask=False):
    """Load light curve data from pickle file into a lightkurve object
    Args:
        fp (str) - file path to pickle file in standard format
        fluxtype (str) - Type of flux to prioritize,
            choose between "raw", "corr", and "PDC"
        mask (bool) - Mask data points non-zero flags in quality

    returns:
        lc (lightkurve.lightcurve.LightCurve) - a LightCurve object
    """

    with open(fp, 'rb') as file:
        lc_list = pickle.load(file)

    fluxes = {"raw": lc_list[7], "corr": lc_list[8], "PDC": lc_list[9]}

    try:
        flux = fluxes[fluxtype]

    except KeyError:
        print("""
        The flux type must be 'raw', 'corr', or 'PDC'. Defaulting to 'PDC'.""")
        flux = fluxes["PDC"]

    finally:
        time = lc_list[6]
        flux_err = lc_list[10]
        quality = lc_list[11]

        if mask:
            mask = lc_list[11] == 0
            flux = flux[mask]
            time = time[mask]
            flux_err = flux_err[mask]
            quality = quality[mask]  # just 0's if masked

        # for meta information
        fluxes.update(
            {"TESS Magnitude": lc_list[3], "filename": fp.split("/")[-1]})
        lc = lk.lightcurve.TessLightCurve(
            time=time, flux=flux, flux_err=flux_err, targetid=lc_list[0],
            quality=quality, camera=lc_list[4], ccd=lc_list[5],
            ra=lc_list[1], dec=lc_list[2], label=f"TIC {lc_list[0]}",
            meta=fluxes
        )

    return lc


# old name updated to PEP8 class convention
lc_collection = LightCurveCollection


def load_masked_lc(fp):
    """ Loads a light curve with bad quality data masked
    non-zero quality flags from eleanor masked as well as
    cadences determined to have signficant systematics
    """
    lc = load_lc(fp)
    comb_mask = [(lc.quality[i] == 0) and (threshold_mask[i] == 0)
                 for i in range(len(lc))]
    lc = lc[comb_mask]
    return lc


def load_masked_lc_wrapper(fps, sec, cam):
    """ A wrapper for the load_masked_lc method
    Created to avoid loading the threshold_mask file for each
    lightcurve import
    """
    # hardcoded references to files for use by this work.
    mad_mask = "./data/threshold_mask.json"
    # This is bad... but it works...
    global threshold_mask
    with open(mad_mask, "r") as file:
        threshold_mask = pd.read_json(file)[f"{sec}-{cam}"]
    with Pool(4) as p:
        lcs = p.map(load_masked_lc, fps)
    # lcs = [load_masked_lc(fp, threshold_mask) for fp in fps]
    return lcs


def load_lc_arr(fp, fluxtype="PDC", mask=False):
    """Load light curve data from pickle file into a flux array
    Args:
        fp (str) - file path to pickle file in standard format
        fluxtype (str) - Type of flux to prioritize,
            choose between "raw", "corr", and "PDC"
        mask (bool) - Mask data points non-zero flags in quality

    returns:
        (time, flux, quality, lc_list[0]) (tuple)
        time (array)
        flux (array)
        quality (array)
        lc_list[0] (int)
    """
    with open(fp, 'rb') as file:
        lc_list = pickle.load(file)
    fluxes = {"raw": lc_list[7], "corr": lc_list[8], "PDC": lc_list[9]}
    try:
        flux = fluxes[fluxtype]
    except KeyError:
        print("""
        The flux type must be 'raw', 'corr', or 'PDC'. Defaulting to 'PDC'.""")
        flux = fluxes["PDC"]
    finally:
        time = lc_list[6]
        flux_err = lc_list[10]
        quality = lc_list[11]
        if mask:
            mask = lc_list[11] == 0
            flux = flux[mask]
            time = time[mask]
            flux_err = flux_err[mask]
            quality = quality[mask]  # just 0's if masked

    return (time, flux, quality, lc_list[0])


def make_subset(data_dir,
                sector=1,
                camera=1,
                ccd=1,
                mag_min=0,
                mag_max=15,
                save_dir="/home/dgiles1/data/"
                ):
    ref = pd.read_csv(data_dir+f"sector{sector}lookup.csv")
    lc_set = ref[(ref.Camera == camera) & (ref.CCD == ccd)]
    lc_set = lc_set[(lc_set.Magnitude < mag_max) &
                    (lc_set.Magnitude > mag_min)]

    # =========================================
    # WRITE A FILE LIST, SAVE LOCALLY
    # =========================================
    fp = f'./data/{sector}_{camera}_{ccd}-{mag_min}_{mag_max}.txt'
    with open(fp, 'w') as file:

        for filename in lc_set.Filename:
            file.write(f"gs://tess-goddard-lcs/{filename} \n")
        cat_cmd = f"cat ./data/{sector}_{camera}_{ccd}-{mag_min}_{mag_max}.txt"
        gs_cmd = f"gsutil -m cp -I {save_dir}"
        os.system(f"{cat_cmd}|{gs_cmd}")
        if __name__ != "__main__":
            print(f"""
                  Run the following in the terminal to download data locally:
                  {cat_cmd} | {gs_cmd}
                  """)

    return lc_set


def make_scd_subset(
        data_dir,
        sector=1,
        camera=1,
        ccd=1,
        mag_min=0,
        mag_max=15,
        save_dir="/home/dgiles1/data/"
):
    try:
        ref = pd.read_csv(data_dir+f"sector{sector}lookup.csv")
    except FileNotFoundError:
        print(f"""
        FileNotFoundError: the file {data_dir}sector{sector}lookup.csv
        cannot be found. Make sure the Google Bucket is properly mounted.

        Try running:
        gcsfuse --implicit-dirs tess-goddard-lcs /home/[USER]/[MOUNT POINT]/

        and don't forget to provide the --data-path argument
            when rerunning this script:
        python lc_mad --data-path /home/[USER]/[MOUNT POINT]/
        """)
    lc_set = ref[(ref.Camera == camera) & (ref.CCD == ccd)]
    lc_set = lc_set[(lc_set.Magnitude < mag_max) &
                    (lc_set.Magnitude > mag_min)]

    # =========================================
    # WRITE A FILE LIST, SAVE LOCALLY
    # =========================================
    try:
        # check if files are downloaded
        # TODO: replace with a more straightforward "does file exist?" test
        load_lc(save_dir+lc_set.iloc[-1].Filename, mask=False)

    except FileNotFoundError:
        lines = [
            f"gs://tess-goddard-lcs/{filename}\n"
            for filename in lc_set.Filename
            ]
        files = [file.split('/')[-1] for file in lc_set.Filename]
        fl1 = f'/home/dgiles1/data/{sector}_{camera}_{ccd}'
        fl2 = f'{int(mag_min)}_{int(mag_max)}'
        filelist = f'{fl1}-{fl2}.txt'
        files_done = os.listdir(save_dir)
        done = [elem in files_done for elem in files]
        with open(filelist, 'w') as file:
            file.writelines(lines[not done])

        if __name__ == "__main__":
            os.system(f"cat {filelist} | gsutil -m cp -I {save_dir}")
            print(f"Data saved to {save_dir}, remember to clean up.")
            pass
        else:
            print("Make sure data is downloaded or accessible.\n")
            print("You could use:\n")
            print(f"cat {filelist} | gsutil -m cp -I {save_dir}")

    return lc_set


def load_scd_subset(
    sector,
    camera,
    ccd,
    mag_min,
    mag_max,
    data_path,
    save_path
):

    lc_set = make_scd_subset(data_path, sector, camera,
                             ccd, mag_min, mag_max, save_path)
    lc_set_filenames = [save_path+f.split('/')[-1] for f in lc_set.Filename]
    return lc_set_filenames


def make_tmc_subset(
        data_dir,
        sector=1,
        camera=1,
        save_dir="/home/dgiles1/data/"
):
    """ Create the subset and download the appropriate data
    """
    try:
        ref = pd.read_csv(data_dir+f"sector{sector}lookup.csv")
    except FileNotFoundError:
        print(f"""
        FileNotFoundError: the file {data_dir}sector{sector}lookup.csv
        cannot be found. Make sure the Google Bucket is properly mounted.

        Try running:
        gcsfuse --implicit-dirs tess-goddard-lcs /home/[USER]/[MOUNT POINT]/

        and don't forget to provide the --data-path argument
            when rerunning this script:
        python lc_mad --data-path /home/[USER]/[MOUNT POINT]/
        """)
    tmcl = ["2_min_cadence" in fn for fn in ref.Filename]
    lc_set = ref[(ref.Camera == camera) & tmcl]

    # =========================================
    # WRITE A FILE LIST, SAVE LOCALLY
    # =========================================
    try:
        # check if files are downloaded
        # TODO: replace with a more straightforward "does file exist?" test
        load_lc(save_dir+lc_set.iloc[-1].Filename, mask=False)

    except FileNotFoundError:
        lines = [
            f"gs://tess-goddard-lcs/{filename}\n"
            for filename in lc_set.Filename
            ]
        files = [file.split('/')[-1] for file in lc_set.Filename]
        filelist = f'/home/dgiles1/data/TMC{sector}-{camera}.txt'
        files_done = os.listdir(save_dir)
        done = [elem in files_done for elem in files]
        with open(filelist, 'w') as file:
            file.writelines(lines[not done])

        if __name__ == "__main__":
            os.system(f"cat {filelist} | gsutil -m cp -I {save_dir}")
            print(f"Data saved to {save_dir}, remember to clean up.")
            pass
        else:
            print("Make sure data is downloaded or accessible.\n")
            print("You could use:\n")
            print(f"cat {filelist} | gsutil -m cp -I {save_dir}")

    return lc_set


def load_tmc_subset(sector, camera, data_path, save_path):
    """Load a set of filenames as a list, downloading the files if necessary.
    """
    lc_set = make_tmc_subset(data_path, sector, camera, save_path)
    lc_set_filenames = [save_path+f.split('/')[-1] for f in lc_set.Filename]
    return lc_set_filenames


def load_subset(data_path,
                lc_set_filenames,
                mask=False,
                simple=False,
                ncpus=1):
    """Loads light curves matching given criteria from a single SCD combo.
    Args:
        ref_csv (pandas.DataFrame) - sector reference from csv
        lc_set_filenames (list) - array of light curve pickle file names
            If there is a directory substructure,
            these names should include that structure.
        mask (bool) - whether to mask bad quality data
        simple (bool) - if true returns a tuple for each light curve,
            if false returns as lightkurve.TessLightCurve
        ncpus (int) - number of CPUs to use for import
    Returns:
        lcs (list) - each entry containing time, flux, quality, and TIC ID
    """

    # =========================================
    # LOADING THE SUBSET OF LIGHT CURVES
    # =========================================

    # this should be easily parallelized (CPU not GPU)
    # not worth it on 2 cpus with 704 lcs, but probably worth it
    # for 10k or 100k and more CPUs.
    if ncpus != 1:
        from multiprocessing import Pool
        paths = [data_path+f for f in lc_set_filenames]
        if ncpus == -1:
            with Pool() as p:
                lcs = p.map(load_lc_arr, paths)
        else:
            with Pool(ncpus) as p:
                lcs = p.map(load_lc_arr, paths)
    else:
        lcs = []
        for f in lc_set_filenames:
            if simple:
                time, flux, quality, lc_list = load_lc_arr(
                    data_path+f, mask=mask)
                lcs.append([time, flux, lc_list])
            else:
                lcs.append(load_lc(data_path+f, mask=False))

    return lcs


def load_ref(sector, data_dir="~/data/"):
    # Define where to mount the data,
    # I would suggest creating an empty directory
    # off of your home directory.
    #  data_dir = "/home/dgiles1/mountpoint/"  # Remote machine on GCP
    #  data_dir = "/home/dgiles/Documents/tesslcs/" # Machine at Adler

    try:
        # check if the lookup file for sector one exists
        ref = pd.read_csv(data_dir+f"sector{sector}lookup.csv")
    except FileNotFoundError:
        # If not, try to mount it.
        # ! only works on a VM
        import os
        os.system(
            f"gcsfuse --implicit-dirs tess-goddard-lcs {data_dir}"
        )
        ref = pd.read_csv(data_dir+f"sector{sector}lookup.csv")
    return ref


def tmc_subset(ref):
    # optional, the 2-minute cadence targets are an easy subset to pull from
    tmcl = ["2_min_cadence" in fn for fn in ref.Filename]
    tmc = ref[tmcl]
    return tmc


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sector", type=int, default="1", help="Sector number",
    )

    parser.add_argument(
        "--camera", type=int, default="1", help="Camera number",
    )

    parser.add_argument(
        "--ccd", type=int, default="1", help="CCD detector number",
    )
    parser.add_argument(
        "--mag-min", type=float, default="1",
        help="magnitude minimum (lowest value)",
    )

    parser.add_argument(
        "--mag-max", type=float, default="15",
        help="magnitude maximum (highest value)",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="/home/dgiles1/mountpoint/",
        help="Directory to locally access light curve data from",
    )

    parser.add_argument(
        "--save-path",
        type=str,
        default="/home/dgiles1/data/",
        help="Directory to locally save light curve data to",
    )

    args = parser.parse_args()

    make_subset(args.data_path,
                args.sector,
                args.camera,
                args.ccd,
                args.mag_min,
                args.mag_max,
                args.save_path
                )
