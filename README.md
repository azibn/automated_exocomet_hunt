# Automated Exocomet Hunt

Code for automated detection of exocomets in light curves.

**Note: This work has been mainly developed with focus on the internal lightcurves in the collaboration. However, the scripts should still be compatible with lightcurves obtained via MAST or `lightkurve`.**

## Repository structure

```
.
├── scripts/            # Main pipeline and analysis scripts
│   ├── analysis_tools_cython.pyx   # Core search algorithms (Cython)
│   ├── batch_analyse.py            # Main entry point for running the search
│   ├── som/                        # Self-organising map tools
│   └── xrpdata/                    # Package data (bad times, MAD tables, masks)
├── notebooks/          # Analysis and exploration notebooks
├── data/               # Local light curve data and catalogues (not tracked)
├── outputs/            # Search results (not tracked)
├── plots/              # Generated figures (not tracked)
├── archive/            # Old material, posters (not tracked)
├── environment.yml     # Conda environment
├── setup.py            # Builds the Cython extension
└── make                # Shortcut for `./setup.py build_ext --inplace`
```

## Installation

```
git clone https://github.com/azibn/automated_exocomet_hunt
conda env create -f environment.yml
conda activate auto_exo
./make
```

Alternatively:

```
git clone https://github.com/azibn/automated_exocomet_hunt
conda create -n <environment name> python jupyter jupyterlab scipy astropy numpy pandas pip cython matplotlib
conda activate <environment name>
pip install lightkurve kplr eleanor
./make
```

Different package versions may cause conflicts, so it is recommended to run this code using the virtual environment setup above.

Tip: [mamba](https://mamba.readthedocs.io/) is a faster drop-in replacement for conda — `mamba env create -f environment.yml` works too.

## Usage

### `batch_analyse.py`

`batch_analyse.py` is the main script of this project, and can run on a single file, a directory of files, or an entire sector. Results are output to a text file with one row per light curve.

Our analysis uses an earlier iteration of the `GSFC-ELEANOR-LITE` lightcurves, a lightweight version of the [Eleanor](https://ui.adsabs.harvard.edu/abs/2019PASP..131i4502F/abstract) lightcurves stored locally (more information [here](https://archive.stsci.edu/hlsp/gsfc-eleanor-lite)). The pipeline also works with [SPOC](https://ui.adsabs.harvard.edu/abs/2020RNAAS...4..201C/abstract) and [QLP](https://ui.adsabs.harvard.edu/abs/2020RNAAS...4..204H/abstract) lightcurves downloaded from MAST, with work progressing on [TASOC](https://ui.adsabs.harvard.edu/abs/2019AAS...23320207B/abstract) lightcurves. Kepler lightcurves can be obtained from [MAST](https://archive.stsci.edu/kepler/).

On a `.pkl` lightcurve:

    python scripts/batch_analyse.py /storage/.../tesslcs_sector_6_104/2_min_cadence_targets/tesslc_270577175.pkl

On a `.fits` lightcurve:

    python scripts/batch_analyse.py hlsp_tess-spoc_tess_phot_0000000270577175-s0006_tess_v1_lc.fits

The script has multiple arguments (number of threads, output file location, smoothing method from `wotan`, etc.), some of which are mandatory. Run with `-h` for details.

### Injection testing

`scripts/injection_testing.py` runs an injection test on a user-specified (default 100000) number of lightcurves between a magnitude range. The depths of the injected comets are random.

**Note: This is currently only for the `.pkl` files. It is not yet compatible with other file types.**

### Integration with `lightkurve`

The functions work with lightcurves obtained from the `lightkurve` package. However, the main search function, `processing`, requires the data in the format of `time`, `flux`, `quality`, `flux error` in either an `astropy.Table` or `pandas.DataFrame`, so make sure to convert to this format before processing.

## Code style

Code style in `.py` scripts is formatted with the [Black Python Formatter](https://black.readthedocs.io/en/stable/index.html) and must be standardised with Black before pushing to the repository. Black formatting checks run as part of the Git workflow.

    black <name_of_script>.py
