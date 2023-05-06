**Note: This work has been mainly developed with focus on the internal lightcurves in the collaboration. However, they should still be compatible with lightcurves obtained via MAST or `lightkurve`.**

Code for automated detection of exocomets in light curves.

# Installation

Latest tested in Python 3.7
	
	git clone https://github.com/azibn/automated_exocomet_hunt
	conda env create -f environment.yml
	conda activate auto_exo

Alternatively, you can try this:

	git clone https://github.com/azibn/automated_exocomet_hunt
	conda create -n <environment name> python jupyter jupyterlab scipy astropy numpy pandas pip cython matplotlib
	conda activate <environment name>
	pip install lightkurve kplr eleanor
	./make
 
Different package versions may cause conflicts, so it is recommended to run this code using the virtual environment setup placed above. 

### Don't forget about mamba!

Mamba is the conda package manager reimplemented in C++ for faster outputs. One can do

```conda install mamba -n base -c conda-forge```

to the base environment and then 

```mamba env create -f environment.yml```

`conda` commands will also work with mamba, eg: `conda activate <env>`.

# Usage

## `batch_analyse.py`

These scripts can run on TESS and Kepler lightcurves. Our analysis uses an earlier iteration of the `GSFC-ELEANOR-LITE` lightcurves, a lightweight version of the [Eleanor](https://ui.adsabs.harvard.edu/abs/2019PASP..131i4502F/abstract) lightcurves that has been stored locally. More information on the `GSFC-ELEANOR-LITE` lightcurves can be found [here](https://archive.stsci.edu/hlsp/gsfc-eleanor-lite). The pipeline also works with [SPOC](https://ui.adsabs.harvard.edu/abs/2020RNAAS...4..201C/abstract) and [QLP](https://ui.adsabs.harvard.edu/abs/2020RNAAS...4..204H/abstract) lightcurves downloaded from MAST. Work currently progressing on adapting with [TASOC](https://ui.adsabs.harvard.edu/abs/2019AAS...23320207B/abstract) lightcurves.

Kepler lightcurves can be obtained from [MAST](https://archive.stsci.edu/kepler/). 

`batch_analyse` is the main file for this project, and can run on a single file, a directory of files, or the entire sector. Results will be output to a text file with one row per file. For example, with a `.pkl` lightcurve you can run:

    python batch_analyse.py /storage/.../tesslcs_sector_6_104/2_min_cadence_targets/tesslc_270577175.pkl

For `.fits` format lightcurves for example, one can run:

    python batch_analyse.py hlsp_tess-spoc_tess_phot_0000000270577175-s0006_tess_v1_lc.fits
    
The script has multiple arguments that you can call (number of threads, output file location, smoothing method from `wotan` etc), where some are mandatory. For more information on these flags, run the script with `-h`. 

## `injection_testing`

This script runs an injection test on a user-specified (default 100000) number of lightcurves between a magnitude range. The depths of the injected comets are random. 

**Note: This is currently only for the `.pkl` files. I have not yet made it compatible with other file types.**

## Integration with `lightkurve`

The functions should work with lightcurves obtained from the `lightkurve` package. However, the main function of the search, `processing`, requires the data to be in the format of `time`, `flux`, `quality`, `flux error` in either a `astropy.Table` or `pandas.DataFrame` format right now, so make sure to have this format if you are looking to process the lightcurve.

# Code Style
Code style in `.py` scripts is formatted with [Black Python Formatter](https://black.readthedocs.io/en/stable/index.html) and must be standardised with Black before pushing to the repository. Black formatting checks run as part of the Git workflow. It is responsibility of the user to format their code before pushing to repo.

To format your files, enter shell and run:

`black <name_of_script>.py`

or

`black *`

to format contents of the entire directory.

# Docker

Progress on this pipeline has been made to integrate it with Docker if the setup fails for whatever reason, or if you are just a Docker enthusiast. If you are not familiar with Docker, [this page](https://www.docker.com/resources/what-container) is a good starting point. But you don't really need to know how it works to use it! This is still in development, as we're still working out the system requirements etc, so if there are any issues do get in touch. 

### Requirements
- Docker Desktop 
- 64-bit CPU
- 8GB RAM
