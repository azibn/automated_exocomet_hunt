Code for automated detection of exocomets in light curves.

### Installation

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

### Usage

These scripts currently run on TESS and Kepler lightcurves. Our analysis uses [Eleanor](https://ui.adsabs.harvard.edu/abs/2019PASP..131i4502F/abstract) lightcurves from the SETI collaboration. Pipeline also works with [SPOC](https://ui.adsabs.harvard.edu/abs/2020RNAAS...4..201C/abstract) and [QLP](https://ui.adsabs.harvard.edu/abs/2020RNAAS...4..204H/abstract) pipelines. Work currently progressing on adapting with [TASOC](https://ui.adsabs.harvard.edu/abs/2019AAS...23320207B/abstract) lightcurves

Kepler lightcurves can be obtained from [MAST](https://archive.stsci.edu/kepler/). 

TESS SPOC lightcurves can also be obtained from [MAST](https://archive.stsci.edu/missions/tess/tid/).

`single_analysis.py` runs on a single file, for example:
 
 TESS:

    python single_analysis_xrp.py tesslcs_sector_6_104_2_min_cadence_targets_tesslc_270577175.pkl
 
 Kepler:

    wget https://archive.stsci.edu/missions/kepler/lightcurves/0035/003542116/kplr003542116-2012088054726_llc.fits
    python single_analysis.py kplr003542116-2012088054726_llc.fits


`batch_analyse.py` runs on directories of files, outputting results to a text file with one row per file. `archive_analyse.sh` is a bash script for processing compressed archives of light curve files, extracting them temporarily to a directory.  Both these scripts have multiple options (number of threads, output file location ...), run with help flag (`-h`) for more details.

### Code Style
Code style in `.py` scripts is formatted with [Black Python Formatter](https://black.readthedocs.io/en/stable/index.html) and must be standardised with Black before pushing to the repository. Black formatting checks run as part of the Git workflow. It is responsibility of the user to format their code before pushing to repo.

To format your files, enter shell and run:

`black scripts/<name_of_script>.py`

or

`black scripts/*`

to format contents of the entire directory.

### Output

https://github.com/greghope667/comet_project_results contains a description of the output table format produced by this code, as well as the output when run on the entire Kepler dataset. See there also for description of the format of the txt files with dips, and how to filter then with the awk scripts.

* all_snr_gt_5.txt is the full list of 67,532 potential transits
* all_snr_gt_5_ok.txt is the final list of 7,217 transits


