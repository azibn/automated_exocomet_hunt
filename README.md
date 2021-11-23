Code for automated detection of potential exocomets in light curves.

### Installation

Latest tested in Python 3.9
	
	git clone https://github.com/azibn/automated_exocomet_hunt
	conda create -f environment.yml
	conda activate auto_exo
	./make

Note: M1 users may have issues installing this due to the current tensorflow incompatibility. If you are not using the `eleanor` package, you can try:

	git clone https://github.com/azibn/automated_exocomet_hunt
	conda create -n <environment name> python jupyter jupyterlab scipy astropy numpy pandas pip cython matplotlib
	conda activate <environment name>
	pip install lightkurve kplr
	./make
 
If you still need to use `eleanor`, please see the [issues page](https://github.com/afeinstein20/eleanor/issues/188) for alternatives.

The original installation is below, if neither of the above work.

----

Requires Python 3 (tested in 3.5). Needs Numpy, Scipy, Astropy and Matplotlib libraries, and a working Cython install. 

Install by running:

    git clone https://github.com/greghope667/comet_project
    cd comet_project
    ./make

### Usage

These scripts currently run on TESS and Kepler lightcurves. For TESS, we have currently only used Eleanor lightcurves kindly provided by the XRP group, saved as pickle files and stored in a Google Bucket. Work in progress to utilise other lightcurve (SPOC, TASOC etc) formats is ongoing. 

Kepler lightcurves can be obtained from [MAST](https://archive.stsci.edu/kepler/).

`single_analysis.py` runs on a single file, for example:

    python single_analysis_xrp.py tess/tesslcs_sector_6_104_2_min_cadence_targets_tesslc_270577175.pkl

    wget https://archive.stsci.edu/missions/kepler/lightcurves/0035/003542116/kplr003542116-2012088054726_llc.fits
    ./single_analysis.py kplr003542116-2012088054726_llc.fits


`batch_analyse.py` runs on directories of files, outputting results to a text file with one row per file. `archive_analyse.sh` is a bash script for processing compressed archives of light curve files, extracting them temporarily to a directory.  Both these scripts have multiple options (number of threads, output file location ...), run with help flag (`-h`) for more details.

### Output

https://github.com/greghope667/comet_project_results contains a description of the output table format produced by this code, as well as the output when run on the entire Kepler dataset. See there also for description of the format of the txt files with dips, and how to filter then with the awk scripts.

* all_snr_gt_5.txt is the full list of 67,532 potential transits
* all_snr_gt_5_ok.txt is the final list of 7,217 transits


### Other files

* The jupyter notebook figs.ipynb contains code to explore individual light curves, and makes most of the plots in the paper.
* XRPNotebook.ipynb contains the interactive version of single_analysis.py with Beta Pic as the target star.
* The text file artefact_list.txt contains a list of artefacts found among candidates.
* dr2.xml and young-cl.xml contain votables of stars from Gaia used in the HR diagrams.
