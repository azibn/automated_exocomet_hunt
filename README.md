Code for automated detection of comets in light curves.

### Installation

This TESS portion of the project was developed with a Conda environment (latest tested in Python 3.8). Install by running in the terminal:
	
	git clone https://github.com/azibn/automated_exocomet_hunt
	conda create -n <Environment Name>
	conda activate <Environment Name> --file requirements.txt
	./make

The original installation method is below, if the above does not work.

----

Requires Python 3 (tested in 3.5). Needs Numpy, Scipy, Astropy and Matplotlib libraries, and a working Cython install. 

Install by running:

    git clone https://github.com/greghope667/comet_project
    cd comet_project
    ./make

### Usage

These scripts now run on TESS and Kepler light curve files. For TESS, we specifically use Eleanor lightcurves taken from the XRP group only, saved as pickle files. Work in progress to utilise other TESS light curves (SPOC, TASOC etc) formats is ongoing. 

Kepler light curves can be obtained from [MAST](https://archive.stsci.edu/kepler/)

`single_analysis.py` runs on a single file, for example:

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
