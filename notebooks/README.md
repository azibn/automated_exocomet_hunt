# Notebooks

Analysis and exploration notebooks supporting the exocomet search pipeline in `scripts/`. Highlights:

- `process.ipynb`: A breakdown of the main functions of the search method.
- `eda_clustering_umap.ipynb`: An Exploratory Data Analysis of the UMAP technique to cluster candidates from `batch_analyse.py`.
- `injection_testing.ipynb`: Builds off `injection_testing.py`, and forms the injection recovery plot.
- `masking_lcs.ipynb`: How the lightcurves are detrended. Spikes at the beginning and end of lightcurves and missing data at the downlinks can create artefacts mistaken for exocomet transits (see [Kennedy et al. 2018](https://arxiv.org/abs/1811.03102)). We use the Median Absolute Deviation (MAD) with a threshold to cut anomalous data points.
- `smoothing_windows.ipynb`: Sandbox for testing different smoothing windows from `wotan`.
- `SOM*.ipynb` / `som_prep.ipynb`: Self-organising map experiments for candidate classification.
- `Candidates.ipynb` / `CBV-fit-to-candidates.ipynb`: Candidate vetting and cotrending basis vector fits.

The remaining notebooks are exploratory or produce one-off figures (HR diagrams, MAD plots, per-target checks such as `tic229790952.ipynb`). Scratch notebooks live in `notebooks/eda/` and are not tracked.
