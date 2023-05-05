### Notebooks

This is a series of test notebooks so that I have an idea of what I am doing.

`analyse_methods.ipynb`:

`asymmetry_test_1.ipynb`: We find that the asymmetry parameter defined previously needs modifying. This notebook goes through a way to define the asymmetry by flipping the lightcurve at a "minimum point", subtracing the two lightcurves, and comparing the residuals. This method turns out to be inconclusive for us, as the integration of area is indistinguishable for a planet case and for a comet case.

`asymmetry_test_2.ipynb`: We find that the asymmetry parameter defined previously needs modifying. This notebook defines asymmetry using a skewed Gaussian model, as opposed to a "half-Gaussian, half-exponential" model. This fitting routine works, and current progress is testing the robustness of this skewed fit.

`barcharts.ipynb`:

`bucket_compare.ipynb`:

`calc_shape_width.ipynb`: Cutouts of the lightcurves are made at the most significant SNR in the lightcurve (provided the criteria are met). This notebook revamps the previous way of defining the width. This is a less hard-coded way of specifying cutout sizes.

`comet_creations.ipynb`:

`comparison.ipynb`:

`creating_gif.ipynb`: How to create a gif of the rolling mean!

`eda_clustering_umap.ipynb`: An Exploratory Data Analysis of the UMAP technique to cluster candidates from `batch_analyse.py`

`eda_output_skewd_gauss.ipynb`: An Exploratory Data Analysis of how the skewed Gaussian affects the distribution of the output information.

`eda_star_sample.ipynb`:

`injection_testing`: This notebook builds off `injection_testing.py`, and forms the injection recovery plot.

`mad_plots.ipynb`:

`masking_lcs.ipynb` shows how we detrended these lightcurves. We noticed a common trend at the beginning and end of lightcurves where there is a spike in flux, which we needed to remove. Furthermore, the missing data at the downlinks had also needed to be considered as these can create artefacts which can be mistaken as exocomet transits, see [Kennedy et al 2018](https://arxiv.org/abs/1811.03102). We implement a concept of the Median Absolute Deviation (MAD), and set a threshold from the MAD to cut off the anomalous data points. 

`notebook_run.ipynb`:

`onestep_vs_twostep.ipynb`:

`process.ipynb`: A breakdown of the main functions of the search method.

`redefine_asymmetry.ipynb`:

`skewed_gaussian_model.ipynb`:

`smoothing_windows.ipynb`: Sandbox on testing different smoothing windows from `wotan`.

`stitching_lightcurves.ipynb`: A potentially useful notebook on how to stitch lightcurves from different TESS sectors.

`twostep_analysis_s10.ipynb`:


