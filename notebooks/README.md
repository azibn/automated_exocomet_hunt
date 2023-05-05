### Notebooks

This is a series of test notebooks so that I have an idea of what I am doing.

`analyse_methods.ipynb`:

`asymmetry_test_1.ipynb`: We find that the asymmetry parameter defined previously needs modifying. This notebook goes through a way to define the asymmetry by flipping the lightcurve at a "minimum point", subtracing the two lightcurves, and comparing the residuals. This method turns out to be inconclusive for us, as the integration of area is indistinguishable for a planet case and for a comet case.

`asymmetry_test_2.ipynb`: We find that the asymmetry parameter defined previously needs modifying. This notebook defines asymmetry using a skewed Gaussian model, as opposed to a "half-Gaussian, half-exponential" model. This fitting routine works, and current progress is testing the robustness of this skewed fit.


`masking.ipynb` shows how we detrended these lightcurves. We noticed a common trend at the beginning and end of lightcurves where there is a spike in flux, which we needed to remove. Furthermore, the missing data at the downlinks had also needed to be considered as these can create artefacts which can be mistaken as exocomet transits, see [Kennedy et al 2018](https://arxiv.org/abs/1811.03102). We implement a concept of the Median Absolute Deviation (MAD), and set a threshold from the MAD to cut off the anomalous data points. 


