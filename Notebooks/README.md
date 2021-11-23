### Notebooks

`masking.ipynb` shows how we detrended these lightcurves. We noticed a common trend at the beginning and end of lightcurves where there is a spike in flux, which we needed to remove. Furthermore, the missing data at the downlinks had also needed to be considered as these can create artefacts which can be mistaken as exocomet transits, see [Kennedy et al 2018](https://arxiv.org/abs/1811.03102). We implement a concept of the Median Absolute Deviation (MAD), and set a threshold from the MAD to cut off the anomalous data points. 
