### Notebooks

`masking.ipynb` shows how we detrended these lightcurves. We noticed a common trend at the beginning and end of lightcurves where there is a spike in flux, which we needed to remove. Furthermore, the missing data at the downlinks had also needed to be considered as these can create artefacts which can be mistaken as exocomet transits, see [Kennedy et al 2018](https://arxiv.org/abs/1811.03102). We implement a concept of the Median Absolute Deviation (MAD), and set a threshold from the MAD to cut off the anomalous data points. 


`workbook.ipynb` shows the thought process and procedure for the step-by-step analysis ahead of using `single_analysis_xrp.py`. We also compare directly with `Lightkurve` and `Eleanor`. The example shown here was using Beta Pictoris' exocomet transit, where we discover that by using a custom aperture on the `Eleanor` lightcurve, we see the exocomet transit, whereas the XRP lightcurves do not show that. This is something to consider ahead of our search.
