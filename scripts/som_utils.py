"""
@author: Azib Norazman

description: Tools related to using the SOM to cluster exocomet candidates.
"""

import pandas as pd
import numpy as np



def create_som_cutout(table, minT, half_cutout_length=120):
    """creates cutout of lightcurve to prepare for SOM. The SOM requires all lightcurves to be the same length.

    inputs:
    :table: lightcurve data
    :minT: the time which `processing` output
    :half_cutout_length: the size of half the window desired in number of cadences eg: 120 cadences corresponds to 2.5 days wide, therefore the full window is 5 days long.

    returns:
    :cutout: a sliced lightcurve centred on the depth from the model fitting
    """

    if not isinstance(table, pd.DataFrame):
        table = table.to_pandas()
    som_cutout = table[table.iloc[(table['time']-minT).abs().argsort()[:1]]['time'].index[0] - half_cutout_length: table.iloc[(table['time']-minT).abs().argsort()[:1]]['time'].index[0] + half_cutout_length]
    return som_cutout