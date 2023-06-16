"""
@author: Azib Norazman

description: Tools related to using the SOM to cluster exocomet candidates.
"""

import pandas as pd

def create_som_cutout(table, min_T, half_cutout_length=120):
    """creates cutout of lightcurve to prepare for SOM. The SOM requires all lightcurves to be the same length.

    inputs:
    :table (pd.DataFrame or astropy.table): The input lightcurve data from which the cutout will be created. If the table is in an astropy format, will be converted.
    :minT (numeric): the time which `processing` output
    :half_cutout_length (optional, numeric): the size of half the window desired in number of cadences eg: 120 cadences corresponds to 2.5 days wide, therefore the full window is 5 days long.
    
    returns:
    :cutout (pd.DataFrame): a sliced lightcurve centred from the minimum time value.
    """

    if not isinstance(table, pd.DataFrame):
        table = table.to_pandas()
    som_cutout = table[
        table.iloc[(table["time"] - min_T).abs().argsort()[:1]]["time"].index[0]
        - half_cutout_length : table.iloc[(table["time"] - min_T).abs().argsort()[:1]][
            "time"
        ].index[0]
        + half_cutout_length
    ]
    return som_cutout
