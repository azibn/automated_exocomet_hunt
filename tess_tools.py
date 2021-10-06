import pandas as pd
import numpy as np
from astropy.table import Table


def import_XRPlightcurve(file_path):
    """
    Importing the compressed TESS lightcurves from the XRP group.

    file_path: path to file
    quality: specifies which

    returns
        - table: Astropy table format of lightcurve
        - store: additional information about the lightcurve (TIC ID, camera, etc)
    """
    data = pd.read_pickle(file_path)

    ## extracting the lightcurve data and converting to Series from lists
    for i in range(len(data)):
        if isinstance(data[i], np.ndarray):
            data[i] = pd.Series(data[i])
    for_df = data[6:]  # data[0:6] is not relevant in this case.
    columns = ['time', 'raw flux', 'corrected flux', 'PCA flux', 'flux error', 'quality']
    df = pd.DataFrame(data=for_df).T
    df.columns = columns
    table = Table.from_pandas(df)

    return table, data[0:6]

def normalise_lc(flux):
    return flux/flux.mean()

def remove_zeros(data):
    return data[data['PCA flux'] != 0]


# def remove_zeros(data):
#     if type(data) == "astropy.table.table.Table":
#         data = data.to_pandas(df)
#         return data.loc[(data != 0).any(axis=1)]
#     elif type(data) == pd.core.frame.DataFrame:
#         return data.loc[(data != 0).any(axis=1)]
