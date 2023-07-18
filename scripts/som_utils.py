"""
@author: Azib Norazman

description: Tools related to using the SOM to cluster exocomet candidates.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def create_som_cutout(table, min_T: float, half_cutout_length=120):
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


### requires testing: includes extrapolation of data if data points are not 2.5 days apart. Ensures all lightcurves have same number of data points for SOM.


def create_som_cutout_test(table, min_T: float, half_cutout_length=120):
    """creates cutout of lightcurve to prepare for SOM. The SOM requires all lightcurves to be the same length.

    inputs:
    :table (pd.DataFrame or astropy.table): The input lightcurve data from which the cutout will be created. If the table is in an astropy format, it will be converted.
    :min_T (numeric): the time value around which the cutout will be centered.
    :half_cutout_length (optional, numeric): the size of half the window desired in number of cadences.

    returns:
    :cutout (pd.DataFrame): a sliced lightcurve with exactly 240 points, centered on min_T.
    """

    if not isinstance(table, pd.DataFrame):
        table = table.to_pandas()

    time_values = table["time"].values
    min_time = np.min(time_values)
    max_time = np.max(time_values)

    target_min_index = np.searchsorted(time_values, min_T) - half_cutout_length
    target_max_index = target_min_index + 2 * half_cutout_length

    if target_min_index < 0 or target_max_index >= len(time_values):
        # Extrapolation is required

        # Interpolate the existing data
        interpolator = interp1d(time_values, table.values.T, fill_value="extrapolate")
        cadence_duration = time_values[1] - time_values[0]
<<<<<<< Updated upstream
        print(cadence_duration)
=======
>>>>>>> Stashed changes
        # Generate the extrapolated time values
        extrapolated_times = np.linspace(
            min_time - half_cutout_length * cadence_duration,
            max_time + half_cutout_length * cadence_duration,
            2 * half_cutout_length + 1,
        )

        # Interpolate the extrapolated data
        extrapolated_data = interpolator(extrapolated_times).T

        # Create a new DataFrame with the extrapolated data
        column_names = table.columns
        extrapolated_table = pd.DataFrame(extrapolated_data, columns=column_names)

        # Extract the cutout around the min_T value
        cutout = extrapolated_table.iloc[
            np.searchsorted(extrapolated_table["time"].values, min_T)
            - half_cutout_length : np.searchsorted(
                extrapolated_table["time"].values, min_T
            )
            + half_cutout_length
            + 1
        ]
    else:
        # No extrapolation needed, extract the cutout directly
        cutout = table.iloc[target_min_index : target_max_index + 1]

    return cutout
