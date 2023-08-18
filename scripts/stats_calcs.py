import numpy as np

###A set of statistics equations for use in the analysis of transit signals. Made into separate script to neaten up some code###.


def rmse(flux, fit):
    """
    Calculate the Root Mean Squared Error (RMSE) for a given lightcurve and model fit.

    The Root Mean Squared Error (RMSE) is a metric used to assess the accuracy of a model's predictions
    when applied to time series data, such as lightcurve data. It measures the square root of the average
    squared difference between the predicted values (fit) and the actual data (flux).

    Parameters:
        flux (array-like): The actual data representing the observed lightcurve flux values.
        fit (array-like): The predicted data representing the model's fit to the lightcurve data.

    Returns:
        float: The calculated Root Mean Squared Error (RMSE) value.

    Raises:
        AssertionError: If the lengths of 'flux' and 'fit' arrays are not the same.
    """
    assert len(flux) == len(fit), "Input arrays must have the same length."

    # Calculate the squared differences between the predicted values (fit) and the actual data (flux)
    squared_diff = (fit - flux) ** 2

    # Calculate the mean squared difference
    mean_squared_diff = np.mean(squared_diff)

    # Calculate the RMSE by taking the square root of the mean squared difference
    return np.sqrt(mean_squared_diff)


def mae(flux, fit):
    """
    Calculate the Mean Absolute Error (MAE) for a given lightcurve and model fit.

    The Mean Absolute Error (MAE) is a metric used to assess the accuracy of a model's predictions
    when applied to time series data, such as lightcurve data. It measures the average absolute
    difference between the predicted values (fit) and the actual data (flux).

    Parameters:
        flux (array-like): The actual data representing the observed lightcurve flux values.
        fit (array-like): The predicted data representing the model's fit to the lightcurve data.

    Returns:
        float: The calculated Mean Absolute Error (MAE) value.

    Raises:
        AssertionError: If the lengths of 'flux' and 'fit' arrays are not the same.
    """
    assert len(flux) == len(fit), "Input arrays must have the same length."

    # Calculate the absolute differences between the predicted values (fit) and the actual data (flux)
    absolute_diff = np.abs(fit - flux)

    # Calculate the mean absolute difference
    return np.mean(absolute_diff)


def chisquare(flux, fit, flux_err):
    """Returns the chi-squared value of the fit to the flux.

    Args:
        time (array): The time values.
        flux (array): The flux values.
        fit (array): The flux values of the model.
        flux_err (array): The flux error.
        model_params: The parameters `params` from the model fit.

    Returns:
        reduced_chisq (float): The reduced chi-squared value.
    """
    # Calculate the chi-squared value
    chi_squared = np.sum(((flux - fit) / flux_err) ** 2)

    return chi_squared


def reduced_chisquare(flux, fit, num_params, flux_err):
    """Returns the reduced chi-squared value of the fit to the flux.

    Args:
        time (array): The time values.
        flux (array): The flux values.
        fit (array): The flux values of the model.
        flux_err (array): The flux error.
        num__params: The number of parameters `params` from the model fit.

    Returns:
        reduced_chisq (float): The reduced chi-squared value.
    """
    # Calculate the chi-squared value
    chi_squared = np.sum(((flux - fit) / flux_err) ** 2)

    # Calculate the reduced chi-squared value
    N = len(flux)
    degrees_of_freedom = N - num_params
    reduced_chisq = chi_squared / degrees_of_freedom

    return reduced_chisq
