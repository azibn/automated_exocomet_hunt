from analysis_tools_cython import skewed_gaussian


def inject_lightcurve(time, flux, depth, injected_time, sigma, skewness):

    """Injects a simulated transit signal into a given lightcurve.

    This function takes a lightcurve represented by time and flux values and injects
    a simulated transit signal into it. The transit signal is modeled as a skewed
    Gaussian profile centered at the `injected_time` with a specified `depth`, `sigma`,
    and `skewness`.

    Parameters:
    -----------
    time : array-like
        The time values of the original lightcurve data.

    flux : array-like
        The flux values of the original lightcurve data.

    depth : float
        The depth of the simulated transit signal, representing the fractional decrease
        in flux during the transit. Should be between 0 and 1.

    injected_time : float
        The time at which the center of the simulated transit signal occurs.

    sigma : float
        The standard deviation of the Gaussian profile, controlling the width of the
        transit signal.

    skewness : float
        The skewness of the Gaussian profile, controlling the asymmetry of the transit
        signal. Positive values skew the transit signal to the right, while negative
        values skew it to the left.

    Returns:
    --------
    injected_flux : array-like
        The flux values of the lightcurve with the simulated transit signal injected.

    Notes:
    ------
    This function assumes that the `skewed_gaussian` function is available and properly
    defined elsewhere in the codebase."""

    return flux * (
        1 - skewed_gaussian(time, depth, injected_time, sigma, skewness)
    )
