from analysis_tools_cython import comet_curve


def inject_lightcurve(flux, time, depth, injected_time):
    return flux * (
        1 - comet_curve(time, depth, injected_time, 3.02715600e-01, 3.40346173e-01)
    )
