#cython: language_level=3

import os
import math
import eleanor
import sys
import kplr
import xrpdata
import warnings
import json
import pandas as pd
import numpy as np
cimport numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.stats import sigma_clip
from astropy.coordinates import SkyCoord
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from astropy.timeseries import LombScargle
from astroquery.simbad import Simbad
from matplotlib import pyplot as plt
from matplotlib.colorbar import Colorbar
import matplotlib.patches as patches
import matplotlib.gridspec as gs
from post_processing import *
from stats_calcs import *
from wotan import flatten
from scipy.stats import skewnorm
from som_utils import *
plt.rcParams['agg.path.chunksize'] = 10000
warnings.filterwarnings("ignore")



def download_lightcurve(file, path='.'):
    """
    Function: Downloads a lightcurve file from Kepler based on the given file name. It utilizes the kplr package to interact with the Kepler Input Catalog (KIC) API.

    Parameters:
    :file (str): The name of the lightcurve file to download.
    :path (str, optional): The path where the lightcurve file will be saved. The default is the current directory ('.').
    
    Returns:
    :file_path (str): The path of the downloaded lightcurve file."""

    kic_no = int(file.split('-')[0].split('kplr')[1])

    file_path = path+'/data/lightcurves/{:09}/'.format(kic_no)+file
    if os.path.exists(file_path):
        return file_path

    kic = kplr.API()
    kic.data_root = path

    star = kic.star(kic_no)
    lcs = star.get_light_curves(short_cadence=False)

    for i,l in enumerate(lcs):
        if file in l.filename:
            f_i = i
            break

    _ = lcs[i].open() # force download
    return lcs[i].filename

### this function might be deleted
def import_eleanor(tic,sector=None):
    """Converting eleanor lightcurves to dataframes to work with our pipeline
    tic: TIC ID
    sector: sector of target. If none is specified, eleanor will download the latest observation.
    returns:
        - df: dataframe of the eleanor target
    """
    star = eleanor.Source(tic=tic, sector=sector)
    data = eleanor.TargetData(star, height=15, width=15, bkg_size=31, do_psf=False, do_pca=True, regressors='corner')
    df = pd.DataFrame([data.time,data.corr_flux,data.quality]).T
    columns =['time','corrected flux','quality']
    df.columns = columns
    return df

def import_XRPlightcurve(file_path,sector: int,clip=3,drop_bad_points=True,ok_flags=[],return_type='astropy'):

    """
     Function: Imports XRP lightcurve and performs data cleaning.
    
    
    Parameters:
    :file_path (str): The path to the lightcurve files (in .pkl format) to import.
    :sector (int): TESS Sector.
    :clip (float, optional): The sigma value used for sigma clipping. The default value is 3.
    :drop_bad_points (bool, optional): Removes outliers based on non-zero quality flags. The default value is True.
    :ok_flags (list, optional): A list of additional quality flags that are considered acceptable and should not be dropped during the preprocessing. 
        - New flag [14] is the MAD excluded data.
    :return_type (str, optional): Specifies the format of the returned data. 'astropy' returns an astropy Table object, while 'pandas' returns a pandas DataFrame. The default value is 'astropy'.
    
    Returns:
    :data (pd.DataFrame or astropy.table.Table): The preprocessed lightcurve data.
    :info (list): Information about the lightcurve: TIC ID, RA, DEC, TESS magnitude, Camera, and Chip."""

    if file_path.endswith('.pkl'):
        lc = pd.read_pickle(file_path)

        for i in range(len(lc)):
            if isinstance(lc[i], np.ndarray):
                lc[i] = pd.Series(lc[i])
        for_df = lc[6:]  # TIC ID, RA, DEC, TESS magnitude, Camera, Chip not included

        columns = [
            "time",
            "raw flux",
            "corrected flux",
            "PCA flux",
            "flux error",
            "quality",
        ]
        df = pd.DataFrame(data=for_df).T 
        df.columns = columns
        info = lc[0:6].append(sector)

        table = Table.from_pandas(df)

    # loading Ethan Kruse bad times
    bad_times = xrpdata.load_bad_times()
    bad_times = bad_times - 2457000
    
    # loading MAD 
    mad_df = xrpdata.load_mad()
    sec = sector

    camera = lc[4]
    mad_arr = mad_df.loc[:len(table)-1,f"{sec}-{camera}"]
    sig_clip = sigma_clip(mad_arr,sigma=clip,masked=True)

    # applied MAD cut to keep points within selected sigma
    mad_cut = mad_arr.values < ~sig_clip.mask 
    
    # return indices of values above MAD threshold
    matched_ind = np.where(~mad_cut) # indices of MAD's above threshold

    # Change quality of matched indices to 2**(17-1) (or add 2**(17-1) if existing flag already present)
    table['quality'][matched_ind] += 2**(17-1)
    table['quality'] = table['quality'].astype(np.int32) # int32 set so it can work with `get_quality_indices` function

    # Ethan Kruse bad time mask
    mask = np.ones_like(table['time'], dtype=bool)
    for i in bad_times:
        newchunk = (table['time']<i[0])|(table['time']>i[1])
        mask = mask & newchunk
        
    # Apply Kruse bad mask to table
    table = table[mask]

    if drop_bad_points:
        bad_points = []
        q_ind = get_quality_indices(table['quality'])
    
        for j,q in enumerate(q_ind): # j=index, q=quality
            if j+1 not in ok_flags:
                bad_points += q.tolist()
        table.remove_rows(bad_points)

    # Delete rows containing NaN values. 
    nan_rows = [ i for i in range(len(table)) if
            math.isnan(table[i][2]) or math.isnan(table[i][0]) ] 
    table.remove_rows(nan_rows)

    # Smooth data by deleting overly 'spikey' points.
    spikes = [ i for i in range(1,len(table)-1) if \
            abs(table[i][1] - 0.5*(table[i-1][1]+table[i+1][1])) \
            > 3*abs(table[i+1][1] - table[i-1][1])]

    for i in spikes:
        table[i][1] = 0.5*(table[i-1][1] + table[i+1][1])

    if return_type == 'pandas':

        return table.to_pandas(), lc[0:6]
    else:

        return table, lc[0:6]


def mad_cuts(table,info,clip=3):
    # loading Ethan Kruse bad times
    bad_times = xrpdata.load_bad_times()
    bad_times = bad_times - 2457000
    
    # loading MAD 
    mad_df = xrpdata.load_mad()
    sec = info[2]
    camera = info[3]

    mad_arr = mad_df.loc[:len(table)-1,f"{sec}-{camera}"]
    sig_clip = sigma_clip(mad_arr,sigma=clip,masked=True)

    # applied MAD cut to keep points within selected sigma
    mad_cut = mad_arr.values < ~sig_clip.mask 
    
    # return indices of values above MAD threshold
    matched_ind = np.where(~mad_cut) # indices of MAD's above threshold

    # Change quality of matched indices to 2**(17-1) (or add 2**(17-1) if existing flag already present)
    table['QUALITY'][matched_ind] += 2**(17-1)
    table['QUALITY'] = table['QUALITY'].astype(np.int32) # int32 set so it can work with `get_quality_indices` function

    # Ethan Kruse bad time mask
    mask = np.ones_like(table['TIME'], dtype=bool)
    for i in bad_times:
        newchunk = (table['TIME']<i[0])|(table['TIME']>i[1])
        mask = mask & newchunk
        
    # Apply Kruse bad mask to table
    table = table[mask]

    return table


def import_lightcurve(file_path, drop_bad_points=True, flux='PDCSAP_FLUX', 
                      ok_flags=[], pipeline='eleanor-lite', return_type='astropy'):

    """
    Function: Imports lightcurve and performs data cleaning.
    
    Parameters:
    :file_path (str): The path to the lightcurve file to import.
    :drop_bad_points (bool, optional): Specifies whether to drop points flagged as bad during the preprocessing. The default value is True.
    :flux (str, optional): The flux type of the lightcurve. Options: 'PDCSAP_FLUX', 'SAP_FLUX', 'FLUX'. The default is 'PDCSAP_FLUX'. 
    :ok_flags (list, optional): A list of additional quality flags that are considered acceptable and should not be dropped during the preprocessing. Flags deemed to be OK:
        - 5: reaction wheel zero crossing, matters for short cadence (Kepler)
    :pipeline: The pipeline used to process the lightcurve. Options: 'eleanor-lite', 'eleanor', 'kplr', 'ktwo', 'tasoc'. The default is 'eleanor-lite'.
    :return_type (str, optional): Specifies format of the returned data. Options: 'astropy', 'pandas'. The default value is 'astropy'.

    Returns:
    :data (astropy.table.table orpd.DataFrame): The lightcurve with bad points removed by default.
    :info (list): Information about the imported data: ID, Magnitude, Sector/Quarter number, RA (Right Ascension), and DEC (Declination)."""
    try:
        hdulist = fits.open(file_path)
    except FileNotFoundError:
        print("Import failed: file not found")
        return

    objdata = hdulist[0].header
    scidata = hdulist[1].data

    pipeline_dict = {
    'eleanor-lite': {
        'columns': ['TIME', 'CORR_FLUX', 'PCA_FLUX', 'QUALITY', 'FLUX_ERR'],
        'info': ['TIC_ID', 'TMAG', 'SECTOR', 'CAMERA','CCD','RA_OBJ', 'DEC_OBJ']
        # TMAG on eleanor-lite is set as 999 for all lightcurves. Don't know why.
    },
    'kplr': {
        'columns': ['TIME', 'flux', 'SAP_QUALITY', 'SAP_FLUX_ERR'],
        'info': ['OBJECT', 'KEPLERID', 'KEPMAG', 'QUARTER', 'RA_OBJ', 'DEC_OBJ']
    },
    'ktwo': {
        'columns': ['TIME', 'flux', 'SAP_QUALITY', 'PDSCAP_FLUX_ERR'],
        'info': ['OBJECT', 'KEPLERID', 'KEPMAG', 'CAMPAIGN', 'RA_OBJ', 'DEC_OBJ']
    },
    'tasoc': {
        'columns': ['TIME', 'flux', 'QUALITY'],
        'info': []
    },
    'spoc': {
        'columns': ['TIME', 'PDCSAP_FLUX', 'QUALITY','PDCSAP_FLUX_ERR'],
        'info': ['TICID','TESSMAG','SECTOR','CAMERA', 'CCD','RA_OBJ','DEC_OBJ']
    },


    }

    #try:
    table_columns = pipeline_dict[pipeline]['columns']
    table = Table(scidata)[table_columns]
    info = [objdata[field] for field in pipeline_dict[pipeline]['info']]
    #except KeyError:
    #    print("Pipeline not specified. Exiting.")
    #    return
    
    hdulist.close()

    # for eleanor lightcurves, perform MAD cuts by default.
    if (pipeline == 'eleanor-lite'):
        table = mad_cuts(table,info)

    if drop_bad_points:
        bad_points = []
        if 'kplr' in file_path or 'ktwo' in file_path:
            q_ind = get_quality_indices(table['SAP_QUALITY'])
        else:
            q_ind = get_quality_indices(table['QUALITY'])
        
        for j,q in enumerate(q_ind): # j=index, q=quality
            if j+1 not in ok_flags:
                bad_points += q.tolist() # adds bad_points by value of q (the quality indices) and converts to list
    
        # bad_points = [i for i in range(len(table)) if table[i][2]>0]
        table.remove_rows(bad_points)

    # Delete rows containing NaN values. 
    ## if flux or time columns are NaN's, remove them.
    nan_rows = [ i for i in range(len(table)) if
            math.isnan(table[i][1]) or math.isnan(table[i][0]) ]

    table.remove_rows(nan_rows)

    # Smooth data by deleting overly 'spikey' points.
    ## if flux - 0.5*(difference between neihbouring points) > 3*(distance between neighbouring points), spike identified
    spikes = [ i for i in range(1,len(table)-1) if \
            abs(table[i][1] - 0.5*(table[i-1][1]+table[i+1][1])) \
            > 3*abs(table[i+1][1] - table[i-1][1])]

    ## flux smoothened out by changing those points to 0.5*distance between neighbouring points
    for i in spikes:
        table[i][1] = 0.5*(table[i-1][1] + table[i+1][1])

    if (return_type == 'pandas') or (return_type == 'pd'):
        return table.to_pandas(), info

    return table, info


def calculate_timestep(table):
    """
    Function: Calculates the median value of the time differences between data points in a given table. 
    Provides an estimate of the timestep (or time delta) between consecutive data points.

    Parameters:
    :table (array or pandas.DataFrame): The input table containing time-series data.

    Returns:
    :dt (float): The estimated time interval or timestep between consecutive data points."""

    try:
        dt = [ table[i+1][0] - table[i][0] for i in range(len(table)-1) ] # calculates difference between (ith+1) - (ith) point 
        dt.sort()
        return dt[int(len(dt)/2)] # median of them.
    except:
        return np.median(np.diff(table['time'])) ## change this to account for any time column names

    

def clean_data(table):
    """
    Function: Interpolating missing data points, ensuring equal time gaps between points. 
    Returns five numpy arrays: time, flux, quality, real, and flux_error. Real is 0 if data point interpolated, 1 otherwise.

    Parameters:
    :table (astropy.table.table): The input table containing time-series data.
    
    Returns:
    :time (numpy.ndarray): An array of timestamps for each data point, including the interpolated points.
    :flux (numpy.ndarray): An array of flux values for each data point, including the interpolated points.
    :quality (numpy.ndarray): An array indicating the quality of each data point, including the interpolated points.
    :real (numpy.ndarray): An array indicating whether each data point is real (1) or interpolated (0).
    :flux_error (numpy.ndarray): An array of flux error values for each data point, including the interpolated points."""


    time = []
    flux = []
    quality = []
    real = []
    flux_error = []
    timestep = calculate_timestep(table)
    factor = ((1/48)/timestep)
    for row in table:
        ti, fi, qi, fei = row

        if len(time) > 0:
            steps = int(round( (ti - time[-1])/timestep * factor)) # (y2-y1)/(x2-x1)
            if steps > 1:
                fluxstep = (fi - flux[-1])/steps
                fluxerror_step = (fei - flux_error[-1]/steps)
                # For small gaps, pretend interpolated data is real.
                if steps > 2:
                    set_real=0
                else:
                    set_real=1

                for _ in range(steps-1):
                    time.append(timestep + time[-1])
                    flux.append(fluxstep + flux[-1])
                    flux_error.append(fluxerror_step + flux_error[-1])
                    quality.append(0)
                    real.append(set_real)
        time.append(ti)
        flux.append(fi)
        quality.append(qi)
        real.append(1)
        flux_error.append(fei)
    return [np.array(x) for x in [time,flux,quality,real,flux_error]]


def normalise_flux(flux):
    """
    Function: Normalises flux values to 0.
    - Normalisation is x --> (x/median(x)) - 1

    Parameters:
    :flux (numpy.ndarray): The input flux to be normalized.
    
    Returns:
    :normalised flux (numpy.ndarray): The normalized flux array."""

    flux = np.nan_to_num(flux,nan=np.nanmedian(flux))
    return flux/np.nanmedian(flux) - np.ones(len(flux))


def fourier_filter(flux,freq_count):
    """Function: Attempts to remove periodic noise from the flux by finding and subtracting a specified (freq_count)
     number of peaks in the discrete Fourier transform.

    Parameters:
    :flux (numpy.ndarray or array): The input flux array to be filtered.
    :freq_count (int): The number of peaks to be removed from the Fourier transform.

    Returns:
    :filtered_flux (numpy.ndarray): The filtered flux array after subtracting the periodic approximation."""

    A = np.fft.rfft(flux)
    A_mag = np.abs(A)

    # Find frequencies with largest amplitudes.
    freq_index = np.argsort(-A_mag)[0:freq_count]

    # Mult by 1j so numpy knows we are using complex numbers
    B = np.zeros(len(A)) * 1j
    for i in freq_index:
        B[i] = A[i]

    # Fitted flux is our periodic approximation to the flux
    fitted_flux = np.fft.irfft(B,len(flux))

    return flux - fitted_flux


def lombscargle_filter(time,flux,real,min_score):
    """Also removes periodic noise, using lomb scargle methods."""
    time_real = time[real == 1]

    period = time[-1]-time[0] # length of observation (size of sampling interval)
    N = len(time)
    nyquist_period = (2*period)/N # ??

    min_freq = 1/period # Need at least two sampled points in every period you want to capture
    nyquist_freq = N/(2*period) 

    try:
        for _ in range(30):
            flux_real = flux[real == 1]
            ls = LombScargle(time_real,flux_real)
            freq,powers = ls.autopower(method='fast',minimum_frequency=min_freq,maximum_frequency=nyquist_freq,samples_per_peak=10)
            i = np.argmax(powers)
            if powers[i] < min_score:
                break

            flux -= ls.model(time,freq[i])
        
            del ls
    except:
        pass

def lombscargle_plotting(time,flux,real,min_score):
    time_real = time[real == 1]

    period = time[-1]-time[0] # length of observation (sampling interval)
    N = len(time)
    nyquist_period = (2*period)/N

    min_freq = 1/period
    nyquist_freq = N/(2*period) 

    for _ in range(30):
        flux_real = flux[real == 1]
        freq,powers = LombScargle(time_real,flux_real).autopower(method='fast', minimum_frequency=min_freq,maximum_frequency=nyquist_freq,samples_per_peak=10)
    return freq, powers


def test_statistic_array(np.ndarray[np.float64_t,ndim=1] flux, int max_half_width):
    """
    Function: Calculates the test statistic array for a given flux array and maximum half width. 
    The test statistic array represents the statistical significance of the flux values at different widths.
    - Maximum half width is given in in cadences (eg 2.5 days: (48*2.5)/2) for 30 min)

    Parameters:
    flux (np.ndarray[np.float64_t, ndim=1]): Input flux array.
    max_half_width (int): Maximum half width in cadences.

    Returns:
    t_test (np.ndarray[np.float64_t, ndim=2]): Test statistic array.
    """

    cdef int N = flux.shape[0]
    cdef int n = max_half_width # int(max_half_width) max number of cadences in width array should be 120 (2.5 days)

    cdef int i, m, j
    cdef float mu,sigma,norm_factor
    sigma = flux.std()

    cdef np.ndarray[dtype=np.float64_t,ndim=2] t_test = np.zeros([2*n,N])
#    cdef np.ndarray[dtype=np.float64_t,ndim=1] flux_points = np.zeros(2*n)
    """
    m: number of cadences
    """

    for m in range(1,2*n): # looping over the different (full) widths (first cadence to last cadence in terms of data points.)

        m1 = math.floor((m-1)/2) # indices for that width: x
        m2 = (m-1) - m1 # upper bound

        norm_factor = 1 / (m**0.5 * sigma) # noise

        mu = flux[0:m].sum()
        t_test[m][m1] = mu * norm_factor

        for i in range(m1+1,N-m2-1): # the actual search from start of lc to end of lc
        #"""starts from slightly inside the lightcurve"""

            ##t_test[m][i] = flux[(i-m1):(i+m2+1)].sum() * norm_factor
            mu += (flux[i+m2] - flux[i-m1-1]) # flux between some point and sum
            t_test[m][i] = mu * norm_factor

    return t_test


def gauss(t,A,t0,sigma):
    """
    Function: Returns the value of a Gaussian distribution at a given time.

    Parameters:
        :t (float or array): Time or array of times at which to evaluate the Gaussian function.
        :A (float): Amplitude of the Gaussian peak.
        :t0 (float): Mean or centre of the Gaussian distribution.
        :sigma (float): Standard deviation or width of the Gaussian distribution.

    Returns:
        float or array: Value of the Gaussian function at the given time(s)."""

    return abs(A)*np.exp( -(t - t0)**2 / (2 * sigma**2) )

def single_gaussian_curve_fit(x,y):

    assert not (np.isnan(y).any()), "y array contains NaN" 
    assert not (np.isinf(y).any()), "y array contains inf values" 

    """
    Function: Performs a curve fit to the Gaussian function given time and flux.

    Parameters:
        x (array): Independent variable (x-axis) values (time).
        y (array): Dependent variable (y-axis) values (flux).

    Returns:
        params, cov (tuple): The fitted parameters and the covariance matrix."""

    # Initial parameters guess
    i = np.argmax(y)
    A0 = y[i]
    mu0 = x[i]
    sigma0 = (x[-1]-x[0])/4

    params_bounds = [[0,x[0],0], [np.inf,x[-1],sigma0*4]]
    params,cov = curve_fit(gauss,x,y,[A0,mu0,sigma0],bounds=params_bounds)
    return params, cov

def comet_curve(t,A,t0,sigma,tail):
    """
    Function: Calculates the values of an asymmetric Gaussian function representing a comet curve. 
    The difference is the exponential 1/tail term after the mid-transit.

    Parameters:
        t (array): Independent variable (time) values.
        A (float): Amplitude of the Gaussian curve.
        t0 (float): Mean (centre) of the Gaussian curve.
        sigma (float): Standard deviation of the Gaussian curve.
        tail (float): Tail parameter controlling decay rate after t0.

    Returns:
        array: The computed values of the asymmetric Gaussian curve."""

    x = np.zeros(len(t))
    for i in range(len(t)):
        if t[i] < t0:
            x[i] = gauss(t[i],A,t0,sigma)
        else:
            x[i] = A*math.exp(-abs(t[i]-t0)/tail)
    return x

def comet_curve_fit(x,y):
    # Initial parameters guess
    # x = time
    # y = flux
    i = np.argmax(y)

    width = x[-1]-x[0]

    params_init = [y[i],x[i],width/3,width/3]

    params_bounds = [[0,x[0],0,0], [np.inf,x[-1],width/2,width/2]]
    params,cov = curve_fit(comet_curve,x,y,params_init,bounds=params_bounds)
    return params, cov

def skewed_gaussian_curve_fit(x,y,y_err):
    """
    Fits a skewed Gaussian curve to the given data points.

    Parameters:
        x (array-like): time.
        y (array-like): lightcurve flux.
        y_err (array-like): Associated errors for lightcurve flux.

    Returns:
        tuple: A tuple containing two elements:
            - params (array-like): The optimized parameters of the skewed Gaussian curve fit.
            - cov (ndarray): The estimated covariance of the optimized parameters.
    """
    
    i = np.argmax(y)
    width = x[-1]-x[0]
    
    ### params initialisation for skewness, time, mean and sigma
    # amplitude, t0, sigma, skewness
    params_init = [y[i],x[i],width/3,1]
    
    params_bounds=[[0,x[0],0,-30], [np.inf,x[-1],np.inf,30]]
    params,cov = curve_fit(skewed_gaussian,x,y,p0=params_init,sigma=y_err,bounds=params_bounds,maxfev=100000)
    
    return params, cov 

def skewed_gaussian(x, A, t0, sigma, alpha):
    """
    Skewed Gaussian function using the Skewed Student's t-distribution.

    Parameters:
        x: Input data points.
        A: Amplitude of the Gaussian.
        t0: Mean of the Gaussian.
        sigma: Standard deviation of the Gaussian.
        alpha: Skewness parameter (positive for right-skewed, negative for left-skewed).

    Returns:
        y: The value of the skewed Gaussian at each input data point x.
    """
    y = A * skewnorm.pdf(x,alpha,loc=t0, scale=sigma)
    return y

def nonzero(T):
    """Returns a 1d array of the nonzero elements of the array T"""
    return np.array([i for i in T.flat if i != 0])

def score_fit(y,fit):
    """sum of squares"""
    return sum(((y[i]-fit[i])**2 for i in range(len(y))))


def interpret(params):
    # Choose A1,mu1,sigma1 to be stats for larger peak
    if params[0]>params[3]:
        A1,mu1,sigma1,A2,mu2,sigma2 = params
    else:
        A2,mu2,sigma2,A1,mu1,sigma1 = params

    height_ratio = A2/A1
    separation = (mu2 - mu1)/sigma1

    return height_ratio,separation


def classify(m,n,real,asym):
    """
    Function: Classifies the lightcurve based on parameters from the T-statistic and `calc_shape`.

    Parameters:
    :m (int):
    :n (int):
    :real (int):
    :asym (float): The asymmetry score calculated by diving the Gaussian by the skewed Gaussian model.

    Returns:
    :classification: Classification of data. Options are: "maybeTransit", "artefact", "noModelFitted", "gapJustBefore", "gap", "end".
    """

    if asym == -2:
        return "end"
    elif asym == -4:
        return "gap"
    elif asym == -5:
        return "gapJustBefore"
    elif asym == -6:
        return "gapJustAfter"
    elif asym == -3:
        return "noModelFitted"
    elif m < 3:
        return "point"
    elif real[(n-2*m):(n-m)].sum() < 0.5*m:
        return "artefact"
    else:
        return "maybeTransit"

def calc_shape(m,n,time,flux,quality,real,flux_error,n_m_bg_start=3,n_m_bg_scale_factor=1):
    """Fit both symmetric and comet-like transit profiles and compare fit.

    original time: time before interpolation step

    Returns:
    (1) Asymmetry: ratio of (errors squared)
    Possible errors and return values:
    -1 : Divide by zero as comet profile is exact fit
    -2 : Too close to edge of light curve to fit profile
    -3 : Unable to fit model (e.g. timeout)
    -4 : Too much empty space in overall light curve or near dip
    -5 : Transit event too close to (before) data gap, within 1.5 days of gap.
    -6 : Transit event too close to (after) data gap, within 1.5 day after gap.
    (2,3) Widths of comet curve fit segments.
    info: t, x, q, fit1 and fit3 are the transit shape elements 

    Asymmetry score, 

    """
    ## how many transit widths to take the general linear trend from. start is 1/4 length of cutout from beginning, end is 1 from end.
    #first_index = n - (n_m_bg_start*n)
    #last_index = n - (n_m_bg_end*m)
    
    
    ## the transit widths of the cutout from the T-statistic minimum value. 
    ## this project requires the cutout to have more transit widths after the midtransit, to cover more of the tail.
    ## default is set to 1 transit width before and 2 transit widths after 

    n_m_bg_end = n_m_bg_scale_factor*n_m_bg_start

    cutout_before = n-(m*n_m_bg_start)
    cutout_after = n+(m*n_m_bg_end)
    
    
    if cutout_before>= 0 and cutout_after < len(time):

        t = time[cutout_before:cutout_after]
        #time_ori = original_time[cutout_before:cutout_after]

        # total time span / (cadence * length of lightcurve)
        if (t[-1]-t[0]) / (np.median(np.diff(t)) * len(t)) > 1.5:
            print(-4)
            return -4,-4,-4,-4,-4,-4,-4, -4
        
        # min time from T-statistic
        t0 = time[n]
        
        ## the time array without interpolation is used to find any data gaps from distance of data points along time axis.
        time_ori = time[real == 1]
        diffs = np.diff(time_ori)
        
        ### if a transit is less than 0.5 days within 2 days before or after transit centre, remove.
        for i,diff in enumerate(diffs):
            if diff > 0.5 and abs(t0-time_ori[i]) < 1.5: 
                return -5,-5,-5,-5,-5,-5,-5,-5
            
            ### after the data gap
            if diff > 0.5 and abs(t0 - time_ori[i + 1]) < 1.5:
                return -6,-6,-6,-6,-6,-6,-6,-6
            

        x = flux[cutout_before:cutout_after]
        q = quality[cutout_before:cutout_after]
        fe = flux_error[cutout_before:cutout_after]
        
        bg_before = np.mean(x[:int(m/4)])
        bg_time_before = np.mean(t[:int(m/4)])
        bg_after = np.mean(x[-int(round(m/4)):])
        bg_time_after = np.mean(t[-int(round(m/4)):])

        grad = (bg_after-bg_before)/(bg_time_after-bg_time_before)
        background_level = bg_before + grad * (t - bg_time_before)
        x = x - background_level

        try:
            params1, pcov1 = single_gaussian_curve_fit(t,-x)
            params2, pcov2 = comet_curve_fit(t,-x)
            params3, pcov3 = skewed_gaussian_curve_fit(t,-x,fe)
        except:
            return -3,-3,-3,-3,-3,-3,-3, -3

        fit1 = -gauss(t,*params1)
        fit2 = -comet_curve(t,*params2)
        fit3 = -skewed_gaussian(t,*params3)
        depth = fit3.min() # depth of comet (based on minimum point; not entirely accurate, but majority of the time true
        #min_time = t[np.argmin(x)] # time of midtransit/at minimum point

        scores = [score_fit(x,fit) for fit in [fit1,fit3]] # changed for the skewed gaussian fit
        if scores[1] > 0:
            skewness_error = np.sqrt(np.diag(pcov3)[3])
            # params3[0] is the amplitude of the gaussian...
            # params3[2] is the sigma/width of the gaussian...
            # params3[3] is the skewness...

            return scores[0]/scores[1], params3[0], params3[2], params3[3], skewness_error, depth, [t,x,q,fe,background_level], [fit1,fit2,fit3]
        
        else:

            return -1,-1,-1,-1,-1,-1,-1, -1
    else:     

        return -2,-2,-2,-2,-2,-2,-2, -2


def d2q(d):
    '''Convert Kepler day to quarter'''
    qs = [130.30,165.03,258.52,349.55,442.25,538.21,629.35,719.60,802.39,
          905.98,1000.32,1098.38,1182.07,1273.11,1371.37,1471.19,1558.01,1591.05]
    for qn, q in enumerate(qs):
        if d < q:
            return qn


def get_quality_indices(sap_quality):
    """
    Function: Returns a list of indices where each quality bit is set.

    Parameters:
        sap_quality (array): Array containing the SAP_QUALITY values.

    Returns:
        list: List of indices where each quality bit is set."""

    q_indices = []
    for bit in np.arange(21)+1:
        q_indices.append(np.where(sap_quality >> (bit-1) & 1 == 1)[0]) # returns sap_quality as bit (2**bit) 

    return q_indices

def calc_mstatistic(flux):
    """
    Function: Calculates the M Statistic, which quantifies the deviation of average flux from the extrema.
    - First calculates the average and standard deviation of the flux values. Then, identifies the extrema by considering the minimum and maximum 10% of flux values.

    Parameters:
        flux (array): Flux.

    Returns:
        float: M-Statistic."""

    avg = np.nanmedian(flux)
    stdev = np.nanstd(flux)
    # Extrema defined as the min and max 10% fluxes
    ten_pctl = np.percentile(flux, 10)  # returns a float
    nty_pctl = np.percentile(flux, 90)  # retruns a float

    minima = np.where(flux < ten_pctl)  # indices of min
    maxima = np.where(flux > nty_pctl)  # indices of max
    extrema = np.append(minima, maxima)   # all extrema inds
    ext_flux = flux[extrema]
    diff = np.round((avg-np.mean(ext_flux))/stdev, 3)  # ! The M Statistic
    return diff

def smoothing(table,method,window_length=2.5,power=0.08):
    """
    Function: Smoothing function for lightcurve data.

    Parameters:
    :table (astropy.table.table): lightcurve data (minimum input for the table is to have time and flux).
    :method: Choices from: 'lomb-scargle','fourier', 'biweight','lowess','median','mean','rspline','hspline','trim_mean','medfilt','hspline','savgol'.
    - 'lomb-scargle' and 'fourier' are from the Kepler search.
    - All other smoothing options are from the `Wotan` library.

    Returns:
    :cleaned_flux: The cleaned flux by the chosen smoothing method.
    :trend_flux: The trend that was removed in the smoothing process.
    """
    wotan_methods = ['biweight','lowess','median','mean','rspline','hspline','trim_mean','medfilt','hspline','savgol']

    if method in wotan_methods:
        flattened_flux, trend_lc = flatten(table[table.colnames[0]],table[table.colnames[1]],method=method,window_length=window_length,return_trend=True)
        return flattened_flux, trend_lc 
    elif (method == 'lomb-scargle') or (method == 'fourier'): # this block of code is the same for methods 1 and 2
        t, flux, quality, real, flux_error = clean_data(table)
        flux = normalise_flux(flux)
        flux_ls = np.copy(flux)
        lombscargle_filter(t,flux_ls,real,power) 
        periodicnoise_ls = flux - flux_ls 
        flux_ls *= real
        return flux_ls, periodicnoise_ls # returns one-step Lomb Scargle
    elif method==None:
        return table[table.colnames[1]], np.zeros(len(table[table.colnames[1]])) # the "trend flux" is just an array of zeros
    else:
        print("method type not specified. Try again")
        return

def smoothing_twostep(t,timestep,real,flux,m,n,power=0.08):
    masked_flux = np.copy(flux)                    
    masked_flux[n - 3*math.ceil(n*timestep) : n + 3*math.ceil(n*timestep)] = 0  
    original_masked_flux = np.copy(masked_flux)
    lombscargle_filter(t, masked_flux, real, power)
    periodicnoise_ls2 = original_masked_flux - masked_flux
    masked_flux = masked_flux * real
    final_flux = flux - periodicnoise_ls2
    final_flux *= real
    return final_flux, periodicnoise_ls2, original_masked_flux

def processing(table,f_path='.',lc_info=None,method=None,som_cutouts=False,som_cutouts_directory_name='som_cutouts',make_plots=False,twostep=False): 
    """
    
    Function: The main bulk of the search algorithm.

    Inputs:
    :table: Lightcurve table containing time, flux, quality, and flux error (needs to only be these four columns).
    :f_path: Path to file. Default is '.'
    :lc_info: Metadata about the lightcurve, usually obtained from `import_lightcurve` or `import_XRPlightcurve`. Default is None.
    :method: Choice of smoothing method for lightcurves. Default is None.
    :som_cutouts: Create SOM (self-organizing map) cutouts of the lightcurve. Default is False.
    :som_cutouts_directory_name: Name of directory to save cutouts in.
    :make_plots: Creating plots of lightcurve (pre and post-cleaning), the T-statistic of the lightcurve, its position on the SNR/alpha distribution, 
      and a zoomed-in cut for potential candidates.
    :twostep: Perform two-step smoothing (compatible with Fourier/Lomb-Scargle methods only). Default is False.

    Returns:
    :result_str: A string containing the result of the search algorithm.
    :[t, flux, quality]: A list containing the processed lightcurve data as arryys

    Note: The lightcurve/table needs to be in the format of time, flux, quality, flux error.
    """

    original_table = table.copy()
    file_basename = os.path.basename(f_path)
    try:
        obj_id = lc_info[0]
    except TypeError:
        obj_id = f_path.split('_')[-1]

    if isinstance(table, pd.DataFrame):
        table = Table.from_pandas(table)
    
    if len(table) > 120: # 120 represents 2.5 days of data
        
        # smoothing operation and normalisation of flux
        ## note: since Wotan performs time-windowed smoothing without the need for interpolation, `clean_data` is placed after the smoothing step to create interpolated points at data gaps (mostly for visual benefit). 
       
        wotan_methods = ['biweight','lowess','median','mean','rspline','hspline','trim_mean','medfilt','hspline']
        if method in wotan_methods:
            flat_flux, trend_flux = smoothing(table, method=method)
            table = Table([table[table.colnames[0]], flat_flux - np.ones(len(flat_flux)), table[table.colnames[2]], table[table.colnames[3]]/np.nanmedian(table[table.colnames[1]])],names=('time','flux','quality','flux_error'))
            _ , nonnormalised_flux, _, _, _ = clean_data(original_table)
            t, flux, quality, real, flux_error = clean_data(table)
            flux *= real

        elif method in ['lomb-scargle', 'fourier']:
            t, flux, quality, real, flux_error = clean_data(table)
            flux = normalise_flux(flux)
            flux_ls = np.copy(flux)
            lombscargle_filter(t, flux_ls, real, 0.08)
            trend_flux = flux - flux_ls
            flux_ls *= real
            flux = flux_ls

        else:
            t, flux, quality, real, flux_error = clean_data(table)
            flux = normalise_flux(flux)
            flux *= real

        #assert not (np.isnan(t).any() or np.isinf(t).any()), "Time array contains NaN or inf values"
        #assert not (np.isnan(flux).any() or np.isinf(flux).any()), "Flux array contains NaN or inf values"


        ## preparing processing
        timestep = calculate_timestep(table)
        factor = ((1/48)/timestep)
        N = len(t)
        ones = np.ones(N)

        ## fourier and Lomb-Scargle computations
        A_mag = np.abs(np.fft.rfft(normalise_flux(flux)))
  
        freq, powers = LombScargle(t,flux).autopower() # think about that one
        peak_power = powers.max()

        ## M-statistic
        M_stat = calc_mstatistic(flux)

        ## Perform T-statistic search method
        m,n,T1,minT,minT_time,minT_duration,Tm_start,Tm_end,Tm_depth,Ts = run_test_statistic(flux, factor, timestep,t)

        # Second Lomb-Scargle
        if twostep:
            final_flux2, periodicnoise_ls2, original_masked_flux = smoothing_twostep(t,timestep,real,flux,m,n)  
            m,n,T2,minT,minT_time,minT_duration,Tm_start,Tm_end,Tm_depth,Ts = run_test_statistic(final_flux2, factor, timestep,t)


        asym, amplitude, width, skewness, skewness_error, depth, info, fits = calc_shape(m,n,t,flux,quality,real,flux_error)

        ### preparing some variables for statistics ###
        try:
            gauss_fit = fits[0]
            skewed_fit = fits[2]
    
        except:
            pass
  
        try:
            cutout_flux = info[1] # check why this raises errors sometimes
            cutout_flux_error = info[3]


            ### chi square for the two models ###
            chisq_fit1 = chisquare(cutout_flux,gauss_fit,cutout_flux_error)
            chisq_fit3 = chisquare(cutout_flux,skewed_fit,cutout_flux_error)

            ### reduced chi square for the two models ### 
            reduced_chisq_fit1 = reduced_chisquare(cutout_flux,gauss_fit,4,cutout_flux_error)
            reduced_chisq_fit3 = reduced_chisquare(cutout_flux,skewed_fit,4,cutout_flux_error)
            
            ### rmse and mae for both models ###
            rmse_fit1 = rmse(cutout_flux,gauss_fit)
            mae_fit1 = mae(cutout_flux,gauss_fit)

            rmse_fit3 = rmse(cutout_flux,skewed_fit)
            mae_fit3 = mae(cutout_flux,skewed_fit)
        except:
            cutout_flux = None # check why this raises errors sometimes
            cutout_flux_error = None
            chisq_fit1 = chisq_fit3 = reduced_chisq_fit1 = reduced_chisq_fit3 = rmse_fit1 = rmse_fit3 = mae_fit1 = mae_fit3 = 0.0

        ### sorting out the lightcurves into the initial groups ### 
        classification = classify(m,n,real,asym)

        search =\
            f_path+' '+str(obj_id) + ' '+\
            ' '.join([str(round(a,5)) for a in
                [minT, minT/Ts, minT_time,
                asym,amplitude,width, skewness, skewness_error,
                minT_duration,depth, peak_power, M_stat, m,n, chisq_fit1, chisq_fit3,
                reduced_chisq_fit1, reduced_chisq_fit3, rmse_fit1, rmse_fit3, mae_fit1, mae_fit3]])+\
            ' '+classification
        
        ## little fix for string splitting between SPOC lightcurves and XRP ones
        result = search.split()
        midtransit_time = float(result[4])

        if 'TIC' in search:
            final_result = [result[i] + ' ' + result[i+1] if result[i] == 'TIC' else result[i] for i in range(len(result)) if i+1 < len(result)]
        else:
            final_result = [s for s in search.split() if s != 'TIC']

        if make_plots:
            plt.rc('font', family='serif')
        
            try:
                os.makedirs("plots") # make directory plot if it doesn't exist
            except FileExistsError:
                pass
            
            ### column names ###
            with open('colnames.json', 'r', encoding='utf-8') as f:
                check = f.read()
                columns = json.loads(check)
                columns = columns['column_names']
            
            fig = plt.figure(figsize=(20,10)) ## change at top to plt.rcParams["figure.figsize"] = (10,6)

            ### table of results ###
            gs1 = fig.add_gridspec(10,3 ,hspace=0.4,wspace=0.2)
            ax0 = plt.subplot(gs1[0:1,:]) 
            ax0.axis('off')

            search = ax0.table(cellText=[final_result[1:13]], loc='center', colLabels=columns[1:13])
            search.auto_set_font_size(False)
            search.set_fontsize(10)
            ax00 = plt.subplot(gs1[1:2,:]) 
            ax0.axis('off')
            search2 = ax00.table(cellText=[final_result[12:]], loc='center', colLabels=columns[12:])
            search2.auto_set_font_size(False)
            search2.set_fontsize(10)
            ax00.axis('off')


            ### flux and the smoothing function ###
            ax1 = plt.subplot(gs1[2:5,:2]) 
            ax1.scatter(original_table[original_table.colnames[0]], normalise_flux(original_table[original_table.colnames[1]]), s=5,alpha=0.5,zorder=1,label='original lightcurve')
            ax1.scatter(t, flux,s=5,label='smoothened flux',alpha=0.9,zorder=3)
            try:
                ax1.plot(original_table[original_table.colnames[0]],normalise_flux(trend_flux),label='trend',color='black',linewidth=2,zorder=5)
            except:
                pass

            ax1.set_xlim(np.min(t),np.max(t))
            ax1.set_ylabel("Normalised flux")
            ax1.legend(loc="lower left")
            plt.setp(ax1.get_xticklabels(), visible=False)

            ### transit cutout ###
            ax2 = plt.subplot(gs1[2:6,2:]) 
            try:
                gauss_fit = fits[0]
                comet_fit = fits[1]
                skew_fit = fits[2]
                
                cutout_t = info[0]
                cutout_x = info[1]
                    
                ax2.plot(cutout_t, cutout_x,label='data') # flux
                ax2.plot(cutout_t,gauss_fit,label='gaussian model') # gauss fit
                ax2.plot(cutout_t,skew_fit,label='skewed gauss model',color='black') # skewed gaussian
                ax2.plot(cutout_t,comet_fit,label='comet model') # comet fit    
                ax2.set_xlabel("Time - 2457000 (BTJD Days)")
                ax2.legend(loc="lower left")
            except:
                pass

            ###### T-statistic #######
            ax3 = plt.subplot(gs1[5:8, :2])
            im = ax3.imshow(
                T1,
                origin="bottom",
                extent=ax1.get_xlim() + (0, 2.5),
                aspect="auto",
                cmap="rainbow",
            )
            ax3.set_xlabel("Time - 2457000 (BTJD Days)")
            ax3.set_ylabel("Transit width in days") 

            cbax = plt.subplot(gs1[8:9, :2])
            cb = Colorbar(ax=cbax, mappable=im, orientation='horizontal', ticklocation='bottom')#et_x(-0.075)
            try:
                obj_id = lc_info[0]
            except:
                obj_id = input("object id: ")

            #ax5 = plt.subplot(gs1[6:9,2:])
            #ax5.scatter(df.asym_score,abs(df['signal/noise']),alpha=0.3,s=2)
            #ax5.scatter(obj_params.asym_score,abs(obj_params['signal/noise']),color='k')
            #ax5.set_xlabel("Asymmetry Ratio")
            #ax5.set_ylabel("SNR")
            #ax5.title.set_text("SNR vs Asymmetry Ratio")
            #ax5.set_xlim(0, 1.9)
            #ax5.set_ylim(-1, 30)
            #rect = patches.Rectangle((1.05, 5), 2, 30, linewidth=3, edgecolor='r', facecolor='none')
            #ax5.add_patch(rect)

            ## projection in the sky
            #ax6 = plt.subplot(gs1[16:,:3],projection="aitoff")
            #ra = info[-2] * u.degree
            #dec = info[-1] * u.degree
            #d = SkyCoord(ra=ra, dec=dec, frame='icrs')
            #ax6.figure(figsize=(13,9))
            #ax6.set_title("Aitoff projection")
            #ax6.grid(True)
            #ax6.plot(ra, dec, 'o', alpha=2)

            ## HR Diagram
            #ax7 = plt.subplot(gs1[16:,3:])
            #ax7.set_title("HR Diagram")

            ### finding information from simbad ###
            customSimbad = Simbad()
            customSimbad.add_votable_fields("sptype","parallax")

            obj_id = "TIC" + str(obj_id)

            try:
                obj = customSimbad.query_object(obj_id).to_pandas().T
                obj_name, obj_sptype, obj_parallax = obj.loc['MAIN_ID'][0], obj.loc['SP_TYPE'][0],obj.loc['PLX_VALUE'][0]
                fig.suptitle(f" {obj_id}, {obj_name}, Spectral Type {obj_sptype}, Parallax {obj_parallax} (mas)", fontsize = 16,y=0.93)     
                fig.tight_layout()
            except UnboundLocalError:
                print("object ID not found.")
                fig.suptitle("ID not identified.",fontsize = 16,y=0.93)
                pass
            except AttributeError:
                pass
            fig.savefig(f'plots/{obj_id}.png',dpi=300) 


            plt.tight_layout()

            plt.show()
            plt.close()
    else:
        search = file_basename+' 0 0 0 0 0 0 0 0 notEnoughData'

    if som_cutouts:
        try:
            os.makedirs(f'{som_cutouts_directory_name}')
        except FileExistsError:
            pass
        #data_to_cut = pd.DataFrame(data=[original_table[original_table.colnames[0]],original_table[original_table.colnames[1]],original_table[original_table.colnames[2]],original_table[original_table.colnames[3]]]).T # this is normalised, need the original flux
        data_to_cut = pd.DataFrame(data=[t, nonnormalised_flux, quality, flux_error]).T 
        data_to_cut.columns = ['time','flux','quality','flux_err']
        som_lightcurve = create_som_cutout_test(data_to_cut,min_T=midtransit_time,half_cutout_length=36) # 2 day window either side 

        #x1 = np.mean(som_lightcurve.flux[0:12])
        #x2 = np.mean(som_lightcurve.flux[-13:-1]) # the last 12 points

        #y1 = np.mean(som_lightcurve.time[0:24])
        #y2 = np.mean(som_lightcurve.time[-25:-1])
        #grad = (x2-x1)/(y2-y1)
        #background_level = x1 + grad * (som_lightcurve.time - y1)
        #som_lightcurve.flux = som_lightcurve.flux - background_level



        try:
            save_unique_file(obj_id, som_lightcurve,som_cutouts_directory_name)
        except TypeError:
            obj_id = input("object id: ")
            save_unique_file(obj_id, som_lightcurve,som_cutouts_directory_name)

            
        del original_table
    if method == None:
        return search, [t, flux, quality]
    else:
        return search, [t, flux, normalise_flux(trend_flux), quality]


def folders_in(path_to_parent):
    """""
    Yields the paths of subdirectories within a given directory.

    Args:
        path_to_parent (str): The path to the parent directory.
    
    Yields:
        str: The path of a subdirectory within the parent directory.
    
    Raises:
        OSError: If an error occurs while accessing the directory."""

    try:
        for fname in os.listdir(path_to_parent):
            if os.path.isdir(os.path.join(path_to_parent,fname)):
                yield os.path.join(path_to_parent,fname)
    except:
        pass

def run_test_statistic(flux, factor, timestep, t):
        T1 = test_statistic_array(flux,60 * factor)
        m, n = np.unravel_index(
        T1.argmin(), T1.shape
        )  # T.argmin(): location of  T.shape: 2D array with x,y points in that dimension
        minT = T1[m, n] # snr
        minT_time = t[n] # time
        minT_duration = m * timestep
        Tm_start = n-math.floor((m-1)/2)
        Tm_end = Tm_start + m
        Tm_depth = flux[Tm_start:Tm_end].mean() 
        Ts = nonzero(T1[m]).std() # only the box width selected. Not RMS of all T-statistic
        
        return m,n,T1,minT,minT_time,minT_duration,Tm_start,Tm_end,Tm_depth,Ts


def save_unique_file(obj_id, som_lightcurve,som_cutouts_directory_name='som_cutouts'):
    base_filename = f'{som_cutouts_directory_name}/{obj_id}.npz'
    
    # Check if the base filename exists
    if not os.path.exists(base_filename):
        np.savez(base_filename, time=som_lightcurve.time, flux=som_lightcurve.flux, background_subtracted_flux = som_lightcurve.background_subtracted_flux, id=obj_id)
    else:
        # If the base filename exists, find a unique filename
        suffix = 1
        while True:
            unique_filename = f'{som_cutouts_directory_name}/{obj_id}_{suffix}.npz'
            if not os.path.exists(unique_filename):
                np.savez(unique_filename, time=som_lightcurve.time, flux=som_lightcurve.flux, id=obj_id)
                break
            suffix += 1
