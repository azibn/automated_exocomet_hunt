#cython: language_level=3
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from astropy.timeseries import LombScargle
from astroquery.simbad import Simbad
from matplotlib import pyplot as plt
from matplotlib.colorbar import Colorbar
import matplotlib.patches as patches
import matplotlib.gridspec as gs
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
from scripts.post_processing import *
from wotan import flatten
from statistics import median,mean
from scipy.stats import skewnorm, chisquare
import numpy as np
cimport numpy as np
import math
import eleanor
import sys,os
import kplr
import data
import warnings
import scripts.som_utils as som_utils
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
    bad_times = data.load_bad_times()
    bad_times = bad_times - 2457000
    
    # loading MAD 
    mad_df = data.load_mad()
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


def import_lightcurve(file_path, drop_bad_points=True, flux='PDCSAP_FLUX', 
                      ok_flags=[], return_type='astropy'):

    """
    Function: Imports lightcurve and performs data cleaning.
    
    Parameters:
    :file_path (str): The path to the lightcurve file to import.
    :drop_bad_points (bool, optional): Specifies whether to drop points flagged as bad during the preprocessing. The default value is True.
    :flux (str, optional): The flux type of the lightcurve. Options: 'PDCSAP_FLUX', 'SAP_FLUX', 'FLUX'. The default is 'PDCSAP_FLUX'. 
    :ok_flags (list, optional): A list of additional quality flags that are considered acceptable and should not be dropped during the preprocessing. Flags deemed to be OK:
        - 5: reaction wheel zero crossing, matters for short cadence (Kepler)
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

    if 'kplr' in file_path:
        table = Table(scidata)['TIME',flux,'SAP_QUALITY','SAP_FLUX_ERR']
        info = [objdata['OBJECT'],objdata['KEPLERID'],objdata['KEPMAG'],objdata['QUARTER'],objdata['RA_OBJ'],objdata['DEC_OBJ']]
    elif 'ktwo' in file_path:
        table = Table(scidata)['TIME',flux,'SAP_QUALITY','PDSCAP_FLUX_ERR']
        info = [objdata['OBJECT'],objdata['KEPLERID'],objdata['KEPMAG'],objdata['CAMPAIGN'],objdata['RA_OBJ'],objdata['DEC_OBJ']]
    elif 'tasoc' in file_path:
        table = Table(scidata)['TIME',flux,'QUALITY']
    else:
        table = Table(scidata)['TIME',flux,'QUALITY','PDCSAP_FLUX_ERR']
        info = [objdata['TICID'],objdata['TESSMAG'],objdata['SECTOR'],objdata['CAMERA'],objdata['CCD'],objdata['RA_OBJ'],objdata['DEC_OBJ']]
    ## To-do: Add TASOC compatibility.
    
    hdulist.close()

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
    - Normalisation is x --> (x/mean(x)) - 1

    Parameters:
    :flux (numpy.ndarray): The input flux to be normalized.
    
    Returns:
    :normalised flux (numpy.ndarray): The normalized flux array."""

    flux = np.nan_to_num(flux)
    return flux/flux.mean() - np.ones(len(flux))


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

    for m in range(1,2*n): # looping over the different (full) widths

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

def skewed_gaussian(x,a,mean,sigma,m):
    """
    m: amplitude
    x: time
    a: skewness
    mean: time
    sigma: sigma/standard deviation    
    
    """
    return -m * skewnorm.pdf(x,a,loc=mean,scale=sigma)

def skewed_gaussian_curve_fit(x, y, y_err):
    # Initial parameters guess
    ## i = index of min time
    ## x = time
    ## y = flux

    i = np.argmin(y)

    ### params initialisation for skewness, time, mean and sigma
    params_init = [0.01, x[i], 0.1, 0.0001]  # i find these good to initialise with

    # params_bounds = [[0,x[0],0,-np.inf], [np.inf,x[-1],np.inf,np.inf]]
    params, cov = curve_fit(
        skewed_gaussian, x, y, p0=params_init, sigma=y_err
    )
    return params, cov


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

    N = len(real)
    if asym == -2:
        return "end"
    elif asym == -4:
        return "gap"
    elif asym == -5:
        return "gapJustBefore"
    elif asym == -3:
        return "noModelFitted"
    elif m < 3:
        return "point"
    elif real[(n-2*m):(n-m)].sum() < 0.5*m:
        return "artefact"
    else:
        return "maybeTransit"


def calc_shape(m,n,time,flux,quality,flux_error,n_m_bg_start=3,n_m_bg_scale_factor=1):
    """Fit both symmetric and comet-like transit profiles and compare fit.
    Returns:
    (1) Asymmetry: ratio of (errors squared)
    Possible errors and return values:
    -1 : Divide by zero as comet profile is exact fit
    -2 : Too close to edge of light curve to fit profile
    -3 : Unable to fit model (e.g. timeout)
    -4 : Too much empty space in overall light curve or near dip
    (2,3) Widths of comet curve fit segments.
    info: t, x, q, fit1 and fit2 are the transit shape elements 

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
        if (t[-1]-t[0]) / np.median(np.diff(t)) / len(t) > 1.5:
            return -4,-4,-4,-4,-4,-4,-4
        t0 = time[n]
        diffs = np.diff(t)

        for i,diff in enumerate(diffs):
            if diff > 0.5 and abs(t0-t[i])<2:
                return -5,-5,-5

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
            #params2, pcov2 = comet_curve_fit(t,-x)
            params3, pcov3 = skewed_gaussian_curve_fit(t,x,fe)
           
        except:
            return -3,-3,-3,-3,-3,-3,-3

        fit1 = -gauss(t,*params1)
        #fit2 = -comet_curve(t,*params2)
        fit3 = skewed_gaussian(t,*params3)
        depth = fit3.min() # depth of comet (based on minimum point; not entirely accurate, but majority of the time true)
        min_time = t[np.argmin(x)] # time of midtransit/at minimum point
        scores = [score_fit(x,fit) for fit in [fit1,fit3]] # changed for the skewed gaussian fit
        if scores[1] > 0:
            skewness = params3[0]
            skewness_error = np.sqrt(np.diag(pcov3)[0])
            return scores[0]/scores[1], params3[2], params3[3], depth, [t,x,q,fe,fit1,fit3,background_level], skewness, skewness_error # fit2
        else:

            return -1,-1,-1,-1,-1,-1,-1
    else:     

        return -2,-2,-2,-2,-2,-2,-2


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

def processing(table,f_path='.',lc_info=None,method=None,som_cutouts=False,make_plots=False,twostep=False): 
    """
    
    Function: The main bulk of the search algorithm.

    Inputs:
    :table: Lightcurve table containing time, flux, quality, and flux error (needs to only be these four columns).
    :f_path: Path to file. Default is '.'
    :lc_info: Metadata about the lightcurve, usually obtained from `import_lightcurve` or `import_XRPlightcurve`. Default is None.
    :method: Choice of smoothing method for lightcurves. Default is None.
    :som_cutouts: Create SOM (self-organizing map) cutouts of the lightcurve. Default is False.
    :make_plots: Creating plots of lightcurve (pre and post-cleaning), the T-statistic of the lightcurve, its position on the SNR/alpha distribution, 
      and a zoomed-in cut for potential candidates.
    :twostep: Perform two-step smoothing (compatible with Fourier/Lomb-Scargle methods only). Default is False.

    Returns:
    :result_str: A string containing the result of the search algorithm.
    :[t, flux, quality]: A list containing the processed lightcurve data as arryys

    Note: The lightcurve/table needs to be in the format of time, flux, quality, flux error.
    """



    original_table = table
    file_basename = os.path.basename(f_path)
    try:
        obj_id = lc_info[0]
    except TypeError:
        obj_id = f_path.split('_')[-1]

    if isinstance(table, pd.DataFrame):
        table = Table.from_pandas(table)
    if len(table) > 120: # 120 represents 2.5 days

        ## normalising errors - did not implement this as it messed up skewness calculations
        #table[table.colnames[3]] = normalise_flux(table[table.colnames[3]]) ## generalised for lightcurves with different header names
        
        # smoothing operation and normalisation of flux
        ## note: since Wotan performs time-windowed smoothing without the need for interpolation, `clean_data` is placed after the smoothing step to create interpolated points at data gaps (mostly for visual benefit). 
       
        wotan_methods = ['biweight','lowess','median','mean','rspline','hspline','trim_mean','medfilt','hspline']
        if method in wotan_methods:
            flat_flux, trend_flux = smoothing(table, method=method)
            table['flux'] = flat_flux - np.ones(len(flat_flux))
            a = Table()
            a['time'] = table[table.colnames[0]]
            a['flux'] = flat_flux - np.ones(len(flat_flux)) # resets normalisation to zero.
            a['quality'] = table[table.colnames[2]]
            a['flux_error'] = table[table.colnames[3]]
            t, flux, quality, real, flux_error = clean_data(a)
            flux *= real
            table = a

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


        ## preparing processing
        timestep = calculate_timestep(table)
        factor = ((1/48)/timestep)
        N = len(t)
        #flux_error = normalise_error(flux_error)
        ones = np.ones(N)

        ## fourier and Lomb-Scargle computations
        A_mag = np.abs(np.fft.rfft(normalise_flux(flux)))

        freq, powers = LombScargle(t,flux).autopower() # think about that one
        peak_power = powers.max()

        ## M-statistic
        M_stat = calc_mstatistic(flux)

        ## chi-square
        chisq = chisquare(flux)[0] ## [1] returns the p-value

        ## Perform T-statistic search method
        T1 = test_statistic_array(flux,60 * factor)
        m, n = np.unravel_index(
        T1.argmin(), T1.shape
        )  # T.argmin(): location of  T.shape: 2D array with x,y points in that dimension
        minT = T1[m, n]
        minT_time = t[n]
        minT_duration = m * timestep
        Tm_start = n-math.floor((m-1)/2)
        Tm_end = Tm_start + m
        Tm_depth = flux[Tm_start:Tm_end].mean() 
        Ts = nonzero(T1[m]).std() # only the box width selected. Not RMS of all T-statistic

        # Second Lomb-Scargle
        if twostep:
            final_flux2, periodicnoise_ls2, original_masked_flux = smoothing_twostep(t,timestep,real,flux,m,n)
            final_flux = final_flux2
            del final_flux2
            T2 = test_statistic_array(final_flux, 60 * factor)
            

            m, n = np.unravel_index(T2.argmin(), T2.shape)

            minT = T2[m,n]
            minT_time = t[n]
            minT_duration = m*timestep
            Tm_start = n-math.floor((m-1)/2)
            Tm_end = Tm_start + m
            Tm_depth = flux[Tm_start:Tm_end].mean() 
            Ts = nonzero(T2[m]).std()

        asym, width1, width2, depth, info, skewness, skewness_error = calc_shape(m,n,t,flux,quality,flux_error)
        s = classify(m,n,real,asym)

        result_str =\
                f_path+' '+str(obj_id) + ' '+\
                ' '.join([str(round(a,8)) for a in
                    [minT, minT/Ts, minT_time,
                    asym,width1,width2,
                    minT_duration,depth, peak_power, M_stat, skewness, skewness_error, m,n, chisq]])+\
                ' '+s
        
        ## little fix for string splitting between SPOC lightcurves and XRP ones
        result = result_str.split()

        if 'TIC' in result_str:
            final_result = [result[i] + ' ' + result[i+1] if result[i] == 'TIC' else result[i] for i in range(len(result)) if i+1 < len(result)]
        else:
            final_result = [s for s in result_str.split() if s != 'TIC']

        if make_plots:
            #diagnostic_plots(result_str,method,table,lc_info,info)
            plt.rc('font', family='serif')
        
            try:
                os.makedirs("plots") # make directory plot if it doesn't exist
            except FileExistsError:
                pass
            columns = [
                "path",
                "TIC_ID",
                "signal",
                "snr",
                "time",
                "asym_score",
                "width1",
                "width2",
                "duration",
                "depth",
                "peak_lspower",
                "mstat",
                "skewness",
                "skewness_err",
                "m",
                "n",
                "chisquare",
                "transit_prob",
            ]

            fig = plt.figure(figsize=(20,10)) ## change at top to plt.rcParams["figure.figsize"] = (10,6)

            ## table of stats
            gs1 = fig.add_gridspec(9,3 ,hspace=0.4,wspace=0.2)
            ax0 = plt.subplot(gs1[0:1,:]) 
            ax0.axis('off')
            result_str_table = ax0.table(cellText=[final_result], loc='center', colLabels=columns)
            #result_str_table.scale(1.1,1)
            result_str_table.auto_set_font_size(False)
            result_str_table.set_fontsize(7)

            ###### flux and the smoothing function overlayed #######
            ax1 = plt.subplot(gs1[1:4,:2]) 
            ax1.scatter(original_table[original_table.colnames[0]], normalise_flux(original_table[original_table.colnames[1]]), s=10,alpha=0.5,zorder=1,label='original lightcurve')
            ax1.scatter(t, flux,s=10,label='smoothened flux',alpha=0.9,zorder=3)
            try:
                ax1.plot(original_table[original_table.colnames[0]],normalise_flux(trend_flux),label='trend',color='black',linewidth=2,zorder=5)
            except:
                pass

            #if method != None:
            #    ax1.plot(table[table.colnames[0]], normalise_flux(trend_flux),color='orange', label="Trend") # trend flux
            ax1.set_xlim(np.min(t),np.max(t))
            #ax1.title.set_text("Lightcurve and the Smoothing filter")
            ax1.set_ylabel("Normalised flux")
            ax1.legend(loc="lower left")
            plt.setp(ax1.get_xticklabels(), visible=False)

            ## transit cutout
            ax2 = plt.subplot(gs1[1:5,2:]) 
            try:
                skew_fit = info[-2]
                t2, x2, y2, w2 = info[0],info[1],info[4],info[5]
                
                ax2.plot(t2, x2,label='data') # flux
                ax2.plot(t2,y2,label='gaussian model') # gauss fit
                ax2.plot(t2,skew_fit,label='skewed gauss model',color='black') # skewed gaussian
                ax2.set_xlabel("Time - 2457000 (BTJD Days)")
                ax2.legend(loc="lower left")
            except:
                pass

            ###### T-statistic #######
            ax3 = plt.subplot(gs1[4:7,:2])
            im = ax3.imshow(
                T1,
                origin="bottom",
                extent=ax1.get_xlim() + (0, 2.5),
                aspect="auto",
                cmap="rainbow"
            )
            ax3.set_xlabel("Time - 2457000 (BTJD Days)")
            ax3.set_ylabel("Transit width in days")
            cbax = plt.subplot(gs1[7:8,:2]) # Place it where it should be.
            cb = Colorbar(ax = cbax, mappable = im, orientation = 'horizontal', ticklocation = 'bottom')#.set_x(-0.075)
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


            customSimbad = Simbad()
            customSimbad.add_votable_fields("sptype","parallax")

            obj_id = "TIC" + str(obj_id)#

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
            #fig.savefig(f'plots/{obj_id}.png',dpi=300) 


            #if twostep:
            #    fig.savefig(f'plots/{obj_id}_twostep_{method}.png',dpi=300)  
            #else:
            #    fig.savefig(f'plots/{obj_id}_twostep_{method}.png',dpi=300)  
            plt.tight_layout()

            plt.show()

    else:
        result_str = file_basename+' 0 0 0 0 0 0 0 0 notEnoughData'

    if som_cutouts:
        try:
            os.makedirs('som_cutouts/')
        except FileExistsError:
            pass
        data_to_cut = pd.DataFrame(data=[t, flux, quality, flux_error]).T
        data_to_cut.columns = ['time','flux','quality','flux_err']
        som_lightcurve = som_utils.create_som_cutout_test(data_to_cut,min_T=float(result[4]),half_cutout_length=120) # I think minT and the results are basically the same, but for consistency using the one from result_str

        try:
            np.savez(f'som_cutouts/{obj_id}.npz',time=som_lightcurve.time,flux=som_lightcurve.flux)
        except TypeError:
            obj_id = input("object id: ")
            np.savez(f'som_cutouts/{obj_id}.npz',time=som_lightcurve.time,flux=som_lightcurve.flux)
            print(f"saved as {obj_id}.npz")

    if method == None:
        return result_str, [t, flux, quality]
    else:
        return result_str, [t, flux, normalise_flux(trend_flux), quality]


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