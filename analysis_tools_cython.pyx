#cython: language_level=3
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.stats import sigma_clip, sigma_clipped_stats
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from astropy.timeseries import LombScargle
from wotan import flatten
import numpy as np
cimport numpy as np
import math
import eleanor
import sys,os
import kplr
import data
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt


def download_lightcurve(file, path='.'):
    """Get a light curve path, downloading if necessary."""

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

def import_XRPlightcurve(file_path,sector: int,clip=3,flux=None,drop_bad_points=True,ok_flags=[],return_type='astropy'):
    """
    file_path: path to file (takes pkl and csv)
    sector = lightcurve sector
    drop_bad_points: Removing outlier points. Default False
    mad_plots: plots MAD comparisons
    q: lightcurve quality, default 0 (excludes all non-zero quality)
    clip: Sigma to be clipped by (default 4)
    return_type: Default 'astropy'. Pandas DataFrame also available with 'pandas' 

    returns
        - table: Astropy table of lightcurve
        - ok_flags = [14]: the MAD excluded data.
        - info: additional information about the lightcurve (TIC ID, RA, DEC, TESS magnitude, Camera, Chip)
    """
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

    else:
        # think about how many tic id's will be here; should there be a for loop to iterate etc?
        tic = pd.read_csv(file_path)
        star = eleanor.Source(tic=tic, sector=sector) # specifies sector. If you want the latest sector, sector=None
        api_data = eleanor.TargetData(star, height=15, width=15, bkg_size=31, do_psf=False, do_pca=True, regressors='corner')
        df = pd.DataFrame([api_data.time,api_data.corr_flux,api_data.quality]).T
        columns = ['time','corrected flux','quality']
        df.columns = columns


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
    #print(len(table),"length at end")

    if return_type == 'pandas':

        return table.to_pandas(), lc[0:6]
    else:

        return table, lc[0:6]


def import_lightcurve(file_path, drop_bad_points=True, flux='PDCSAP_FLUX',
                      ok_flags=[]):
    """Returns (N by 2) table, columns are (time, flux).

    Flags deemed to be OK are:
    5 - reaction wheel zero crossing, matters for short cadence (Kepler)
    """

    try:
        hdulist = fits.open(file_path)
    except FileNotFoundError:
        print("Import failed: file not found")
        return

    objdata = hdulist[0].header
    scidata = hdulist[1].data
  
    try:
        if 'kplr' in file_path:
            table = Table(scidata)['TIME','PDCSAP_FLUX','SAP_QUALITY']
            info = [objdata['OBJECT'],objdata['KEPLERID'],objdata['KEPMAG'],objdata['QUARTER'],objdata['RA_OBJ'],objdata['DEC_OBJ']]
        elif 'tasoc' in file_path:
            table = Table(scidata)['TIME','FLUX_CORR','QUALITY']
        else:
        #elif 'tess' in file_path:
            #try:
            table = Table(scidata)['TIME','PDCSAP_FLUX','QUALITY']
            info = [objdata['OBJECT'],objdata['TICID'],objdata['TESSMAG'],objdata['SECTOR'],objdata['CAMERA'],objdata['CCD'],objdata['RA_OBJ'],objdata['DEC_OBJ']]
            #except:
            #    time = scidata.TIME
            #    flux = scidata.PDCSAP_FLUX
            #    quality = scidata.QUALITY
    except:
        table = Table(scidata)['TIME','SAP_FLUX','QUALITY']
    
    hdulist.close()
    
    if drop_bad_points:
        bad_points = []
        if 'kplr' in file_path:
            q_ind = get_quality_indices(table['SAP_QUALITY'])
        elif 'tess' in file_path:
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

    return table, info


def import_tasoclightcurve(file_path, drop_bad_points=False, flux='FLUX_CORR',
                      ok_flags=[]):

    try:
        hdulist = fits.open(file_path)
    except FileNotFoundError:
        print("Import failed: file not found")
        return

    scidata = hdulist[1].data
    table = Table(scidata)['TIME','FLUX_CORR','QUALITY']

    return table

def calculate_timestep(table):
    """Returns median value of time differences between data points,
    estimate of time delta data points."""

    dt = [ table[i+1][0] - table[i][0] for i in range(len(table)-1) ] # calculates difference between (ith+1) - (ith) point 
    dt.sort()
    return dt[int(len(dt)/2)] # median of them.

    #np.median(np.diff(table['time']))

    

def clean_data(table):
    """Interpolates missing data points, so we have equal time gaps
    between points. Returns three numpy arrays, time, flux, real.
    real is 0 if data point interpolated, 1 otherwise."""

    time = []
    flux = []
    quality = []
    real = []
    timestep = calculate_timestep(table)

    for row in table:
        ti, fi, qi = row

        if len(time) > 0:
            steps = int(round( (ti - time[-1])/timestep )) # (y2-y1)/(x2-x1)
            if steps > 1:
                fluxstep = (fi - flux[-1])/steps
                # For small gaps, pretend interpolated data is real.
                if steps > 3:
                    set_real=0
                else:
                    set_real=1

                for _ in range(steps-1):
                    time.append(timestep + time[-1])
                    flux.append(fluxstep + flux[-1])
                    quality.append(0)
                    real.append(set_real)
        time.append(ti)
        flux.append(fi)
        quality.append(qi)
        real.append(1)
    return [np.array(x) for x in [time,flux,quality,real]]


def normalise_flux(flux):
    """Requires flux to be a numpy array.
    Normalisation is x --> (x/mean(x)) - 1"""
    flux = np.nan_to_num(flux)
    return flux/flux.mean() - np.ones(len(flux))


def fourier_filter(flux,freq_count):
    """Attempt to remove periodic noise by finding and subtracting
    freq_count number of peaks in (discrete) fourier transform."""

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
    inputs:
    - flux
    - maximum half width in cadences (eg 2.5 days: (48*2.5)/2) for 30 min)
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


def gauss(x,A,mu,sigma):
    return abs(A)*np.exp( -(x - mu)**2 / (2 * sigma**2) )

def bimodal(x,A1,mu1,sigma1,A2,mu2,sigma2):
    return gauss(x,A1,mu1,sigma1)+gauss(x,A2,mu2,sigma2)

def skewed_gauss(x,A,mu,sigma1,sigma2):
    y = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] < mu:
            y[i] = gauss(x[i],A,mu,sigma1)
        else:
            y[i] = gauss(x[i],A,mu,sigma2)
    return y


def comet_curve(x,A,mu,sigma,tail):
    y = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] < mu:
            y[i] = gauss(x[i],A,mu,sigma)
        else:
            y[i] = A*math.exp(-abs(x[i]-mu)/tail)
    return y


def single_gaussian_curve_fit(x,y):
    # Initial parameters guess
    i = np.argmax(y)
    A0 = y[i]
    mu0 = x[i]
    sigma0 = (x[-1]-x[0])/4

    params_bounds = [[0,x[0],0], [np.inf,x[-1],sigma0*4]]
    params,cov = curve_fit(gauss,x,y,[A0,mu0,sigma0],bounds=params_bounds)
    return params


def nonzero(T):
    """Returns a 1d array of the nonzero elements of the array T"""
    return np.array([i for i in T.flat if i != 0])


def skewed_gaussian_curve_fit(x,y):
    # Initial parameters guess
    i = np.argmax(y)

    width = x[-1]-x[0]

    params_init = [y[i],x[i],width/3,width/3]

    params_bounds = [[0,x[0],0,0], [np.inf,x[-1],width/2,width/2]]
    params,cov = curve_fit(skewed_gauss,x,y,params_init,
                            bounds=params_bounds)
    return params


def comet_curve_fit(x,y):
    # Initial parameters guess
    i = np.argmax(y)

    width = x[-1]-x[0]

    params_init = [y[i],x[i],width/3,width/3]

    params_bounds = [[0,x[0],0,0], [np.inf,x[-1],width/2,width/2]]
    params,cov = curve_fit(comet_curve,x,y,params_init,bounds=params_bounds)
    return params


def double_gaussian_curve_fit(T):
    """Fit two normal distributions to a test statistic vector T.
    Returns (A1,mu1,sigma1,A2,mu2,sigma2)"""

    data = nonzero(T)
    N = len(data)

    T_min = data.min()
    T_max = data.max()

    # Split data into 100 bins, so we can approximate pdf.
    bins = np.linspace(T_min,T_max,101)
    y,bins = np.histogram(data,bins)
    x = (bins[1:] + bins[:-1])/2


    # We fit the two gaussians one by one, as this is more
    #  sensitive to small outlying bumps.
    params1 = single_gaussian_curve_fit(x,y)
    y1_fit = np.maximum(gauss(x,*params1),1)

    y2 = y/y1_fit
    params2 = single_gaussian_curve_fit(x,y2)

    params = [*params1,*params2]

    return params


def score_fit(y,fit):
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
    N = len(real)
    if asym == -2:
        return "end"
    elif asym == -4:
        return "gap"
    elif asym == -5:
        return "gapJustBefore"
    elif m < 3:
        return "point"
    elif real[(n-2*m):(n-m)].sum() < 0.5*m:
        return "artefact"
    else:
        return "maybeTransit"


def calc_shape(m,n,time,quality,flux,cutout_half_width=3,
               n_m_bg_start=3, n_m_bg_end=1):
    """Fit both symmetric and comet-like transit profiles and compare fit.
    Returns:
    (1) Asymmetry: ratio of (errors squared)
    Possible errors and return values:
    -1 : Divide by zero as comet profile is exact fit
    -2 : Too close to end of light curve to fit profile
    -3 : Unable to fit model (e.g. timeout)
    -4 : Too much empty space in overall light curve or near dip
    -5 : Gap within 2 days before dip

    (2,3) Widths of comet curve fit segments.
    info: t, x, q, fit1 and fit2 are the transit shape elements 

    """
    w = cutout_half_width
    if n-w*m >= 0 and n+w*m < len(time):
        t = time[n-w*m:n+w*m] # time
        if (t[-1]-t[0]) / np.median(np.diff(t)) / len(t) > 1.5:
            return -4,-4,-4
        t0 = time[n]
        diffs = np.diff(t)
        for i,diff in enumerate(diffs):
            if diff > 0.5 and (t0-t[i])>0 and (t0-t[i])<2:
                return -5,-5,-5
        x = flux[n-w*m:n+w*m] # flux
        q = quality[n-w*m:n+w*m]
        # background_level = (sum(x[:m]) + sum(x[(2*w-1)*m:]))/(2*m)
        bg_l1 = np.mean(x[:n_m_bg_start*m])
        bg_t1 = np.mean(t[:n_m_bg_start*m])
        bg_l2 = np.mean(x[(2*w-n_m_bg_end)*m:])
        bg_t2 = np.mean(t[(2*w-n_m_bg_end)*m:])
        grad = (bg_l2-bg_l1)/(bg_t2-bg_t1)
        background_level = bg_l1 + grad * (t - bg_t1)
        x -= background_level

        try:
            params1 = single_gaussian_curve_fit(t,-x)
            params2 = comet_curve_fit(t,-x)
        except:
            return -3,-3,-3, [t,x,q]

        fit1 = -gauss(t,*params1)
        fit2 = -comet_curve(t,*params2)

        scores = [score_fit(x,fit) for fit in [fit1,fit2]]
        if scores[1] > 0:
            return scores[0]/scores[1], params2[2], params2[3], [t,x,q,fit1,fit2]
        else:
            return -1,-1,-1
    else:
        return -2,-2,-2


def d2q(d):
    '''Convert Kepler day to quarter'''
    qs = [130.30,165.03,258.52,349.55,442.25,538.21,629.35,719.60,802.39,
          905.98,1000.32,1098.38,1182.07,1273.11,1371.37,1471.19,1558.01,1591.05]
    for qn, q in enumerate(qs):
        if d < q:
            return qn


def get_quality_indices(sap_quality):
    '''Return list of indices where each quality bit is set'''
    q_indices = []
    for bit in np.arange(21)+1:
        q_indices.append(np.where(sap_quality >> (bit-1) & 1 == 1)[0]) # returns sap_quality as bit (2**bit) 

    return q_indices


def normalise_lc(flux):
    return flux/flux.mean()

def remove_zeros(data, flux):
    return data[data[flux] != 0]

def smoothing(t,flux,method=3,wotan_method='lowess'):
    """
    1: Lomb-Scargle
    2: Two-step Lomb-Scargle
    3: wotan tool: default lowess
    
    """
    if method != 3: # this block of code is the same for methods 1 and 2
        flux = normalise_flux(flux)
        flux_ls = np.copy(flux)
        lombscargle_filter(t,flux_ls,real,power) 
        periodicnoise_ls = flux - flux_ls 
        flux_ls = flux_ls * real
        if method == 1: 
            return flux_ls, periodicnoise_ls # returns one-step Lomb Scargle
        elif method == 2:
            masked_flux = np.copy(flux)
            masked_flux[n - 2*math.ceil(n*timestep) : n + 2*math.ceil(n*timestep)] = 0  # placeholder. need to change to match duration
            original_masked_flux = np.copy(masked_flux)
            lombscargle_filter(t, masked_flux, real, power)
            periodicnoise_ls2 = original_masked_flux - masked_flux
            masked_flux = masked_flux * real
            final_flux = flux - periodicnoise_ls2
            final_flux = final_flux * real
            return final_flux, periodicnoise_ls2, flux_ls, periodicnoise_ls # returns two-step Lomb Scargle
    elif method == 3:
        flattened_flux, trend_flux = flatten(time=t,flux=flux,method=wotan_method,window_length=2.5,return_trend=True)
        return flattened_flux, trend_flux

def processing(table,f_path,make_plots=False,power=0.08,twostep=False): 
    

    f = os.path.basename(f_path)
    if f.endswith('.pkl'):
        tic = f.split('_')[-1].split('.pkl')[0] # eleanor
    elif "spoc" in f:
        tic = f.split('_')[-5]
    else:
        print("TIC ID unidentified. Stopping...")
        break

    
    if len(table) > 120: # 120 represents 2.5 days
       
        t, flux, quality, real = clean_data(table)
        timestep = calculate_timestep(table)
        factor = ((1/48)/timestep)
        # now throw away interpolated points (we're reprocessing
        # and trying to get the shape parameters right)
        #t = t[np.array(real,dtype=bool)]
        #flux = flux[np.array(real,dtype=bool)]
        #quality = quality[np.array(real,dtype=bool)]
        #real = real[np.array(real,dtype=bool)]
        N = len(t)
        ones = np.ones(N)
        #flux = normalise_flux(flux)
        A_mag = np.abs(np.fft.rfft(normalise_flux(flux)))
        # periodicnoise = flux - filteredflux

        # first Lomb-Scargle
        #flux_ls = np.copy(flux)
        #flux_ls =  savgol_filter(flux_ls,145,2)
        #lombscargle_filter(t,flux_ls,real,power) 
        #periodicnoise_ls = flux - flux_ls 
        #flux_ls = flux_ls * real 

        freq, powers = LombScargle(t,flux).autopower()
        if twostep:
            final_flux, smoothing_func, flux_ls, first_smoothing_func = smoothing(t,flux,method=2)
        else:
            final_flux, smoothing_func = smoothing(t,flux,method=3) # default


        T1 = test_statistic_array(final_flux,60 * factor)
        Ts = nonzero(T1).std()

        m, n = np.unravel_index(
        T1.argmin(), T1.shape
        )  # T.argmin(): location of  T.shape: 2D array with x,y points in that dimension
        minT = T1[m, n]
        minT_time = t[n]
        minT_duration = m * timestep
        Tm_start = n-math.floor((m-1)/2)
        Tm_end = Tm_start + m
        Tm_depth = flux[Tm_start:Tm_end].mean() 
        #final_flux = flux_ls.copy() # for plot matching purposes

        # Second Lomb-Scargle
        if twostep:
            #masked_flux = np.copy(flux)
            #masked_flux[n - 2*math.ceil(n*timestep) : n + 2*math.ceil(n*timestep)] = 0  # placeholder. need to change to match duration
            #original_masked_flux = np.copy(masked_flux)
            #periodicnoise_ls2 = original_masked_flux - masked_flux
            ##lombscargle_filter(t, masked_flux, real, power)
            #masked_flux = masked_flux * real
            #final_flux = flux - periodicnoise_ls2
            #final_flux = final_flux * real
            final_flux = smoothing(t,flux,method=2)
            T2 = test_statistic_array(final_flux, 60 * factor)
            
            Ts = nonzero(T2).std()

            m, n = np.unravel_index(T2.argmin(), T2.shape)

            minT = T2[m,n]
            minT_time = t[n]
            minT_duration = m*timestep
            Tm_start = n-math.floor((m-1)/2)
            Tm_end = Tm_start + m
            Tm_depth = flux[Tm_start:Tm_end].mean() 
        try:
            asym, width1, width2, info = calc_shape(m,n,t,quality,final_flux) # check why flux; plotting purposes?
            s = classify(m,n,real,asym)
        except ValueError:
            asym, width1, width2 = calc_shape(m,n,t,quality,final_flux)
            s = classify(m,n,real,asym) 
        
        result_str =\
                f+' '+\
                ' '.join([str(round(a,8)) for a in
                    [minT, minT/Ts, minT_time,
                    asym,width1,width2,
                    minT_duration,Tm_depth]])+\
                ' '+s
        
        if make_plots:
            try:
                os.makedirs("plots") # make directory plot if it doesn't exist
            except FileExistsError:
                pass
            columns = [
                "file",
                "signal",
                "signal/noise",
                "time",
                "asym_score",
                "width1",
                "width2",
                "duration",
                "depth",
                "transit_prob",
            ]
            #result_str_df = pd.DataFrame(data=[result_str.split(' ')],columns=columns)
            fig1, axarr = plt.subplots(10, figsize=(13, 22))

            # get table in pdf
            axarr[0].axis('off')
            axarr[0].table(cellText=[result_str.split(' ')],colLabels=columns,loc='center')
            axarr[1].plot(A_mag)  # fourier plot
            axarr[1].title.set_text("Fourier plot")
            axarr[1].set_xlabel("frequency")
            axarr[1].set_ylabel("power")
            axarr[2].plot(freq, powers)
            axarr[2].title.set_text("Lomb-Scargle plot")
            axarr[2].set_xlabel("frequency")
            axarr[2].set_ylabel("Lomb-Scargle power")
            axarr[3].plot(t, flux, label="flux")
            axarr[3].plot(t, smoothing_func, label="periodic noise")
            axarr[3].set_xlim(np.min(t),np.max(t))
            axarr[3].title.set_text("Original lightcurve and the smoothing plot")
            axarr[3].set_xlabel("Days in BTJD")
            axarr[3].set_ylabel("Normalised flux")
            axarr[3].legend(loc="lower left")
            axarr[4].plot(t, final_flux)  # lomb-scargle plot
            axarr[4].set_xlim(np.min(t),np.max(t))

            # axarr[4].plot(m,marker='o')
            axarr[4].title.set_text("First noise-removed lightcurve (first Lomb-Scargle)")
            axarr[4].set_xlabel("Days in BTJD")
            axarr[4].set_ylabel("Normalised flux")
            im = axarr[5].imshow(
                T1,
                origin="bottom",
                extent=axarr[4].get_xlim() + (0, 2.5),
                aspect="auto",
                cmap="rainbow",
            )

            cax = fig1.add_axes([1.02, 0.09, 0.05, 0.25])
            axarr[5].title.set_text("An image of the lightcurve  with the first Lomb-Scargle")
            axarr[5].set_xlabel("Days in BTJD")
            axarr[5].set_ylabel("Transit width in days")
            axarr[5].set_aspect("auto")

            ## if two step lomb scargle, add subplots ...                

            try:
                axarr[6].plot(t, original_masked_flux + ones, label="masked flux")
                axarr[6].plot(t, periodicnoise_ls2 + ones, label="periodic noise")
                axarr[6].legend()
                axarr[6].set_xlim(np.min(t),np.max(t))
                axarr[6].title.set_text("Lomb-Scargle applied on masked flux")
                axarr[7].plot(t, flux + ones, color="tomato",alpha=0.5)
                axarr[7].scatter(t, flux + ones, color="red",label='original flux',s=3)
                axarr[7].plot(t, final_flux + ones - 0.003,alpha=0.5,color='cadetblue')
                axarr[7].scatter(t, final_flux + ones - 0.003, label="final flux",s=3)
                axarr[7].legend()
                axarr[7].set_xlim(np.min(t),np.max(t))
                axarr[7].title.set_text("Cleaned flux")

                im = axarr[8].imshow(
                    T2,
                    origin="bottom",
                    extent=(axarr[4].get_xlim()) + (0, 2.5),
                    aspect="auto",
                    cmap="rainbow",
                )
                axarr[8].title.set_text("An image of the lightcurve  with the second Lomb-Scargle")
            except: 
                pass

            fig1.colorbar(im, cax=cax)
            fig1.tight_layout()

            try:
                t2, x2, q2, y2, w2 = info[0],info[1],info[2],info[3],info[4]
                axarr[9].plot(t2, x2+1) # flux
                axarr[9].plot(t2,y2+1) # gauss fit
                axarr[9].plot(t2,w2+1) # comet fit
                axarr[9].set_title("Transit shape in box")
                axarr[9].set_xlabel("Days in BTJD")
                axarr[9].set_ylabel("Normalised flux")

            except:
                pass
            
            del original_masked_flux
            
            if twostep:
                fig1.savefig(f'plots/TIC{tic}_twostep.pdf')  
            else:
                fig1.savefig(f'plots/TIC{tic}.pdf')
        

    else:
        result_str = f+' 0 0 0 0 0 0 0 0 notEnoughData'

    del final_flux
    del flux_ls

    return result_str

def folders_in(path_to_parent):
    """Identifies if directory is the lowest directory"""
    for fname in os.listdir(path_to_parent):
        if os.path.isdir(os.path.join(path_to_parent,fname)):
            yield os.path.join(path_to_parent,fname)


