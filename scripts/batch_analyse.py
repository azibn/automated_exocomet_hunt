#!/usr/bin/env python3
"""
Batch analysis script for processing lightcurve files.

Disables built-in multithreading for performance and processes lightcurves
from various pipelines (eleanor-lite, spoc, xrp) with configurable parameters.
"""
import os
import multiprocessing
import sys
import traceback
import argparse
import glob
from astropy.table import Table
import warnings
import numpy as np
from typing import List, Tuple, Optional
import lightkurve as lk
import astropy
from analysis_tools_cython import (
    import_XRPlightcurve,
    import_lightcurve,
    processing,
    _folders_in,
    _clean_lightcurve_data,
    PIPELINE_DICT,
)

os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

def output_columns(comet_model="comet_curve2"):
    """Column names for the output file, written as a header so the result can
    be loaded directly with pd.read_csv(...) (no separate colnames.json needed).

    Order must match the comma-delimited row built in processing() (.pyx). The
    asymmetry parameter at position [3] and the comet-model goodness-of-fit
    columns are named after the selected model: 'skewness'/'_skew' for the
    skewed Gaussian (where [3] really is the skewness/alpha), or 'tail'/'_comet'
    for comet_curve2 (where [3] is the exponential tail decay parameter).
    """
    if comet_model == "skewed_gaussian":
        asym_param, asym_param_err, comet_tag = "skewness", "skewness_err", "skew"
    else:  # 'comet_curve2'
        asym_param, asym_param_err, comet_tag = "tail", "tail_err", "comet"
    return [
        "path", "TIC_ID", "signal", "snr", "time", "asym_score", "amplitude",
        "width", asym_param, asym_param_err, "duration", "depth", "peak_lspower",
        "mstat", "m", "n", "chisq_gauss", f"chisq_{comet_tag}", "rchisq_gauss",
        f"rchisq_{comet_tag}", "rmse_gauss", f"rmse_{comet_tag}", "mae_gauss",
        f"mae_{comet_tag}", "transit_prob",
    ]

# Define base pipeline configurations
# 
# Note: Not all pipelines have been adapted yet. Additional pipelines can be added:
# - K2SFF
# - QLP (Quick Look Pipeline) 
# - Custom pipelines
# Simply add new config dictionaries with 'columns' and 'info' fields.

_eleanor_lite_config = {
    'columns': ['TIME', 'CORR_FLUX', 'QUALITY', 'FLUX_ERR','FLUX_BKG','X_CENTROID','Y_CENTROID','PCA_FLUX','RAW_FLUX'],
    'info': ['TIC_ID', 'TMAG', 'SECTOR', 'CAMERA','CCD','RA_OBJ', 'DEC_OBJ']
    # TMAG on eleanor-lite is set as 999 for all lightcurves. Don't know why.
}

_kepler_config = {
    'columns': ['TIME', 'flux', 'SAP_QUALITY', 'SAP_FLUX_ERR'],
    'info': ['OBJECT', 'KEPLERID', 'KEPMAG', 'QUARTER', 'RA_OBJ', 'DEC_OBJ']
}

_k2_config = {
    'columns': ['TIME', 'FCOR', 'SAP_QUALITY', 'PDSCAP_FLUX_ERR'],
    'info': ['OBJECT', 'KEPLERID', 'KEPMAG', 'CAMPAIGN', 'RA_OBJ', 'DEC_OBJ']
}

_tess_spoc_config = {
    'columns': ['TIME', 'PDCSAP_FLUX', 'QUALITY','PDCSAP_FLUX_ERR','SAP_BKG'],
    'info': ['TICID','TESSMAG','SECTOR','CAMERA', 'CCD','RA_OBJ','DEC_OBJ']
}

# _eleanor_xrp_config = {
#     'columns': ['time', 'corr_flux', 'quality','flux_err','pca_flux'],
#     'info': ['TIC ID', 'RA', 'DEC', 'TESSMAG', 'Camera','CCD']
# }

pipeline_dict = {
    # Eleanor-lite aliases
    'eleanor-lite': _eleanor_lite_config,
    'ELEANOR-LITE': _eleanor_lite_config,
    'eleanor': _eleanor_lite_config,
    
    # Kepler aliases
    'kepler': _kepler_config,
    'Kepler': _kepler_config,
    'kplr': _kepler_config,
    
    # K2 aliases
    'K2': _k2_config,
    'k2': _k2_config,
    'ktwo': _k2_config,
    
    # TESS-SPOC aliases
    'TESS-SPOC': _tess_spoc_config,
    'tess-spoc': _tess_spoc_config,
    'spoc': _tess_spoc_config,
    
    # Legacy (commented out, not in use)
    # 'eleanor-xrp': _eleanor_xrp_config,
}

# Mission can always be derived from the pipeline, so --mission never needs
# to be passed explicitly for file-based runs.
_PIPELINE_TO_MISSION = {
    'eleanor-lite': 'TESS',
    'TESS-SPOC': 'TESS',
    'eleanor-xrp': 'TESS',
    'Kepler': 'Kepler',
    'K2': 'K2',
    'everest': 'K2',
}

# Detected pipeline names that don't have their own key in the Cython
# PIPELINE_DICT and need to be resolved to an existing config for column
# extraction. EVEREST K2 lightcurves use the same FCOR/FRAW_ERR layout as the
# existing 'K2' config, so they share it.
_PIPELINE_DICT_ALIAS = {
    'everest': 'K2',
}


def detect_pipeline_from_path(file_path: str) -> str:
    """
    Infer the pipeline (and implicitly the mission) from a lightcurve filename.

    Relies on the standard MAST/HLSP naming conventions, e.g.:
        hlsp_gsfc-eleanor-lite_tess_ffi_s0012-..._lc.fits    -> eleanor-lite
        hlsp_tess-spoc_tess_phot_..._lc.fits                 -> TESS-SPOC
        hlsp_everest_k2_llc_202139994-c00_kepler_..._lc.fits -> everest
        kplr<id>-..._llc.fits                                -> Kepler
        ktwo<id>-c<campaign>_..._llc.fits                    -> K2
        tess<date>-s00XX-..._lc.fits                         -> TESS-SPOC
        *.pkl (XRP eleanor pickles)                          -> eleanor-xrp

    Returns a pipeline name; 'everest' shares the 'K2' column config via
    _PIPELINE_DICT_ALIAS. Raises ValueError if the filename doesn't match a
    known convention (pass -pipeline explicitly in that case).
    """
    name = os.path.basename(file_path).lower()

    # XRP eleanor pickles
    if name.endswith(".pkl"):
        return 'eleanor-xrp'

    # HLSP convention: hlsp_<producer>_<mission>_...
    if name.startswith("hlsp_"):
        parts = name.split("_")
        producer = parts[1] if len(parts) > 1 else ""
        mission = parts[2] if len(parts) > 2 else ""
        if "eleanor" in producer:
            return 'eleanor-lite'
        if "spoc" in producer:
            return 'TESS-SPOC'
        if "everest" in producer:
            return 'everest'
        # Unknown producer: fall back to the mission token
        if mission == "k2":
            return 'K2'
        if mission == "kepler":
            return 'Kepler'
        if mission == "tess":
            return 'TESS-SPOC'

    # Raw MAST archive names
    if name.startswith("kplr"):
        return 'Kepler'
    if name.startswith("ktwo"):
        return 'K2'
    if name.startswith("tess"):
        return 'TESS-SPOC'

    raise ValueError(
        f"Could not auto-detect pipeline from filename: {os.path.basename(file_path)}. "
        f"Pass -pipeline explicitly."
    )


def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Set up and configure command-line argument parser.
    
    Returns:
        argparse.ArgumentParser: Configured parser with all script options
        
    Available arguments:
        path: Target directories or files to process
        -t: Number of processing threads (default: 1)
        -f: Flux type selection (default: CORR_FLUX)
        -c: Sigma clipping threshold for outlier removal (default: 3)
        -pipeline: Data pipeline choice (eleanor-lite, spoc, xrp)
        -p: Enable plotting output
        -metadata: Save lightcurve metadata to file
        -n: Skip saving output file (print to terminal only)
    """
    parser = argparse.ArgumentParser(description="Analyse lightcurves in target directory.")
    
    parser.add_argument(help="target directory(s)", default=".", nargs="*", dest="path")
    parser.add_argument(
        "-t", help="number of threads to use", default=1, dest="threads", type=int
    )
    parser.add_argument("-o", default="output.txt", dest="of", help="output file")
    parser.add_argument(
        "-f",
        help='select flux. "CORR_FLUX is default (`eleanor-lite`). XRP lightcurve options are "corrected flux", "PCA flux" or "raw flux". For other lightcurves, options are "PDCSAP_FLUX"',
        dest="f",
        default="CORR_FLUX",
    )
    parser.add_argument(
        "-c",
        help="set sigma clipping threshold for MAD cuts.",
        dest="c",
        default=3,
        type=int,
    )
    parser.add_argument(
        "-step",
        help="enable twostep. used in conjuction with -m fourier",
        dest="step",
        action="store_true",
    )
    parser.add_argument("-p", help="enable plotting", action="store_true", dest="p")
    parser.add_argument(
        "-metadata",
        help="save metadata of the lightcurves as a .txt file",
        dest="metadata",
        action="store_true",
    )
    parser.add_argument("-nice", help="set niceness", dest="nice", default=8, type=int)
    parser.add_argument(
        "-m",
        help="set smoothing method. Default is None.",
        dest="m",
        default=None,
        type=str,
    )
    parser.add_argument("-n", help="does not save output file", action="store_true")
    parser.add_argument(
        "--mission",
        help="NASA mission ('Kepler', 'K2', 'TESS'). Optional: auto-derived from the "
             "detected/selected pipeline when omitted.",
        default=None,
        type=str
    )
    parser.add_argument(
        "-pipeline",
        help="Optional pipeline override. When omitted, the pipeline is auto-detected "
             "from each file's name (eleanor-lite, TESS-SPOC, Kepler, K2, eleanor-xrp).",
        default=None,
        type=str
    )
    parser.add_argument(
        "-q",
        help="drop bad quality data. To keep bad quality data, call this argument. Default is True",
        action="store_false",
    )
    parser.add_argument(
        "-comet_model",
        help="model used for comet fitting/asymmetry score. Options: 'comet_curve2' "
             "(default) or 'skewed_gaussian'.",
        default="comet_curve2",
        choices=["comet_curve2", "skewed_gaussian"],
        dest="comet_model",
    )
    parser.add_argument(
        "-plots_dir",
        help="Directory to save plots in. Default is 'plots'.",
        default="plots",
        dest="plots_dir",
    )
    parser.add_argument(
        "-id",
        help="Process single lightcurve by ID (e.g., '3542116' for KIC or '270577175' for TIC). Catalog inferred from -pipeline argument. Cannot be used with path arguments.",
        dest="target_id",
    )

    parser.add_argument(
        "-s","--sector","--campaign","--quarter",
        help="TESS Sector/Kepler Quarter/K2 Campaign (when used with `-id`.)",
        dest="sector",
    )
    
    return parser


def download_lightcurve(target_id: str, mission: str = 'TESS', author: str = 'TESS-SPOC', sector: int = None):
    """
    Download lightcurve using lightkurve for a given target.
    
    Args:
        target_id: Target identifier (e.g., '270577175')
        mission: Mission name ('TESS', 'Kepler', 'K2')
        author: Author/pipeline ('SPOC', 'ELEANOR', 'K2', 'EVEREST')
        sector: Specific sector/quarter/campaign to download (optional)
        
    Returns:
        Lightcurve table for processing
    """
    if mission == 'TESS':
        target_name = f"{target_id}"
    elif mission == 'Kepler':
        target_name = f"{target_id}"
    elif mission == 'K2':
        target_name = f"{target_id}"
    else:
        target_name = target_id
    
    print(f"Searching for {target_name} lightcurves with mission={mission}, author={author}")
    
    # Search with sector/quarter/campaign filter if specified
    if sector is not None:
        if mission == 'TESS':
            search_result = lk.search_lightcurve(target_name, mission=mission, author=author, sector=sector)
        elif mission == 'Kepler':
            search_result = lk.search_lightcurve(target_name, mission=mission, author=author, quarter=sector)
        elif mission == 'K2':
            search_result = lk.search_lightcurve(target_name, mission=mission, author=author, campaign=sector)
    else:
        search_result = lk.search_lightcurve(target_name, mission=mission, author=author)
    
    if len(search_result) == 0:
        if sector is not None:
            raise ValueError(f"No lightcurves found for {target_name} in {mission} sector/quarter/campaign {sector}")
        else:
            raise ValueError(f"No lightcurves found for {target_name}")
    
    lightcurve = search_result.download()
    
    return lightcurve


def get_files_from_path(path: str) -> List[str]:
    """
    Discover and collect lightcurve files from various input sources.
    
    Handles multiple input types:
    - Single .fits file: Returns the file directly
    - File list (.txt/.csv): Reads file paths from text file
    - Directory: Searches for *lc.fits and *.pkl files
    - Nested directories: Recursively searches subdirectories
    
    Args:
        path: File path, directory path, or file list to process
        
    Returns:
        List of lightcurve file paths ready for processing
        
    Note:
        Uses _folders_in() to determine if recursive search is needed
    """
    if not os.path.isdir(path):
        if os.path.isfile(path):
            if path.endswith(".fits"):
                return [path]
            elif path.endswith((".txt", ".csv")):
                with open(path, "r") as file:
                    return [line.strip() for line in file.readlines()]
        return []
    
    if list(_folders_in(path)):
        # Recursive: **/*lc.fits and **/*.pkl
        fits = glob.glob(os.path.join(path, "**/*lc.fits"), recursive=True)
        pkl = glob.glob(os.path.join(path, "**/*.pkl"), recursive=True)
    else:
        # Direct: *lc.fits and *.pkl  
        fits = glob.glob(os.path.join(path, "*lc.fits"))
        pkl = glob.glob(os.path.join(path, "*.pkl"))
    
    return fits + pkl


try:
    multiprocessing.set_start_method("fork")  # default for >=3.8 is spawn
except RuntimeError:  # there might be a timeout sometimes.
    pass

m = multiprocessing.Manager()
lock = m.Lock()

# pipeline_options = {"xrp":".pkl","spoc":".fits"}


def run_lc(input_data) -> None:
    """
    Process a lightcurve from either a file path or a downloaded lightkurve object.
    
    Args:
        input_data: Either a file path (str) or tuple (lightcurve_object, target_id)
        
    Global Dependencies:
        - args: Command line arguments (flux type, pipeline, options)
        - sector: Sector number for XRP lightcurves
        - lock: Multiprocessing lock for safe file writing
    """
    try:
        # Determine if input is a file path or downloaded lightcurve
        if isinstance(input_data, str):
            # File-based processing
            file_path = input_data
            filename = os.path.basename(file_path)
            print(file_path)

            # Pipeline is auto-detected from the filename unless overridden via
            # -pipeline; mission is then derived from the pipeline.
            pipeline = args.pipeline or detect_pipeline_from_path(file_path)
            mission = args.mission or _PIPELINE_TO_MISSION.get(pipeline, 'unknown')
            config_key = _PIPELINE_DICT_ALIAS.get(pipeline, pipeline)
            extracted_cols = PIPELINE_DICT.get(config_key, {}).get('columns', [])
            print(
                f"  -> mission={mission} | pipeline={pipeline} | "
                f"extracting columns: {extracted_cols}"
            )

            if file_path.endswith(".pkl"):
                table, lc_info = import_XRPlightcurve(
                    file_path, sector=sector, clip=args.c, drop_bad_points=args.q
                )
                table = table[table.colnames[:5]]
            else:
                table, lc_info = import_lightcurve(file_path, flux=args.f, pipeline=config_key)
                if (pipeline == 'eleanor-lite') and ('pca' in args.f.lower()):
                    table = table['TIME','PCA_FLUX','QUALITY','FLUX_ERR','FLUX_BKG','X_CENTROID','Y_CENTROID','CORR_FLUX']
                    table = table[table.colnames[:5]]

            process_name = file_path
            metadata_filename = filename
            
        else:
            # Downloaded lightcurve processing
            lightcurve, target_id = input_data
            
            print(f"Processing downloaded lightcurve for {target_id}")
            
            #table = lightcurve.to_panads()
            table = lightcurve.to_pandas().reset_index()
            table = Table.from_pandas(table)
            table = table.filled(np.nan)

            lc_info = [target_id, lightcurve.mission, getattr(lightcurve, 'sector', 'unknown')]
            
            process_name = f"downloaded_{target_id}"
            metadata_filename = target_id

            # Download mode has no filename to sniff; fall back to -pipeline or
            # the eleanor-lite default.
            pipeline = args.pipeline or 'eleanor-lite'
            mission = args.mission or _PIPELINE_TO_MISSION.get(pipeline, 'unknown')
            config_key = _PIPELINE_DICT_ALIAS.get(pipeline, pipeline)
            table_cols_lower_map = {col.lower(): col for col in table.colnames}

            expected_cols = pipeline_dict[pipeline]['columns']
            expected_cols_lower = [col.lower() for col in expected_cols]

            available_cols = [table_cols_lower_map[col] for col in expected_cols_lower if col in table_cols_lower_map]

            print(
                f"  -> mission={mission} | pipeline={pipeline} | "
                f"extracting columns: {expected_cols}"
            )

            expected_cols = pipeline_dict[pipeline]['columns']
            
            # Map expected columns to actual columns in the table
            time_col = expected_cols[0].lower() if len(expected_cols) > 0 and expected_cols[0].lower() in table_cols_lower_map else None
            flux_col = expected_cols[1].lower() if len(expected_cols) > 1 and expected_cols[1].lower() in table_cols_lower_map else None
            quality_col = expected_cols[2].lower() if len(expected_cols) > 2 and expected_cols[2].lower() in table_cols_lower_map else None
            flux_error_col = expected_cols[3].lower() if len(expected_cols) > 3 and expected_cols[3].lower() in table_cols_lower_map else None
            
            # Get actual column names from the mapping
            if time_col:
                time_col = table_cols_lower_map[time_col]
            if flux_col:
                flux_col = table_cols_lower_map[flux_col]
            if quality_col:
                quality_col = table_cols_lower_map[quality_col]
            if flux_error_col:
                flux_error_col = table_cols_lower_map[flux_error_col]
            
            table = _clean_lightcurve_data(table,
                                          drop_bad_points=args.q,
                                          ok_flags=[],
                                          time_col=time_col,
                                          flux_col=flux_col,
                                          quality_col=quality_col)

            table = Table([table[time_col], table[flux_col], table[quality_col], table[flux_error_col]], 
                          names=['time', 'flux', 'quality', 'flux_error'])

        
        result_str, save_data = processing(
            table,
            process_name,
            lc_info,
            method=args.m,
            make_plots=args.p,
            twostep=args.step,
            plots_dir=args.plots_dir,
            pipeline=config_key,
            comet_model=args.comet_model,
        )

        if args.metadata:
            lc_info_str = " ".join([str(i) for i in lc_info])
            os.makedirs("metadata", exist_ok=True)
            
            if isinstance(input_data, str):
                metadata_file_path = os.path.join("metadata", f"s{sector}.txt")
            else:
                metadata_file_path = os.path.join("metadata", "downloaded.txt")
                
            with open(metadata_file_path, "a") as metadata_file:
                metadata_file.write(f"{metadata_filename} {lc_info_str}\n")

        if args.n:
            print(result_str)
            return
        
        os.makedirs(os.path.join("outputs", "k2"), exist_ok=True)

        lock.acquire()
        with open(os.path.join("outputs", "k2", args.of), "a") as output_file:
            output_file.write(f"{result_str}\n")
        lock.release()
        
    except (KeyboardInterrupt, SystemExit):
        print("Process terminated early, exiting", file=sys.stderr)
        raise
    except Exception as e:
        error_name = input_data if isinstance(input_data, str) else f"downloaded lightcurve for {input_data[1]}"
        print(f"\nError processing {error_name}: {e}", file=sys.stderr)
        traceback.print_exc()


if __name__ == "__main__":
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    os.nice(args.nice)
    paths = [os.path.expanduser(path) for path in args.path]
    
    if ("sector" in args.path[0]) & (args.path[0].endswith('.pkl')):
        sector = int(os.path.split(args.path[0])[0].split("sector")[1].split("_")[1])
        print(f"Processing Sector {sector}")
    elif args.metadata:
        print("Saving lightcurve metadata")
        sector = input("Sector/Quarter/Campaign? ")

    # Handle single target download mode
    if args.target_id:
        # No filename to sniff in download mode: use -pipeline or default.
        dl_pipeline = args.pipeline or 'eleanor-lite'
        dl_mission = args.mission or _PIPELINE_TO_MISSION.get(dl_pipeline, 'TESS')
        print(
            f"Downloading lightcurve for target ID: {args.target_id} "
            f"(mission={dl_mission}, pipeline={dl_pipeline})"
        )

        try:
            lightcurve = download_lightcurve(args.target_id, mission=dl_mission, author=dl_pipeline, sector=args.sector)
            run_lc((lightcurve, args.target_id))
        except Exception as e:
            print(f"Error downloading/processing target {args.target_id}: {e}", file=sys.stderr)
            sys.exit(1)

        sys.exit(0)

    if args.pipeline is None:
        print(f"using {args.f} (pipeline/mission auto-detected per file)")
    elif (args.pipeline == 'eleanor-lite') and ('pca' in args.f.lower()):
        print(f"using PCA FLUX from {args.pipeline} (override)")
    else:
        print(f"using {args.f} from {args.pipeline} (override)")

    # Collect all files to process
    all_files = []
    for path in paths:
        files = get_files_from_path(path)
        all_files.extend(files)
        
        # Print status messages for user feedback
        if not os.path.isdir(path):
            if path.endswith((".txt", ".csv")):
                print("Processing file list")
        elif not list(_folders_in(path)):
            print("this is the lowest subdirectory. running the search...")
        else:
            print("globbing subdirectories")
    
    # Write the CSV header once, before the workers append rows. Only when the
    # output file is new/empty so re-runs that append don't duplicate it.
    if not args.n:
        out_path = os.path.join("outputs", "k2", args.of)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if (not os.path.exists(out_path)) or (os.path.getsize(out_path) == 0):
            with open(out_path, "w") as f:
                f.write(",".join(output_columns(args.comet_model)) + "\n")

    # Process all files
    pool = multiprocessing.Pool(processes=args.threads)
    pool.map(run_lc, all_files)
    pool.close()
    pool.join()
