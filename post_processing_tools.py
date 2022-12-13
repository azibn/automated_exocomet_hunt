import pandas as pd
import numpy as np
import glob
import os
from analysis_tools_cython import *


def get_output(file_path):
    """Imports batch_analyse output file as pandas dataframe.

    file_path: ouptut file (.txt format)

    Returns:
    - df: DataFrame of output file.

    """
    if isinstance(file_path, list):
        df = pd.DataFrame(data=file_path)
    elif file_path.endswith(".txt"):
        df = pd.read_csv(file_path, sep=" ", header=None)


    cols = [
        "file",
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
        "transit_prob",
    ]
    df.columns = cols
    return df


def get_metadata(file_path):
    """Imports batch_analyse output file as pandas dataframe.

    file_path: ouptut file (.txt format)

    Returns:
    - df: DataFrame of output file.

    """
    if isinstance(file_path, list):
        df = pd.DataFrame(data=file_path)
    elif file_path.endswith(".txt"):
        df = pd.read_csv(file_path, sep=" ", header=None)


    cols = [
        "file",
        "ticid",
        "ra",
        "dec",
        "magnitude",
        "camera",
        "chip",
        "sector"
    ]
    df.columns = cols
    return df


def get_lightcurves(data, storage_path, sec, plots=False, clip=3):
    """Uses dataframe obtained from `get_output` to retrieve the xrp lightcurves of desired TIC object.

    Notes: `mad_df` must be called separately in the script/notebook.

    data: Pandas DataFrame
    storage_path: path to directory where lightcurve pkl files are saved.

    Returns:
    - Plots in 2x2 format:
        - Top left: the MAD array
    """
    for i in data.file:
        file_paths = glob.glob(os.path.join(storage_path, f"**/{i}"))[0]
        table, info = import_XRPlightcurve(
            file_paths, sector=sec, clip=clip, drop_bad_points=True
        )
        raw_table, _ = import_XRPlightcurve(
            file_paths, sector=sec, clip=clip, drop_bad_points=False
        )  # want to plot raw lightcurve to see if the MAD has really worked
        if plots:
            tic = info[0]
            camera = info[4]
            mad_df = pd.read_json("./data/Sectors_MAD.json")
            mad_arr = mad_df.loc[: len(table) - 1, f"{sec}-{camera}"]
            sig_clip = sigma_clip(mad_arr, sigma=clip, masked=False)
            med_sig_clip = np.nanmedian(sig_clip)
            rms_sig_clip = np.nanstd(sig_clip)

            # fig,(ax1,ax2,ax3,ax4) = plt.subplots(4,figsize=(15,8))
            fig = plt.figure(figsize=(10, 4))
            ax1 = fig.add_subplot(221)
            ax1.scatter(range(0, len(table["time"])), mad_arr, s=2)
            ax1.axhline(np.nanmedian(mad_arr), c="r", label="median line")
            ax1.axhline(
                np.nanmedian(mad_arr) + 10 * np.std(mad_arr[900:950]),
                c="blue",
                label="visualised MAD",
            )  # 10 sigma threshold
            ax1.axhline(
                med_sig_clip + clip * (rms_sig_clip),
                c="black",
                label="sigma clipped MAD",
            )
            ax1.set_title(f"{sec}-{camera} at {clip} sigma")
            ax1.set_ylim([0.5 * np.nanmedian(mad_arr), 4 * np.nanmedian(mad_arr)])
            ax1.legend()
            ax2 = fig.add_subplot(223, sharex=ax1)
            ax2.scatter(range(0, len(raw_table["time"])), raw_table["quality"], s=5)
            ax2.set_yscale("log")
            ax2.set_title("Cadence vs bit")
            ax3 = fig.add_subplot(222)
            ax3.scatter(
                raw_table["time"], normalise_lc(raw_table["corrected flux"]), s=0.4
            )
            ax3.set_title(f"Raw lightcurve for TIC {tic}")
            ax4 = fig.add_subplot(224, sharex=ax3)
            ax4.scatter(table["time"], normalise_lc(table["corrected flux"]), s=0.4)
            ax4.set_title(f"MAD corrected lightcurve for TIC {tic}")

            fig.tight_layout()
            fig.suptitle(f"TIC {tic}", fontsize=16, y=1.05)


def filter_df(df, signal=5, min_asym_score=-0.5, max_asym_score=2.0, duration=0.5):
    """filters df for given parameter range.
    Default settings:
    - `signal/noise` greater than 5.
        - Minimum test statistic is always negative. We flip the sign in plots for convenience.
    - `duration` set to greater than 0.5 days.
    - `asym_score` between 1.00 to 2.0.
    """
    return df[
        (df.duration >= duration)
        & (df["snr"] <= -(signal))
        & (df["asym_score"] >= min_asym_score)
        & (df["asym_score"] <= max_asym_score)
    ]


def distribution(x, y,savefig=False,savefig_path="distribution.png"):
    # box = df[y <= -7.4) & (x >= 1.30) & (df['transit_prob'] == 'maybeTransit') & (x <= 1.60) & (y >= -12)]
    """plots asymmetry score vs signal/noise over a signal of 5"""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(x, y, s=2)
    ax.set_xlim(-0, 1.9)
    ax.set_ylim(-1, 30)
    ax.set_title("SNR vs asymmetry plot", fontsize=14)
    ax.set_xlabel("$\\alpha$", fontsize=12)
    ax.set_ylabel("$S$", fontsize=12)

    #ax.xaxis.label.set_color("white")  # setting up X-axis label color to yellow
    #ax.yaxis.label.set_color("white")  # setting up Y-axis label color to blue
    #ax.tick_params(axis="x", colors="white")  # setting up X-axis tick color to red
    #ax.tick_params(axis="y", colors="white")

    #ax.spines["left"].set_color("white")  # setting up Y-axis tick color to red
    #ax.spines["top"].set_color("white")
    #ax.spines["right"].set_color("white")  # setting up Y-axis tick color to red
    #ax.spines["bottom"].set_color("white")
    fig.tight_layout()
    if savefig:
        fig.savefig(savefig_path,dpi=300)