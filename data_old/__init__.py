import pandas as pd
import numpy as np
import pkg_resources


def load_mad():
    stream = pkg_resources.resource_stream(__name__, 'Sectors_MAD.json')
    mad_table = pd.read_json(stream)
    return mad_table


def load_10s_mask():
    stream = pkg_resources.resource_stream(__name__, 'threshold_mask.json')
    mask_10sigma = pd.read_json(stream)
    return mask_10sigma


def load_bad_times():
    stream = pkg_resources.resource_stream(__name__, 'tess_bad_times.txt')
    tess_bad_times = np.loadtxt(stream, comments='#')
    return tess_bad_times

def mount_drive(mount_point):
    try:
        # check if the lookup file for sector one exists
        ref = pd.read_csv(mount_point+"sector1lookup.csv")
        print("Disk already mounted")
    except FileNotFoundError:
        import os
        # If not, try to mount it.
        # presumes the disk is sdc
        os.system(f"sudo mount -o discard,ro /dev/sdc {mount_point}")
        print("Disk mounted")