#!/usr/bin/env python3

from astropy.io import fits
import os
import shutil
from pathlib import Path
import argparse

def get_tmag(fits_file):
    try:
        with fits.open(fits_file) as hdul:
            tmag = hdul[0].header['TESSMAG']
            return tmag
    except:
        return None

def organize_by_tmag(input_dir):
    # Create directories for each magnitude range
    for i in range(0, 20):  # Adjust range as needed
        tmag_dir = f"tmag{i}-{i+1}"
        Path(tmag_dir).mkdir(exist_ok=True)

    # Process each fits file
    for file in Path(input_dir).glob('**/*.fits'):
        tmag = get_tmag(file)
        if tmag is not None:
            # Determine which directory this belongs in
            tmag_floor = int(tmag)
            dest_dir = f"tmag{tmag_floor}-{tmag_floor+1}"
            
            # Move the file
            dest_file = Path(dest_dir) / file.name
            shutil.copy2(file, dest_file)