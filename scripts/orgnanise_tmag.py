#!/usr/bin/env python3

from astropy.io import fits
import os
import shutil
import glob
import argparse
import sys
from tqdm import tqdm
import pandas as pd
import re

def load_tic_catalog(catalog_file):
    print("Loading TIC catalogue...")
    tic_df = pd.read_csv(catalog_file, sep='\t')
    return dict(zip(tic_df['tic_id'], tic_df['Tmag']))

def get_tic_id(filename):
    match = re.search(r'.*?(\d+).*?', filename)
    if match:
        return int(match.group(1))
    return None

def organise_by_tmag(input_dir, tic_catalog):
    print(f"Processing files in {input_dir}")
    
    # Create directories for each magnitude range
    for i in range(0, 20):
        tmag_dir = f"tmag{i}-{i+1}"
        os.makedirs(tmag_dir, exist_ok=True)
    print("Created Tmag directories")

    # Process each fits file
    for file in tqdm(glob.glob(f"{input_dir}/**/*.fits", recursive=True)):
        tic_id = get_tic_id(file)
        if tic_id and tic_id in tic_catalog:
            tmag = tic_catalog[tic_id]
            tmag_floor = int(tmag)
            dest_dir = f"tmag{tmag_floor}-{tmag_floor+1}"
            dest_file = os.path.join(dest_dir, os.path.basename(file))
            shutil.copy2(file, dest_file)

def main():
    try:
        parser = argparse.ArgumentParser(description='Organise TESS FITS files by Tmag into subdirectories')
        parser.add_argument('input_dir', help='Directory containing FITS files')
        parser.add_argument('--tic_catalog', help='Path to TIC catalogue file', default='tic_catalog_v2.txt')
        
        if len(sys.argv) == 1:
            parser.print_help()
            sys.exit(1)
            
        args = parser.parse_args()
        
        # Load TIC catalog
        tic_catalog = load_tic_catalog(args.tic_catalog)
        
        print(f"Starting to organise files from {args.input_dir}")
        organise_by_tmag(args.input_dir, tic_catalog)
        print("Finished organising files")
        
    except Exception as e:
        print(f"Error: {e}")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()