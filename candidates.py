from analysis_tools_cython import *
from post_processing_tools import get_output
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser(description="getting candidates from TESS runs")
parser.add_argument(help="Target output file", nargs=1, dest="file")
parser.add_argument("--min_asym",default=1.05, dest="min_asym")
parser.add_argument("--max_asym",default=2, dest="max_asym")
parser.add_argument("--min_snr",default=5, dest="max_snr")
parser.add_argument("--max_snr",default=20, dest="min_snr")
parser.add_argument("-p",help="saves candidates to csv and saves plots of targets",dest="f")
parser.add_argument("-s",help="saves candidates to csv only",default=f"candidates.csv",dest="s")


args = parser.parse_args()

def get_candidates(output_file):
    return output_file[output_file.transit_prob == 'maybeTransit']

def region_of_interest(candidates,min_asym=args.min_asym,max_asym=args.max_asym,min_snr=args.min_snr,max_snr=args.max_snr):
    candidates_in_region = candidates[(candidates.asym_score >= min_asym) & (candidates.asym_score <= max_asym) & (candidates['signal/noise'] >= min_snr) & (candidates['signal/noise'] <= max_snr)]
    return candidates_in_region

if __name__ == '__main__':
    print("converting output file...")
    try:
        sector_data = get_output(args.file[0])
        print("success! now filtering potential candidates.
    except:
        print("failed. please make sure output file is in the right format.")
    
    candidates = get_candidates(sector_data)
    to_inspect = region_of_interest(candidates)
    try:
        os.makedirs("candidates")  # make directory plot if it doesn't exist
    except FileExistsError:
        pass
    to_inspect.csv(args.s)
    print("candidates in region obtained. saved into 'candidates' directory."
