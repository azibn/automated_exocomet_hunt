from analysis_tools_cython import *
from post_processing_tools import get_output
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import matplotlib.patches as patches
import os

parser = argparse.ArgumentParser(description="getting candidates from TESS runs")
parser.add_argument(help="Target output file", nargs=1, dest="file")
parser.add_argument("--min_asym",default=1.05, dest="min_asym")
parser.add_argument("--max_asym",default=2, dest="max_asym")
parser.add_argument("--min_snr",default=-5, dest="max_snr")
parser.add_argument("--max_snr",default=-20, dest="min_snr")
parser.add_argument("-p",help="saves candidates to csv and saves plots of targets",dest="f")
parser.add_argument("-s",help="saves candidates to csv only",default=f"candidates.csv",dest="s")


args = parser.parse_args()

def get_candidates(output_file):
    return output_file[output_file.transit_prob == 'maybeTransit']

def region_of_interest(candidates,min_asym=args.min_asym,max_asym=args.max_asym,min_snr=args.min_snr,max_snr=args.max_snr):
    candidates_in_region = candidates[(candidates.asym_score >= min_asym) & (candidates.asym_score <= max_asym) & (candidates['signal/noise'] >= min_snr) & (candidates['signal/noise'] <= max_snr)]
    return candidates_in_region

def distribution_plot(data):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(data.asym_score,abs(data['signal/noise']), s=10)
    ax.set_xlim(-0, 1.9)
    ax.set_ylim(-1, 30)
    ax.set_title('SNR vs asymmetry plot',fontsize=20,color='white')
    ax.set_xlabel("$\\alpha$", fontsize=16,color='white')
    ax.set_ylabel("$S$", fontsize=16,color='white')
    rect = patches.Rectangle((1.05, 5), 2, 30, linewidth=3, edgecolor='r', facecolor='none')
    plt.show()

if __name__ == '__main__':
    print("converting output file...")
    try:
        sector_data = get_output(args.file[0])
        print("success! now filtering potential candidates.")
    except:
        print("failed. please make sure output file is in the right format.")
    
    candidates = get_candidates(sector_data)
    to_inspect = region_of_interest(candidates)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(to_inspect.asym_score,abs(to_inspect['signal/noise']), s=10)
    ax.set_xlim(-0, 1.9)
    ax.set_ylim(-1, 30)
    ax.set_title('SNR vs asymmetry plot',fontsize=20,color='white')
    ax.set_xlabel("$\\alpha$", fontsize=16,color='white')
    ax.set_ylabel("$S$", fontsize=16,color='white')
    rect = patches.Rectangle((1.05, 5), 2, 30, linewidth=3, edgecolor='r', facecolor='none')

    plt.show()


    try:
        os.makedirs("candidates")  # make directory plot if it doesn't exist
    except FileExistsError:
        pass
    #to_inspect.to_csv(args.s)
    #print("candidates in region obtained. saved into 'candidates' directory.")
