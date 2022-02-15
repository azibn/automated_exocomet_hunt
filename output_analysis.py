import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as patches


parser = argparse.ArgumentParser(description='Analysing the output files')
parser.add_argument(help='target directory(s)',nargs='1',dest='path')


def import_output(file):
    with open('output.txt') as f:
        lines = f.readlines()
    lc_lists = [word for line in lines for word in line.split()]
    lc_lists = [lc_lists[i:i+10] for i in range(0, len(lc_lists), 10)]  
    cols = ['file','signal','signal/noise','time','asym_score','width1','width2','duration','depth','transit_prob']
    df = pd.DataFrame(data=lc_lists,columns=cols)
    df[cols[1:-1]] = df[cols[1:-1]].astype('float32')
    return df
