import argparse
import os
import sys
import pandas as pd
import subprocess

"""A version of tplfplotter which can run multiple targets in just one command."""


data = pd.read_csv('candidates/272-candidates.csv')
data = data[(data.asteroid == 0) & (data.tags2 == 'red')]
TIC_ID = data['TIC_ID'].to_list()
Sector = data['Sector'].to_list()


for tic, sector in zip(TIC_ID, Sector):
    subprocess.run(['python', 'tpfplotter/tpfplotter.py', str(tic), '--sector', str(sector),'--maglim','13']) # adjust arguments as needed