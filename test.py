import argparse
import os

parser = argparse.ArgumentParser(description="Analyse target lightcurve.")
parser.add_argument(help="Target lightcurve file", nargs=1, dest="fits_file")
parser.add_argument("-n", help="No graphical output", action="store_true")
parser.add_argument(
    "-q", help="Keep only points with SAP_QUALITY=0", action="store_true"
)


args = parser.parse_args()

if os.path.split(args.fits_file[0])[1].startswith('kplr'):
    print('hello there this is a kepler file')
else:
    print('hello there this is a tess file')