import pandas as pd
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Combine lightcurves to one larger dataframe")

parser.add_argument(help="directory storing all smaller dataframes", default=".", nargs="+", dest="path")
parser.add_argument("-o", default=f"combined_dataframe.txt", dest="o", help="output file")

args = parser.parse_args()


folder_path = args.path[0]  # Replace with the actual folder path


final_df = pd.DataFrame()  # Initialize an empty DataFrame

for filename in tqdm(os.listdir(folder_path)):
    
    if filename.endswith('.txt') or filename.endswith('.csv'):  # Adjust the file extension as needed
        file_path = os.path.join(folder_path, filename)
        #print(f"now concatenating {file_path}")
        df = pd.read_csv(file_path, sep=",", header=None)
        final_df = pd.concat([final_df, df], ignore_index=True)


print("Saving to large dataframe now")
# Save the final concatenated DataFrame to a new file if needed
final_df.to_csv(f'{args.o}', index=False)
print(f"Done! Saved as {args.o}")