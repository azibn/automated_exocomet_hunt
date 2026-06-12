import dask.dataframe as dd
from tabulate import tabulate
import argparse

"""
This script allows you to search through the conbined DataFrame created by the search method. The conbined DataFrame is a large dataframe from `eleanor-lite-combined-v3.txt`, the TIC catalog
`tic_catalog.csv` of specified columns queried from the Warwick servers (containing information about the host star), and the lightcurve metadata, `metadata-combined.csv`, 
which contains the relevant metadata from the `eleanor-lite` data products.

Running the script will return a table of the TIC_ID that match the specified conditions, for exmaple: `python scripts/search_tic.py --tic 123456789 --conditions "Sector == 3" --columns TIC_ID Sector Tmag` 
will return a table of the TIC_ID, Sector, and Tmag of the TIC_ID 123456789 in Sector 3.

If conditions were not specified, it will return all the properties (i.e all columns from the combined DataFrame) of that TIC_ID. When the script is run, it gives you an option to search for other TIC_IDs (with the conditions you set initially).

There is the option to save the output of the DataFrame if you call --save. 

There is also the option to query multiple TIC_IDs at once by calling --tic and then entering the TIC_IDs separated by spaces, or by reading a file containing the TIC_IDs. In this case, use --tic-file instead of --tic.


"""


parser = argparse.ArgumentParser(description='Finds TIC ID(s) based on the results of the search method.')
parser.add_argument('--tic', nargs='+', type=int, help='Specify TIC_ID(s) to filter rows. Can be given as number or list')
parser.add_argument('--tic-file', type=str, help='Specify file path containing TIC_IDs')
parser.add_argument('--conditions', nargs='+', help='Specify any conditions for filtering (optional). Examples: "Sector == 19" or "(asym_score > 1.01) & (Tmag < 10)"')
parser.add_argument('--columns', nargs='+', help='Specify columns to include in the output')
parser.add_argument('--show-filepaths', action='store_true', help='By default, the filepaths of the lightcurves are hidden, so that it is easier to interpret the output table. If you wish to include the filepath column in the output, use this flag.')
parser.add_argument('--save', type=str, help='Specify file path to save the output DataFrame')
args = parser.parse_args()


class DataFrameExplorer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.dataframe = self.load_large_dataframe()

    def load_large_dataframe(self):
        return dd.read_csv(self.file_path,assume_missing=True)  # Assuming a CSV file

    def filter_dataframe(self, tic_ids, conditions, selected_columns=None, show_filepaths=False):
        # Exclude the 'path' column by default
        excluded_columns = ['path']

        # Include 'path' column if --show-filepaths is provided and no conditions are specified
        if show_filepaths and not any([tic_ids, conditions]):
            excluded_columns.remove('path')

        # Build a condition for TIC_IDs if specified
        tic_condition = f'TIC_ID in {tuple(tic_ids)}' if tic_ids else None

        # Combine TIC_ID condition with user-specified conditions
        all_conditions = [tic_condition] + conditions if any([tic_condition, conditions]) else []

        # Apply the conditions to filter the DataFrame
        filtered_df = self.dataframe.query(' and '.join(filter(None, all_conditions)))

        # Select columns if specified
        if selected_columns:
            # Include 'path' column if specified in selected_columns
            if 'path' in selected_columns and not show_filepaths:
                selected_columns.remove('path')
            filtered_df = filtered_df[selected_columns]

        return filtered_df

    def save_dataframe(self, dataframe, file_path, file_format='csv'):
        if file_format == 'csv':
            dataframe.to_csv(file_path, index=False, single_file=True)  # Use single_file option for Dask
        elif file_format == 'excel':
            dataframe.to_excel(file_path, index=False, single_file=True)
        elif file_format == 'parquet':
            dataframe.to_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

def main():

    print("Loading DataFrame...")
    explorer = DataFrameExplorer("extreme-df.csv")  
    while True:
        try:
            tic_ids = args.tic if args.tic else []

            # If --tic-file is provided, read TIC IDs from the file
            if args.tic_file:
                with open(args.tic_file, 'r') as file:
                    tic_ids += [int(line.strip()) for line in file.readlines()]

            conditions = args.conditions if args.conditions else []
            selected_columns = args.columns if args.columns else None

            # If no TIC IDs specified, show warning
            if not tic_ids:
                print("Warning: No TIC_IDs specified. Conditions will be applied to the entire DataFrame.")

            filtered_dataframe = explorer.filter_dataframe(
                tic_ids, conditions, selected_columns, show_filepaths=args.show_filepaths
            )

            print(tabulate(filtered_dataframe.compute(), headers='keys', tablefmt='pipe', showindex=False,floatfmt='.4f'))

            if args.save is not None:
                explorer.save_dataframe(filtered_dataframe.compute(), args.save, file_format='csv')  # You can change the file format if needed

            # Ask the user if they want to search for other IDs
            more_ids = input("Do you want to search for other TIC_IDs? (y/n): ").lower()
            if more_ids != 'y' and more_ids != 'yes':
                print("Exiting script...")
                break  # Exit the loop if the user doesn't want to search for more IDs

            # Prompt the user to enter additional TIC_IDs
            additional_tic_ids = input("Enter additional TIC_ID(s) separated by space: ").split()
            args.tic = list(map(int, additional_tic_ids))

        except ValueError as e:
            error_message = str(e)
            if "invalid literal for int()" in error_message:
                print("Error: TIC ID failed. Please enter valid TIC_ID(s).")
            else:
                print(f"Error: {error_message}")

if __name__ == "__main__":
    main()
