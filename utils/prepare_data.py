import os
import pandas as pd

# Define the base directory
base_dir = '/mnt/data/analytical/'

# Get the list of subdirectories
subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# Loop over each subdirectory
for subdir in subdirs:
    # Define the paths to the positive and negative CSV files
    pos_file = os.path.join(base_dir, subdir, 'positive.csv')
    neg_file = os.path.join(base_dir, subdir, 'negative.csv')

    # Check if both files exist
    if os.path.exists(pos_file) and os.path.exists(neg_file):
        # Read the CSV files
        pos_df = pd.read_csv(pos_file, encoding='ISO-8859-1')
        neg_df = pd.read_csv(neg_file, encoding='ISO-8859-1')

        # Concatenate the dataframes
        combined_df = pd.concat([pos_df, neg_df])

        # Save the combined dataframe to a new CSV file in the data folder
        combined_df.to_csv(os.path.join('data', f'{subdir}.csv'), index=False)

