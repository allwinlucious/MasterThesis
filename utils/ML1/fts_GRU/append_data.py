from tqdm import tqdm
import logging
import h5py
import os
import pandas as pd
from utils import multiturn_compensation

with open('data/exceptions.log', 'w'): # clear   
    pass
logging.basicConfig(filename='data/exceptions.log', level=logging.ERROR)
all_files = [os.path.join(root, file) for root, dirs, files in os.walk("../data_fts/") for file in files if file.endswith("ive.csv")]
columns_to_read = ["encoder_motorinc_3" , "encoder_loadinc_3", "joint_velocity_3","joint_torque_current_3","target_joint_torque_3","joint_angles_3",
                   "fts_reading_1","fts_reading_2","fts_reading_3","fts_reading_4","fts_reading_5","fts_reading_6"]
first_file=True
with open('data/data.csv', 'w') as outfile:
    for file_path in tqdm(all_files, desc="Pre-processing files"):
        try:
            df = pd.read_csv(file_path, usecols=columns_to_read)
            df = multiturn_compensation(df,offset_m=-82872,offset_l=1145852)
            df.to_csv(outfile, index=False, header=first_file)
            first_file= False
        except Exception as e:
            logging.error(f"Failed to read file {file_path} due to error: {e}")
