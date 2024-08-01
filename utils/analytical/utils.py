import pandas as pd
import csv
import numpy as np
import math
from scipy.signal import butter, filtfilt
import json
import sys
import os
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from tqdm.notebook import tqdm

count_to_deg = 360/(2**24)
radian_to_deg = 180/np.pi
axis = 3
gear_ratio = 101
offset_m = -462313317
offset_l = 1144971
count_to_deg = 360/(2**24)
radian_to_deg = 180/np.pi

class FrictionEstimator:
    SIGN_VELOCITY_THRESHOLD = 0.001

    def __init__(self, robot_name, robot_dof_size, config_path):
        self.robot_name = robot_name
        self.robot_dof_size = robot_dof_size
        self.config_path = config_path
        self.configure(robot_name, robot_dof_size,config_path)

    def configure(self, robot_name, robot_dof_size,config_path):
        self.robot_dof_size = robot_dof_size
        self.robot_name = robot_name

        self.f_static = np.zeros(robot_dof_size)
        self.f_vel = np.zeros(robot_dof_size)
        self.v_static = np.zeros(robot_dof_size)
        self.delta_v = np.zeros(robot_dof_size)
        return self.read_friction_parameters_from_file(config_path)
    

    def read_friction_parameters_from_file(self, file_name):
        input_file = file_name
        #print(f"[FrictionEstimator] reading parameters from {input_file}")

        try:
            with open(input_file, 'r') as ifs:
                obj = json.load(ifs)

            for i in range(self.robot_dof_size):
                self.f_vel[i] = obj["motor_friction_params"][1][i]
                self.f_static[i] = obj["motor_friction_params"][2][i]
                self.v_static[i] = obj["motor_friction_params"][3][i]
                self.delta_v[i] = obj["motor_friction_params"][4][i]

            # print(f"[Friction] f_vel_ {self.f_vel}")
            # print(f"[Friction] f_static_ {self.f_static}")
            # print(f"[Friction] v_static_ {self.v_static}")
            # print(f"[Friction] delta_v_ {self.delta_v}")

            return True
        except IOError:
            print(f"[Friction] Cannot open config file {file_name}")
            return False
    def compute_sign_with_threshold(self, joint_velocity, threshold):
        if joint_velocity > threshold:
            return 1.0
        elif joint_velocity < -threshold:
            return -1.0
        else:
            return 0.0

    def get_friction_lp_matrix(self, joint_velocities):
        lp_matrix = np.zeros((self.robot_dof_size, 2 * self.robot_dof_size))

        for i in range(self.robot_dof_size):
            lp_matrix[i, 2 * i] = self.compute_sign_with_threshold(joint_velocities[i], self.SIGN_VELOCITY_THRESHOLD)
            lp_matrix[i, 2 * i + 1] = self.compute_sign_with_threshold(joint_velocities[i], self.SIGN_VELOCITY_THRESHOLD) * (self.delta_v[i] * abs(joint_velocities[i]) +np.tanh(self.v_static[i] * abs(joint_velocities[i])))

        return lp_matrix

    def get_friction_lp_vector(self):
        friction_lin_param_vector = np.zeros(2 * self.robot_dof_size)

        for i in range(self.robot_dof_size):
            friction_lin_param_vector[2 * i] = self.f_static[i]
            friction_lin_param_vector[2 * i + 1] = self.f_vel[i]

        return friction_lin_param_vector

    def compute_friction_torques(self, joint_velocities):
        return np.dot(self.get_friction_lp_matrix(joint_velocities), self.get_friction_lp_vector())

def resample2(df,num_points):
    df = df.reset_index(drop=True)
    df = df.sample(n = num_points).sort_index()
    df = df.reset_index(drop=True)
    return df

def number_of_motor_turns(mot_enc,load_enc,offset_m,offset_l,gear_ratio = 101):
    return ((load_enc - offset_l)*gear_ratio + offset_m)//2**24 - mot_enc//2**24
    
def multiturn_compensation(df,offset_m,offset_l,gear_ratio=101):
    
    
	n = number_of_motor_turns(df.encoder_motorinc_3[0],df.encoder_loadinc_3[0],offset_m,offset_l,gear_ratio)
	if n == 0:
	    #print("no offset compensation required")
	    return df
	else:
	    #print("compensating offset")
	    df.encoder_motorinc_3 += n * 2**24
	    return df


def offset_compensation(df,offset_m,offset_l):
    #reduce offsets from df motorenc and load enc
    df.encoder_motorinc_3 -= offset_m
    df.encoder_loadinc_3 -= offset_l
    return df

def filter(data,fc):
    # Low pass filter
    fs = 1 / (2e-3)  # Sampling frequency (Hz), sampled every 2 ms 

    # Normalized cutoff frequency with respect to the Nyquist frequency
    nyquist = fs / 2
    Wn = fc / nyquist

    # Butterworth low-pass filter
    order = 4
    b, a = butter(order, Wn, 'low')

    # Apply the filter to signals using filtfilt
    data["filtered_motor_enc"] = filtfilt(b, a, data.encoder_motorinc_3)
    data["filtered_load_enc"] = filtfilt(b, a,  data.encoder_loadinc_3)
    data["filtered_joint_velocity"] = filtfilt(b, a, data.joint_velocity_3)
    return data

def model_footprint(data, gear_ratio = 101):
    mot_enc = data.filtered_motor_enc.values*count_to_deg
    load_enc = data.filtered_load_enc.values*count_to_deg 

    # Calculating Error from drive
    error = mot_enc/ gear_ratio - load_enc

    # Calculating the fourier_Constants of the Fourier Series (sine and cosine terms)
    fourier_harmonic = 30
    omega_m = [1, 2, 1-1/101, 2-2/101, 4-4/101, 4, 6-6/101, 6, 8-8/101, 10-10/101]
    size_high_freq = len(omega_m)
    size_dataset = len(mot_enc); 
    fourier_Constants = np.zeros((size_dataset , 2 * (fourier_harmonic + size_high_freq) ))
    for i in range(size_dataset):

        for j in range(0, fourier_harmonic ):
            fourier_Constants[i, 2 * j ] = math.cos(j * math.pi * (1/180) * load_enc[i])
            fourier_Constants[i, 2 * j + 1] = math.sin(j * math.pi * (1/180) * load_enc[i])

        l = 0
        for k in range(fourier_harmonic , fourier_harmonic + size_high_freq):
            fourier_Constants[i, 2 * k ] = math.cos(mot_enc[i] * omega_m[l] * math.pi * (1/180))
            fourier_Constants[i, 2 * k+1] = math.sin(mot_enc[i] * omega_m[l] * math.pi * (1/180))
            l = l + 1

        
    Constants_inv =np.linalg.pinv(fourier_Constants) 
    coeff = Constants_inv @ error


    err_calculated = fourier_Constants @coeff 

    residualerror = error - err_calculated
    rms_value_residual = np.sqrt(np.mean(np.square(residualerror)))
    #print("RMS value residual error:", rms_value_residual)
    rms_value_actual = np.sqrt(np.mean(np.square(error)))
    #print("RMS value actual error:", rms_value_actual)
    if(rms_value_residual < rms_value_actual):
       pass
        # print('The Model has reduced the error')
    return coeff

def footprint_error(coeff,mot_enc,load_enc):
    mot_enc = mot_enc*count_to_deg
    load_enc = load_enc*count_to_deg
    fourier_harmonic = 30
    omega_m = [1, 2, 1-1/101, 2-2/101, 4-4/101, 4, 6-6/101, 6, 8-8/101, 10-10/101]
    size_high_freq = len(omega_m)
    error = 0
    for j in range(0, fourier_harmonic ):
        error += coeff[2 * j] * math.cos(j * math.pi * (1/180) * load_enc)
        error += coeff[2 * j + 1] * math.sin(j * math.pi * (1/180) * load_enc)

    l = 0
    for k in range(fourier_harmonic , fourier_harmonic + size_high_freq):
        error += coeff[ 2 * k ] * math.cos(mot_enc * omega_m[l] * math.pi * (1/180))
        error += coeff[2 * k+1] * math.sin(mot_enc * omega_m[l] * math.pi * (1/180))
        l = l + 1
    return error

def friction_model(v,a1,a2,a3,b1,b2,b3):
    #expects v in degrees/s
    return a1*(np.tanh(b1*v)-np.tanh(b2*v))+a2*np.tanh(b3*v)+a3*v

def remove_acceleration(df):
    df_p = df[df['joint_velocity_3'] > 0]
    df_n = df[df['joint_velocity_3'] < 0]
    #remove datapoints with velocity spike
    p_vel_avg = df_p.joint_velocity_3.mean()*radian_to_deg
    bounds = 0.7
    n_vel_avg = df_n.joint_velocity_4.mean()*radian_to_deg
    mask = ((df['joint_velocity_3'] >= (n_vel_avg - bounds)/radian_to_deg) & (df['joint_velocity_3'] <= (n_vel_avg + bounds)/radian_to_deg)) |  ((df['joint_velocity_3'] >= (p_vel_avg - bounds)/radian_to_deg) & (df['joint_velocity_3'] <= (p_vel_avg + bounds)/radian_to_deg))
    #print number of poitns removed
    print("removed",len(df)-len(df[mask]),"datapoints")
    #reset index
    
    return df[mask]

def encoder_error_to_torque(encoder_error):
    #encoder error in counts
    #torque in Nm

    encoder_error = encoder_error*count_to_deg/radian_to_deg #arcmin
    K1 = 31000 
    K2 = 50000 
    K3 = 57000 
    T1 = 14 #Nm
    T2 = 48 #Nm
    THETA1 = T1/K1
    THETA2 = THETA1 + (T2-T1)/K2 
    if encoder_error < THETA1:
        return encoder_error*K1
    elif encoder_error < THETA2:
        return T1 + (encoder_error - THETA1)*K2
    else:
        return T2 + (encoder_error - THETA2)*K3
    
def torque_to_encoder_error(torque):
    #encoder error in counts
    #torque in Nm
    K1 = 31000 
    K2 = 50000 
    K3 = 57000 
    T1 = 14 #Nm
    T2 = 48 #Nm
    THETA1 = T1/K1
    THETA2 = THETA1 + (T2-T1)/K2 
    if torque < T1:
        return (torque/K1)/count_to_deg*radian_to_deg
    elif torque < THETA2:
        return  ((torque - T1)/K2 + THETA1)/count_to_deg*radian_to_deg
    else:
        return ((torque - T2)/K3 + THETA2)/count_to_deg*radian_to_deg
    
def controller_friction_estimate(velocity):

    friction_estimator = FrictionEstimator("lara8", 6, "./utils/analytical/friction_parameters_lara8.json")
    return friction_estimator.compute_friction_torques([0.0, 0.0, 0.0, velocity, 0.0, 0.0, 0.0])[3]

def compute_sign(velocity, threshold):
    friction_estimator =FrictionEstimator("lara8", 6, "./utils/analytical/friction_parameters_lara8.json")
    return friction_estimator.compute_sign_with_threshold(velocity, threshold)

def preprocess(base_dir = '/mnt/data/analytical/lara8_90deg_0load/',output_dir = '/mnt/data/analytical/preprocessed/'):

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define the columns you want to read
    columns_to_read = ["encoder_motorinc_3" , "encoder_loadinc_3", "joint_velocity_3","joint_torque_current_3","target_joint_torque_3","joint_angles_3",
                "fts_reading_1","fts_reading_2","fts_reading_3","fts_reading_4","fts_reading_5","fts_reading_6"] 

    # Get the list of subdirectories
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for subdir in subdirs:
    # Define the paths to the positive and negative CSV files
        pos_file = os.path.join(base_dir, subdir, 'positive.csv')
        neg_file = os.path.join(base_dir, subdir, 'negative.csv')

        # Check if both files exist
        if os.path.exists(pos_file) and os.path.exists(neg_file):
            try:
            # Create an inner progress bar for reading and concatenating files
                # Read the CSV files with specified columns
                pos_df = pd.read_csv(pos_file, encoding='ISO-8859-1', usecols=columns_to_read)
                pos_df = pos_df[200:-200]
                neg_df = pd.read_csv(neg_file, encoding='ISO-8859-1', usecols=columns_to_read)
                neg_df = neg_df[200:-200]
                # Concatenate the dataframes
                combined_df = pd.concat([pos_df, neg_df])
                # Replace 'inf' values with 'NaN'
                combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)

                # Remove rows with 'NaN' values
                combined_df.dropna(inplace=True)

                # Save the combined dataframe to a new CSV file in the data folder
                combined_df.to_csv(os.path.join(output_dir, f'{subdir}.csv'), index=False)
            except pd.errors.ParserError as e:
                print(f"Error parsing files in directory {subdir}: {e}")
            except Exception as e:
                print(f"Unexpected error with files in directory {subdir}: {e}")

	# Define the directory containing the combined CSV files
    directory = '/mnt/data/analytical/preprocessed/'

	# List all CSV files in the directory that end with '.csv'
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

	# Initialize an empty dictionary to store the dataframes
    data = {}

	# Loop over each CSV file
    for file in tqdm(csv_files, desc="preprocessing", colour= "green"):
        try:
            # Read each CSV file into a dataframe
            df = pd.read_csv(os.path.join(directory, file))
            # Store the dataframe in the dictionary
            # Use the filename (without .csv) as the key
            key = os.path.splitext(file)[0]
            data[key] = df
            #print(f"Loaded {key}")
        except Exception as e:
            print(f"Error loading {file}: {e}")

    min_length = None
    min_length_key = None
    for key in data:
        df = data[key]
        length = len(df)
        if min_length is None or length < min_length:
            min_length = length
            min_length_key = key
	#print("The smallest dataframe is ", key, " and has ", min_length, " rows.")
	#resample to create equal length dataframes
    for key in data:
        data[key]= resample2(data[key],min_length)
    for key in data:
        data[key]= multiturn_compensation(data[key],offset_m,offset_l)
    #offset compensation
    for key in data:
        data[key]= offset_compensation(data[key],offset_m,offset_l)
    return data

