
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import  Dataset
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

count_to_deg = 360/(2**24)
radian_to_deg = 180/np.pi

motor_encoder_resolution = 2**24

def floor_division(a, b):
    return (a // b) if a * b >= 0 else -(-a // b)

def number_of_motor_turns(mot_enc,load_enc,offset_m,offset_l,gear_ratio):
    return (load_enc - offset_l)*gear_ratio //2**24 + (+offset_m- mot_enc)//motor_encoder_resolution
    #return floor_division((load_enc - offset_l)*gear_ratio,2**24 )+ floor_division((+offset_m- mot_enc),motor_encoder_resolution)

def multiturn_compensation(df):
    offset_m_list=[-438404 	,-443684,-27785,559160,418900 ,-187955]
    offset_l_list =[ 5.110719e+06,4.178252e+06, 1.146653e+06,4.632860e+06, 1.034963e+07, 1.999400e+06]
    gear_ratio_list=[121,121,101,101,101,101]
    encoder_columns = [
        'encoder_motorinc_1', 'encoder_motorinc_2', 'encoder_motorinc_3',
        'encoder_motorinc_4', 'encoder_motorinc_5', 'encoder_motorinc_6'
    ]
    load_columns = [
        'encoder_loadinc_1', 'encoder_loadinc_2', 'encoder_loadinc_3',
        'encoder_loadinc_4', 'encoder_loadinc_5', 'encoder_loadinc_6'
    ]
    
    for motor_col, load_col, offset_m, offset_l , gear_ratio in zip(encoder_columns, load_columns, offset_m_list, offset_l_list, gear_ratio_list):
        n = number_of_motor_turns(df[motor_col][0], df[load_col][0], offset_m, offset_l, gear_ratio)
    
        #n = calculate_turns(df[motor_col][0], df[load_col][0], offset_m, offset_l, gear_ratio)
        if n != 0:
            df[motor_col] += n * motor_encoder_resolution
            #df[motor_col] +=int(n) * motor_encoder_resolution
            #print(n)
            
           # print("no compensation required")
    return df



def undo_elfin_offset_compensation(data):
    gear_ratio_list=[121,121,101,101,101,101]
    for i in range(1,7):
        if (data["encoder_loadinc_"+str(i)].max()) > 2**24:
            #print("needs positive load side compensation @joint", i)
            data["encoder_loadinc_"+str(i)] -= 2**24
            data["encoder_motorinc_"+str(i)] -= gear_ratio_list[i-1]*2**24
        if (data["encoder_loadinc_"+str(i)].min()) < -2**24:
            #print("needs negative load side compensation @joint", i)
            data["encoder_loadinc_"+str(i)] += 2**24
            data["encoder_motorinc_"+str(i)] += gear_ratio_list[i-1]*2**24
    return data

class CustomDatasetFromCSV(Dataset):
    
    def __init__(self, csv_path, sequence_length=116,device="cpu",mode="train"):
        """
        Args:
            csv_path (string): path to csv file
            sequence_length: length of the input sequence
            device : device to store the tensors
            mode : train or test
        """
        self.Xcolumns = ["encoder_motorinc_1" , "encoder_loadinc_1", "joint_velocity_1","joint_torque_current_1","target_joint_torque_1",
                    "encoder_motorinc_2" , "encoder_loadinc_2", "joint_velocity_2","joint_torque_current_2","target_joint_torque_2",
                    "encoder_motorinc_3" , "encoder_loadinc_3", "joint_velocity_3","joint_torque_current_3","target_joint_torque_3",
                   "encoder_motorinc_4" , "encoder_loadinc_4", "joint_velocity_4","joint_torque_current_4","target_joint_torque_4",
                   "encoder_motorinc_5" , "encoder_loadinc_5", "joint_velocity_5","joint_torque_current_5","target_joint_torque_5",
                   "encoder_motorinc_6" , "encoder_loadinc_6", "joint_velocity_6","joint_torque_current_6","target_joint_torque_6"]
        self.ycolumns = ["fts_reading_1","fts_reading_2","fts_reading_3","fts_reading_4","fts_reading_5","fts_reading_6"]
        self.data = pd.read_csv(csv_path, usecols = self.Xcolumns+self.ycolumns)
        self.mode = mode
        if self.mode == "test":
            self.data = undo_elfin_offset_compensation(self.data)
            self.data = multiturn_compensation(self.data)
        #create one more coulmn named difference for difference of encoder motor and encoder load
        for i in range(1, 7):
            self.data[f"encoder_difference_{i}"] = 101*self.data[f"encoder_loadinc_{i}"]-self.data[f"encoder_motorinc_{i}"]
            self.Xcolumns.append(f"encoder_difference_{i}")
        self.data = self.data.dropna()
        self.device = device
        self.sequence_length = sequence_length
        self.length = len(self.data) - self.sequence_length
        self.X = torch.tensor(self.data[self.Xcolumns].values, device=self.device,dtype = torch.float32)
        self.y = torch.tensor(self.data[self.ycolumns].values, device=self.device,dtype = torch.float32)
        
        if self.mode == "train":
            X_max = self.X.max(dim=0).values
            X_min = self.X.min(dim=0).values
            y_max = self.y.max(dim=0).values
            y_min = self.y.min(dim=0).values
            # TODO:Avoid division by zero 
            # X_max = [X_max[i] + 1 if X_max[i] == X_min[i] else X_max[i] for i in range(len(X_max))]
            # y_max = [y_max[i] + 1 if y_max[i] == y_min[i] else y_max[i] for i in range(len(y_max))]
            #save the mean and std to file
            torch.save(X_max, "utils/ML2/models/X_max.pt")
            torch.save(X_min, "utils/ML2/models/X_min.pt")
            torch.save(y_max, "utils/ML2/models/y_max.pt")
            torch.save(y_min, "utils/ML2/models/y_min.pt")
        elif self.mode == "test":
            X_max = torch.load("utils/ML2/models/X_max.pt",map_location=torch.device('cpu'))
            X_min = torch.load("utils/ML2/models/X_min.pt",map_location=torch.device('cpu'))
            y_max = torch.load("utils/ML2/models/y_max.pt",map_location=torch.device('cpu'))
            y_min = torch.load("utils/ML2/models/y_min.pt",map_location=torch.device('cpu'))
        else:
            raise ValueError("mode should be either train or test")
        #normalize to -1 to 1 to be used with tanh activation function !! not for ReLU !!
        self.X = 2*((self.X - X_min) / (X_max - X_min)) - 1
        self.y = 2*((self.y - y_min) / (y_max - y_min)) - 1

    def __getitem__(self, index):
        X = self.X[index:index+self.sequence_length]
        y = self.y[index+self.sequence_length]
        return (X, y)

    def __len__(self):
        return self.length


def dh_transform(a, alpha, d, theta):
    """
    Compute Denavit-Hartenberg transformation matrix.
    """
    return np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha),  np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
        [np.sin(theta),  np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
        [0,             np.sin(alpha),                 np.cos(alpha),                 d],
        [0,             0,                             0,                             1]
    ])
dh_params = [
    (0, -np.pi/2, 0.26, 0),           # Link 1
    (0.48, 0, 0, -np.pi/2),       # Link 2
    (0, np.pi/2, 0, np.pi/2),        # Link 3
    (0, -np.pi/2, 0.7, 0),          # Link 4
    (0, np.pi/2, 0, 0),              # Link 5
    (0, 0, 0.1735, 0)                 # Link 6
]

def compute_geometric_jacobian(joint_angles,dh_params = dh_params):
    """
    Compute the geometric Jacobian for a robot given DH parameters and joint angles.

    :param dh_params: List of DH parameters [(a, alpha, d, theta), ...]
    :param joint_angles: List of joint angles [theta1, theta2, ..., thetaN]
    :return: Geometric Jacobian matrix (6 x N)
    """
    num_joints = len(joint_angles)
    T = np.eye(4)
    transformations = [T.copy()]
    
    # Compute forward kinematics to get transformations
    for i, (a, alpha, d, theta) in enumerate(dh_params):
        theta += joint_angles[i]  # Update theta with joint angle
        T_i = dh_transform(a, alpha, d, theta)
        T = np.dot(T, T_i)
        transformations.append(T.copy())
    
    # Initialize Jacobian matrix
    jacobian = np.zeros((6, num_joints))
    
    # End-effector position
    T_0_n = transformations[-1]
    o_n = T_0_n[:3, 3]

    # Compute each column of the Jacobian
    for i in range(num_joints):
        T_0_i = transformations[i]
        o_i = T_0_i[:3, 3]
        z_i = T_0_i[:3, 2]

        jacobian[:3, i] = np.cross(z_i, (o_n - o_i))  # Linear part
        jacobian[3:, i] = z_i  # Angular part

    return jacobian

