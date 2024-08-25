

import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import  Dataset
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchview import draw_graph
from utils.ML1.model_defenitions import NN, RNN, LSTM, GRU
from PIL import Image
from IPython.display import display
from IPython.display import Image as IPImage
import io
count_to_deg = 360/(2**24)
radian_to_deg = 180/np.pi

torch.manual_seed(5)
np.random.seed(5)
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
        'encoder_motorinc_3',
        
    ]
    load_columns = [
        'encoder_loadinc_3',

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
    i = 3
    if (data["encoder_loadinc_"+str(i)].max()) > 2**24:
        #print("needs positive load side compensation @joint", i)
        data["encoder_loadinc_"+str(i)] -= 2**24
        data["encoder_motorinc_"+str(i)] -= gear_ratio_list[i-1]*2**24
    if (data["encoder_loadinc_"+str(i)].min()) < -2**24:
        #print("needs negative load side compensation @joint", i)
        data["encoder_loadinc_"+str(i)] += 2**24
        data["encoder_motorinc_"+str(i)] += gear_ratio_list[i-1]*2**24
    return data

def visualize_models():
    # Instantiate the models
    nn_model = NN(7, 6)
    rnn_model = RNN(7, 10, 2, 6)
    lstm_model = LSTM(7, 128, 1, 6)
    gru_model = GRU(7, 128, 1, 6)
    

    # Generate graphs for each model
    nn_graph = draw_graph(nn_model, input_size=(1, 7), expand_nested=False, show_shapes=True)
    rnn_graph = draw_graph(rnn_model, input_size=(1,5, 7), expand_nested=False, show_shapes=True)
    lstm_graph = draw_graph(lstm_model, input_size=(1,5, 7), expand_nested=False, show_shapes=True)
    gru_graph = draw_graph(gru_model, input_size=(1,5, 7), expand_nested=False, show_shapes=True)


    # Get the visual graphs and convert to a format suitable for Matplotlib
    nn_vis = Image.open(io.BytesIO(nn_graph.visual_graph.pipe(format='png')))
    rnn_vis = Image.open(io.BytesIO(rnn_graph.visual_graph.pipe(format='png')))
    lstm_vis = Image.open(io.BytesIO(lstm_graph.visual_graph.pipe(format='png')))
    gru_vis = Image.open(io.BytesIO(gru_graph.visual_graph.pipe(format='png')))

        # Function to resize images to the same size
    def resize_image(img, size=(400, 1200)):
        return img.resize(size, Image.Resampling.LANCZOS)

    # Resize images
    nn_vis = resize_image(nn_vis)
    rnn_vis = resize_image(rnn_vis)
    lstm_vis = resize_image(lstm_vis)
    gru_vis = resize_image(gru_vis)

    # Make a subplot with each image
    fig, axs = plt.subplots(1, 4, figsize=(40, 10))
    axs[0].imshow(nn_vis)
    axs[0].set_title("NN")
    axs[0].axis('off')

    axs[1].imshow(rnn_vis)
    axs[1].set_title("RNN")
    axs[1].axis('off')

    axs[2].imshow(lstm_vis)
    axs[2].set_title("LSTM")
    axs[2].axis('off')

    axs[3].imshow(gru_vis)
    axs[3].set_title("GRU")
    axs[3].axis('off')

    plt.show()

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
    import torch

class CustomDatasetFromCSV(Dataset):
    
    def __init__(self, csv_path, sequence_length,device="cpu",mode="train"):
        """
        Args:
            csv_path (string): path to csv file
            sequence_length: length of the input sequence
            device : device to store the tensors
            mode : train or test
        """
        self.Xcolumns = ["encoder_motorinc_3" , "encoder_loadinc_3", "joint_velocity_3","joint_torque_current_3","target_joint_torque_3","joint_angles_3"]
        self.ycolumns = ["fts_reading_1","fts_reading_2","fts_reading_3","fts_reading_4","fts_reading_5","fts_reading_6"]
        self.data = pd.read_csv(csv_path, usecols = self.Xcolumns+self.ycolumns)
        self.mode = mode
        if self.mode == "test":
            self.data = multiturn_compensation(self.data)
        else:
            pass # multiturn compensation is done in append_data.py for training data
        #create one more coulmn named difference for difference of encoder motor and encoder load
        self.data["encoder_difference_3"] = 101*self.data["encoder_loadinc_3"]-self.data["encoder_motorinc_3"]
        self.Xcolumns.append("encoder_difference_3")
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
            torch.save(X_max, "utils/ML1/models/X_max.pt")
            torch.save(X_min, "utils/ML1/models/X_min.pt")
            torch.save(y_max, "utils/ML1/models/y_max.pt")
            torch.save(y_min, "utils/ML1/models/y_min.pt")
        elif self.mode == "test":
            X_max = torch.load("utils/ML1/models/X_max.pt",map_location=torch.device('cpu'))
            X_min = torch.load("utils/ML1/models/X_min.pt",map_location=torch.device('cpu'))
            y_max = torch.load("utils/ML1/models/y_max.pt",map_location=torch.device('cpu'))
            y_min = torch.load("utils/ML1/models/y_min.pt",map_location=torch.device('cpu'))
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

def show_test_graphs():
    model_folders = ['fts_GRU', 'fts_RNN', 'fts_LSTM', 'fts_NN']  # Add any other model folders as needed
    base_path = "utils/ML1"
    for model in model_folders:
        test_folder = os.path.join(base_path, model)
        graph_path = os.path.join(test_folder, 'plots', 'test.png')
        
        if os.path.exists(graph_path):
            #print(f"Displaying graph for {model}:")
            display(IPImage(filename=graph_path))
        else:
            print(f"No graph found for {model} in {graph_path}")
def visualize_models2():
    # Instantiate the models
    nn_model = NN(7, 6)
    rnn_model = RNN(7, 10, 2, 6)
    lstm_model = LSTM(7, 128, 1, 6)
    gru_model = GRU(7, 128, 1, 6)
    
    # Generate graphs for each model
    nn_graph = draw_graph(nn_model, input_size=(1, 7), expand_nested=False, show_shapes=True)
    rnn_graph = draw_graph(rnn_model, input_size=(1,5, 7), expand_nested=False, show_shapes=True)
    lstm_graph = draw_graph(lstm_model, input_size=(1,5, 7), expand_nested=False, show_shapes=True)
    gru_graph = draw_graph(gru_model, input_size=(1,5, 7), expand_nested=False, show_shapes=True)

    # Get the visual graphs and convert to a format suitable for Matplotlib
    nn_vis = Image.open(io.BytesIO(nn_graph.visual_graph.pipe(format='png')))
    rnn_vis = Image.open(io.BytesIO(rnn_graph.visual_graph.pipe(format='png')))
    lstm_vis = Image.open(io.BytesIO(lstm_graph.visual_graph.pipe(format='png')))
    gru_vis = Image.open(io.BytesIO(gru_graph.visual_graph.pipe(format='png')))

    # Function to resize images to the same size
    def resize_image(img, size=(600, 1500)):
        return img.resize(size, Image.Resampling.LANCZOS)

    # Resize images
    nn_vis = resize_image(nn_vis)
    rnn_vis = resize_image(rnn_vis)
    lstm_vis = resize_image(lstm_vis)
    gru_vis = resize_image(gru_vis)

    # Make a subplot with 2 rows and 2 columns for each image
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    
    axs[0, 0].imshow(nn_vis)
    axs[0, 0].set_title("NN")
    axs[0, 0].axis('off')

    axs[0, 1].imshow(rnn_vis)
    axs[0, 1].set_title("RNN")
    axs[0, 1].axis('off')

    axs[1, 0].imshow(lstm_vis)
    axs[1, 0].set_title("LSTM")
    axs[1, 0].axis('off')

    axs[1, 1].imshow(gru_vis)
    axs[1, 1].set_title("GRU")
    axs[1, 1].axis('off')
    plt.savefig("subplotmodel.png")

