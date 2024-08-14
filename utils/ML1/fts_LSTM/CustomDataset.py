import torch
from torch.utils.data import  Dataset
import pandas as pd


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
        self.data = self.data.dropna()
        self.device = device
        self.sequence_length = sequence_length
        self.length = len(self.data) - self.sequence_length
        self.X = torch.tensor(self.data[self.Xcolumns].values, device=self.device,dtype = torch.float32)
        self.y = torch.tensor(self.data[self.ycolumns].values, device=self.device,dtype = torch.float32)
        self.mode = mode
        if self.mode == "train":
            X_max = self.X.max(dim=0).values
            X_min = self.X.min(dim=0).values
            y_max = self.y.max(dim=0).values
            y_min = self.y.min(dim=0).values
            # TODO:Avoid division by zero 
            # X_max = [X_max[i] + 1 if X_max[i] == X_min[i] else X_max[i] for i in range(len(X_max))]
            # y_max = [y_max[i] + 1 if y_max[i] == y_min[i] else y_max[i] for i in range(len(y_max))]
            #save the mean and std to file
            torch.save(X_max, "models/X_max.pt")
            torch.save(X_min, "models/X_min.pt")
            torch.save(y_max, "models/y_max.pt")
            torch.save(y_min, "models/y_min.pt")
        elif self.mode == "test":
            X_max = torch.load("models/X_max.pt",map_location=torch.device('cpu'))
            X_min = torch.load("models/X_min.pt",map_location=torch.device('cpu'))
            y_max = torch.load("models/y_max.pt",map_location=torch.device('cpu'))
            y_min = torch.load("models/y_min.pt",map_location=torch.device('cpu'))
        else:
            raise ValueError("mode should be either train or test")
        #normalize to -1 to 1 to be used with tanh activation function !! not for ReLU !!
        self.X = 2*((self.X - X_min) / (X_max - X_min)) - 1
        self.y = 2*((self.y - y_min) / (y_max - y_min)) - 1

    def __getitem__(self, index):
        X = self.X[index:index+self.sequence_length]
        y = self.y[index+self.sequence_length]
        #normalize
        return (X, y)

    def __len__(self):
        return self.length
