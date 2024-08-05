import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size,output_size)
        self.activation = nn.Tanh()
        
    def forward(self, x):
        out, _ = self.gru(x)  
        out = out[:, -1, :]
        out = self.activation(self.fc(out))
        return out

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size,output_size)
        
        self.activation = nn.Tanh()
        
    def forward(self, x):
        out, _ = self.lstm(x)  
        out = out[:, -1, :]
        out = self.activation(self.fc1(out))
        return out

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True) # activation: tanh
        self.fc1 = nn.Linear(hidden_size,256)
        self.fc2 = nn.Linear(256,output_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(p=0.0)
        
        
    def forward(self, x):
        out, _ = self.rnn(x)  
        out = out[:, -1, :]
        out = self.activation(self.fc1(out))
        out = self.dropout(out)
        out = self.activation(self.fc2(out))
        return out

class NN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        nn.init.xavier_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.fc2.weight)
        self.fc3 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.fc3.weight)
        self.fc4 = nn.Linear(256, output_size)
        nn.init.xavier_normal_(self.fc4.weight)
        self.activation = nn.Tanh()
        self.output_activation = nn.Tanh()
        self.dropout = nn.Dropout(p=0.25)
        
    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.dropout(out)
        out = self.activation(self.fc2(out))
        out = self.dropout(out)
        out = self.activation(self.fc3(out))
        out = self.dropout(out)
        out = self.output_activation(self.fc4(out))
        return out
