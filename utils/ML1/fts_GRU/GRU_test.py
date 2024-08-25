import torch
from torch.utils.data import DataLoader
from CustomDataset import CustomDatasetFromCSV
from matplotlib import pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats
import warnings
import numpy as np
warnings.filterwarnings('ignore')
set_matplotlib_formats('pdf', 'png')

plt.rcParams['savefig.dpi'] = 75
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 10, 6
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 14

plt.rcParams['text.usetex'] = True
plt.rcParams['font.serif'] = "DejaVu Serif"
plt.rcParams['font.family'] = "serif"

# Additional configurations for grey background and grid
plt.rcParams['axes.facecolor'] = '#E5E5E5'  # Light grey background
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.color'] = '#FFFFFF'  # White grid lines
plt.rcParams['grid.linestyle'] = '-'    # Solid grid lines
plt.rcParams['grid.linewidth'] = 0.5    # Thin grid lines

sequence_length = 5

# To denormalize the predictions
y_max = torch.load('models/y_max.pt', map_location=torch.device('cpu'))
y_min = torch.load('models/y_min.pt', map_location=torch.device('cpu'))

dataset = CustomDatasetFromCSV(csv_path='data/validate.csv', sequence_length=sequence_length, device="cpu", mode="test")
test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

model = torch.jit.load('models/model.pt', map_location=torch.device('cpu'))
model.eval()

pred_list = []
with torch.no_grad():
    for i, (X, y) in enumerate(tqdm(test_dataloader)):
        pred = model(X)
        pred = (pred + 1) / 2 * (y_max - y_min) + y_min
        pred_list.append(pred)

# Denormalize the data
y_truth = ((test_dataloader.dataset.y + 1) / 2 * (y_max - y_min) + y_min).cpu().numpy()
pred_list = torch.cat(pred_list).cpu().numpy()
y_truth = y_truth[5:]

components = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
selected_indices = [0, 2, 4]

rms_errors = []
for idx in selected_indices:
    rms_error = np.sqrt(np.mean((y_truth[:, idx] - pred_list[:, idx]) ** 2))
    rms_errors.append(rms_error)

# Plot the selected components in a 1x3 grid
fig, axes = plt.subplots(1, 3, figsize=(15, 5))



for i, idx in enumerate(selected_indices):
    axes[i].plot(y_truth[:, idx], label="truth")
    axes[i].plot(pred_list[:, idx], label="prediction")
    axes[i].set_title(components[idx])
    axes[i].set_xlabel("index")
    if idx == 4:
        axes[i].set_ylabel("Nm")
    else:
        axes[i].set_ylabel("N")
    
    # Add RMS error below each plot
    axes[i].text(0.5, -0.30, f'RMS Error: {rms_errors[i]:.4f}', 
                 horizontalalignment='center', 
                 verticalalignment='center', 
                 transform=axes[i].transAxes)

axes[0].legend()
fig.suptitle("GRU")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("plots/test.png")
plt.show()
