import torch
from torch.utils.data import DataLoader
from CustomDataset import CustomDatasetFromCSV
from matplotlib import pyplot as plt
from tqdm import tqdm

# To denormalize the predictions
y_max = torch.load('models/y_max.pt', map_location=torch.device('cpu'))
y_min = torch.load('models/y_min.pt', map_location=torch.device('cpu'))
dataset = CustomDatasetFromCSV(csv_path='data/validate.csv', device="cpu", mode="test")
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

# Plot only selected components 0, 2, and 4 in a 1x3 grid
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

components = ['Fx', 'Fz', 'Ty']
selected_indices = [0, 2, 4]

for i, idx in enumerate(selected_indices):
    axes[i].plot(y_truth[:, idx], label="truth")
    axes[i].plot(pred_list[:, idx], label="prediction")
    axes[i].set_title(components[i])
    axes[i].set_xlabel("Index")
    if idx == 4:  # For Ty
        axes[i].set_ylabel("Nm")
    else:  # For Fx, Fz
        axes[i].set_ylabel("N")

axes[0].legend()

plt.tight_layout()
plt.savefig("plots/test.png")
plt.close()
