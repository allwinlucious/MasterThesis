#!/usr/bin/env python
# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from captum.attr import IntegratedGradients
from torch.utils.data import DataLoader
from CustomDataset import CustomDatasetFromCSV
import matplotlib.pyplot as plt

sequence_length = 5

torch.manual_seed(1234)
np.random.seed(1234)

# To denormalize the predictions
y_max = torch.load('models/y_max.pt', map_location=torch.device('cpu'))
y_min = torch.load('models/y_min.pt', map_location=torch.device('cpu'))

dataset = CustomDatasetFromCSV(csv_path='data/validate.csv', sequence_length=sequence_length, device="cpu", mode="test")
test_dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

model = torch.jit.load('models/model.pt', map_location=torch.device('cpu'))
model.train()

complete_data = test_dataloader.dataset

# Extract all data and targets
X_test = torch.stack([complete_data[i][0] for i in range(len(complete_data))])
y_test = torch.stack([complete_data[i][1] for i in range(len(complete_data))])

# Initialize Integrated Gradients
ig = IntegratedGradients(model)

# Define the target indices and titles
targets = [0, 2, 4]
target_names = ['Fx', 'Fz', 'Ty']

# Prepare for plotting
feature_names = ["encoder_motorinc_3", "encoder_loadinc_3", "joint_velocity_3", "joint_torque_current_3", "target_joint_torque_3", "joint_angles_3","enocder_difference"]
x_axis_data = np.arange(len(feature_names))
x_axis_data_labels = list(map(lambda idx: feature_names[idx], x_axis_data))

# Visualization
fig, axes = plt.subplots(3, 1, figsize=(30, 15))

FONT_SIZE = 16
plt.rc('font', size=FONT_SIZE)
plt.rc('axes', titlesize=FONT_SIZE)
plt.rc('axes', labelsize=FONT_SIZE)
plt.rc('legend', fontsize=FONT_SIZE-4)

for i, target in enumerate(targets):
    ig_attr_test = ig.attribute(X_test, n_steps=10, target=target)
    
    # Sum attributions across all samples and time steps, then normalize
    ig_attr_test_sum = ig_attr_test.sum(dim=[0, 1]).detach().cpu().numpy()
    ig_attr_test_norm_sum = ig_attr_test_sum / np.linalg.norm(ig_attr_test_sum, ord=1)
    
    ax = axes[i]
    ax.bar(x_axis_data, ig_attr_test_norm_sum, width=0.4, align='center', alpha=0.8, color='#eb5e7c')
    ax.legend(target_names[i] ,loc='upper left')
    ax.set_ylabel('Normalized Attributions')
    ax.set_xticks(x_axis_data)
    ax.set_xticklabels(x_axis_data_labels)
    #ax.autoscale_view()

plt.tight_layout()
plt.savefig("plots/feature_importance_targets.png")
plt.show()

# Cleanup
del ig
torch.cuda.empty_cache()
