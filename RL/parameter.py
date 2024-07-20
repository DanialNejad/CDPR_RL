import json
import torch
import os
from stable_baselines3 import DDPG

# Load the model
model = DDPG.load("/media/danial/8034D28D34D28596/Projects/Kamal_RL/RL/Models/DDPG_cable_control_circle100new.zip")

# Extract hyperparameters and network parameters
hyperparameters = model.get_parameters()

# Function to convert tensors to lists
def convert_to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    else:
        return obj

# Convert all tensors in the hyperparameters to lists
serializable_hyperparameters = convert_to_serializable(hyperparameters)

# Save hyperparameters to a JSON file
with open('ddpg_hyperparameters.json', 'w') as file:
    json.dump(serializable_hyperparameters, file)
