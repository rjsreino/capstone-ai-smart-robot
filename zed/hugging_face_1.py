import torch
from transformers import pipeline
from PIL import Image
import numpy as np
import os

def generate_virtual_lidar(image_path):
    # 1. Load the Depth Estimation model from Hugging Face
    # 'depth-anything-v2' is current state-of-the-art
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

    # 2. Load your image (any photo of a floor/room)
    image = Image.open(image_path)

    # 3. Predict depth
    # The output is a 'depth' image where pixel intensity = distance
    result = pipe(image)
    depth_map = result["depth"] # This is a PIL Image
    
    # 4. Convert to a Numerical Array (the "LiDAR" data)
    depth_array = np.array(depth_map)
    
    # Normalize or scale to meters (approximation)
    # In a real UGV, you'd calibrate this to actual meters
    depth_in_meters = (depth_array / 255.0) * 10.0 

    print(f"Generated a {depth_in_meters.shape} depth grid.")
    print(f"Closest point value: {np.min(depth_in_meters)}")
    
    return depth_in_meters

# Get the directory where the script is currently located
script_dir = os.path.dirname(os.path.abspath(__file__))
image_full_path = os.path.join(script_dir, "my_room.jpg")

# Implementation: 
depth_data = generate_virtual_lidar(image_full_path)

import matplotlib.pyplot as plt

def visualize_depth(depth_array):
    plt.figure(figsize=(10, 6))
    plt.imshow(depth_array, cmap='magma') # 'magma' or 'plasma' look like thermal/LiDAR views
    plt.colorbar(label="Relative Distance")
    plt.title("UGV Virtual LiDAR Depth Map")
    plt.show()

# Add this to your main execution
visualize_depth(depth_data)