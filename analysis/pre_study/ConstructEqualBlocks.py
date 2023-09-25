#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
from src.preprocess import *
from src.neural_network import *
import glob

datapath = '/mnt/cs/projects/HEFEFTMS/FEFTMS/data'
Pdirs    = glob.glob(os.path.join(datapath, "P[0-9][0-9][0-9]"))
Pfiles   = [file for Pdir in Pdirs for file in glob.glob(os.path.join(Pdir, "*.pkl"))]


#%%
result_dict = {}
for path in Pfiles:
    # load data from path
    df = pd.read_pickle(path)


    def extract_image_name(path):
        return path.split('/')[-1].split('.')[0]


    # Add a new column for image name
    df['image_name'] = df['image'].apply(extract_image_name)
    
    # Group by image name and collect RTs
    grouped = df.groupby('image_name')['RT'].apply(list).to_dict()
    
    # Merge with the main result dictionary
    for key, value in grouped.items():
        if key in result_dict:
            result_dict[key].extend(value)
        else:
            result_dict[key] = value

print(result_dict)
# %%
def compute_avg_rt(data):
    """
    Compute the average reaction time for each image.
    """
    avg_rts = {}
    for img, rts in data.items():
        avg_rts[img] = sum(rts) / len(rts)
    return avg_rts

def compute_variance(numbers):
    """
    Compute the variance for a list of numbers.
    """
    mean = sum(numbers) / len(numbers)
    variance = sum((n - mean) ** 2 for n in numbers) / len(numbers)
    return variance

def partition_images(data, num_blocks=8, block_size=4):
    """
    Partition the images into blocks based on the average reaction times and also 
    provide description for each block.
    """
    avg_rts = compute_avg_rt(data)
    # Sort images by their average reaction time
    sorted_imgs = sorted(avg_rts.keys(), key=lambda x: avg_rts[x])
    
    blocks = []
    descriptions = []
    for i in range(num_blocks):
        block = []
        block.append(sorted_imgs.pop())  # Add a difficult image
        block.append(sorted_imgs.pop(0))  # Add an easy image
        block.extend(sorted_imgs.pop(len(sorted_imgs)//2 - 1) for _ in range(block_size - 2))  # Add medium images
        
        block_rts = [avg_rts[img] for img in block]
        block_mean_rt = sum(block_rts) / len(block_rts)
        block_variance = compute_variance(block_rts)
        descriptions.append({
            'block_mean_rt': block_mean_rt,
            'block_variance': block_variance
        })
        
        blocks.append(block)
    
    return blocks, descriptions


blocks,descriptions = partition_images(result_dict, num_blocks=8, block_size=4)
print(blocks)

# %%
import random
import math

def compute_variance(numbers):
    """
    Compute the variance for a list of numbers.
    """
    mean = sum(numbers) / len(numbers)
    variance = sum((n - mean) ** 2 for n in numbers) / len(numbers)
    return variance

def compute_avg_rt(data):
    """
    Compute the average reaction time for each image.
    """
    avg_rts = {}
    for img, rts in data.items():
        avg_rts[img] = sum(rts) / len(rts)
    return avg_rts

def block_score(block, avg_rts):
    """
    Compute the score for a block which is the sum of the difference between 
    its mean reaction time and the global mean reaction time, and the difference 
    between its variance and the global variance.
    """
    block_rts = [avg_rts[img] for img in block]
    block_mean_rt = sum(block_rts) / len(block_rts)
    block_variance = compute_variance(block_rts)
    
    global_mean = sum(avg_rts.values()) / len(avg_rts)
    global_variance = compute_variance(list(avg_rts.values()))
    
    return abs(block_mean_rt - global_mean) + abs(block_variance - global_variance)

def annealing_optimization(data, blocks, num_iterations=10000, initial_temperature=1000, cooling_rate=0.995):
    """
    Use simulated annealing to optimize the blocks to equalize mean and variance across blocks.
    """
    avg_rts = compute_avg_rt(data)
    current_score = sum(block_score(block, avg_rts) for block in blocks)
    temperature = initial_temperature
    
    for iteration in range(num_iterations):
        # Randomly choose two blocks and two images within those blocks
        block1, block2 = random.sample(blocks, 2)
        img1, img2 = random.choice(block1), random.choice(block2)
        
        # Swap the images and compute the new score
        block1[block1.index(img1)], block2[block2.index(img2)] = img2, img1
        new_score = sum(block_score(block, avg_rts) for block in blocks)
        
        # If the new configuration is better or the temperature is high enough to accept a worse solution
        if new_score < current_score or random.random() < math.exp((current_score - new_score) / temperature):
            current_score = new_score
        else:
            # Revert the swap
            block1[block1.index(img2)], block2[block2.index(img1)] = img1, img2
        
        temperature *= cooling_rate
    
    return blocks

# Initialize the blocks using previous partition_images function
initial_blocks, _ = partition_images(result_dict)

# Optimize the blocks using simulated annealing
optimized_blocks = annealing_optimization(result_dict, initial_blocks)

print(optimized_blocks)

# %%
