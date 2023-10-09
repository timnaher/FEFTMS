#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
from src.preprocess import *
from src.neural_network import *
import glob
import csv
import random
import os
import random
import math

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
    provide description for each block. Return any unused images as a list.
    """
    avg_rts = compute_avg_rt(data)
    # Sort images by their average reaction time
    sorted_imgs = sorted(avg_rts.keys(), key=lambda x: avg_rts[x])
    
    blocks = []
    descriptions = []
    for i in range(num_blocks):
        block = []
        if sorted_imgs:  # Check if there are images left
            block.append(sorted_imgs.pop())  # Add a difficult image
        if sorted_imgs:  # Check if there are images left
            block.append(sorted_imgs.pop(0))  # Add an easy image
        for _ in range(block_size - 2):
            if sorted_imgs:  # Check if there are images left
                block.append(sorted_imgs.pop(len(sorted_imgs)//2 - 1))  # Add a medium image
        
        block_rts = [avg_rts[img] for img in block]
        block_mean_rt = sum(block_rts) / len(block_rts) if block_rts else 0
        block_variance = compute_variance(block_rts) if block_rts else 0
        descriptions.append({
            'block_mean_rt': block_mean_rt,
            'block_variance': block_variance
        })
        
        blocks.append(block)
    
    # At this point, any images that were not used will still be in the sorted_imgs list
    unused_images = sorted_imgs
    
    return blocks, descriptions, unused_images

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

datapath = '/mnt/cs/projects/HEFEFTMS/FEFTMS/data'
Pdirs    = glob.glob(os.path.join(datapath, "P[0-9][0-9][0-9]"))
Pfiles   = [file for Pdir in Pdirs for file in glob.glob(os.path.join(Pdir, "*.pkl"))]

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

blocks,descriptions,unused_targets = partition_images(result_dict, num_blocks=8, block_size=4)
print(blocks)


# Initialize the blocks using previous partition_images function
#initial_blocks, _ = partition_images(result_dict)

# Optimize the blocks using simulated annealing
optimized_blocks = annealing_optimization(result_dict, initial_blocks)
# add a .jpg to each image name
optimized_blocks = [[img + '.jpg' for img in block] for block in optimized_blocks]



# Define the path to images
path2images = '/cs/projects/HEFEFTMS/data/images/Waldo'

# Get a list of all filenames in the directory
non_target_images = os.listdir(path2images)

# Filter out all filenames that end with 't.jpg'
non_target_images = [filename for filename in non_target_images if not filename.endswith('t.jpg')]

# Define the custom path
custom_path = "C:/Users/FEFuser/Documents/FEFTMS/images/Waldo/"

# Create 8 CSV files

for i in range(8):
    session = 'A' if i < 4 else 'B'
    filename = f"{session}{i % 4 + 1}.csv"
    
    # Extract target images for the current CSV file
    target_images = optimized_blocks[i]
    
    # Extract image numbers from target images
    target_image_numbers = {img.split('_')[0] for img in target_images}
    
    # Filter non-target images ensuring they are not the same as the target images
    available_non_targets = [img for img in non_target_images if img.split('_')[0] not in target_image_numbers]
    
    # Randomly select 6 different non-target images
    selected_non_targets = random.sample(available_non_targets,4)
    
    # Combine target and non-target images
    blockImages = target_images + selected_non_targets

    # remove selected non-target images from the list of all non-target images
    non_target_images = [img for img in non_target_images if img not in selected_non_targets]
    
    # Shuffle the rows to randomize the order of target and non-target images
    random.shuffle(blockImages)

    
    # Repeat the non-target images 3 times and append to all_images
    #all_images += selected_non_targets * 3
    
    # Add the custom path to each image name
    blockImages = [os.path.join(custom_path, img) for img in blockImages]
    #all_images = [img for img in all_images if img not in used_images]

    # keep track of all images that have been used
    

    
    # Write to CSV file
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['IMAGE'])
        for img in blockImages:
            writer.writerow([img])
    
    print(f"{filename} has been created.")


#%% SANITY CHECK
import csv
import os
from collections import defaultdict

# specify the directory where the CSV files are located
csv_directory = '/cs/projects/HEFEFTMS/FEFTMS/analysis/pre_study'

def check_duplicate_paths(csv_directory):
    # dictionary to store the count of occurrences for each image path
    path_counts = defaultdict(int)

    # iterate through all files in the specified directory
    for filename in os.listdir(csv_directory):
        # check if the file is a CSV file
        if filename.endswith('.csv'):
            # construct the full path to the CSV file
            csv_path = os.path.join(csv_directory, filename)
            
            # read the CSV file
            with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                next(reader, None)  # skip the header
                for row in reader:
                    # increment the count for this image path
                    path_counts[row[0]] += 1

    # check for duplicate paths
    duplicates = {path: count for path, count in path_counts.items() if count > 1}
    
    # print the duplicate paths and their counts
    if duplicates:
        print(f'Duplicate paths found:')
        for path, count in duplicates.items():
            print(f'{path}: {count} occurrences')
    else:
        print('No duplicate paths found.')

# Call the function to check for duplicate paths
check_duplicate_paths(csv_directory)

#%%

# select randomy 2 non target images from non_target_images
# add a .jpg
practice = [random.sample(non_target_images,2),unused_targets]
items    = [item for sublist in practice for item in sublist]
items    = [os.path.join(custom_path, img) for img in items]

random.shuffle(items)

with open('PRACTICE.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['IMAGE'])
    for img in items:
        writer.writerow([img])
        
    print("PRACTICE.csv has been created.")


#%%


        

# %%
