import os, sys, inspect

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import numpy as np
from PIL import Image, ImageOps
from pickle import dump
from natsort import natsorted
from random import randint

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils import find_centre, find_closest_centroid

####################################################
# CONTROL VARIABLES
####################################################
# the values are the default ones
if(len(sys.argv) == 1):
    PROCESS_NUMBER = "1"
    DATASET_LENGTH_PROPORTION = 1
    MODE = "regular" # either "regular" or "reduced"

# the values are coming from somewhere else
else:
    PROCESS_NUMBER = sys.argv[1]
    DATASET_LENGTH_PROPORTION = float(sys.argv[2])
    MODE = sys.argv[3]

def create_256_version(): # auxiliary function, creates an upscaled version of every mask (useful for the image registration algorithm)

    masks_dirs = natsorted(list(filter(lambda x : x[0] != ".", os.listdir("synthetic_dataset_G/segmentation_maps_" + PROCESS_NUMBER))))

    for idx, i in enumerate(masks_dirs):

        print(str(idx + 1) + "/" + str(len(masks_dirs)))

        masks = list(filter(lambda x : x[0] != ".", os.listdir("synthetic_dataset_G/segmentation_maps_" + PROCESS_NUMBER + "/" + i)))

        for j in masks:
            img = Image.open("synthetic_dataset_G/segmentation_maps_" + PROCESS_NUMBER + "/" + i + "/" + j).resize((256, 256), Image.NEAREST)
            img.save("synthetic_dataset_G/segmentation_maps_" + PROCESS_NUMBER + "/" + i + "/" + j.replace(".png", "_256.png"))

def create_mirrored_copies(): # auxiliary function, creates mirrored copies of each synthetic pair

    masks_dirs = natsorted(list(filter(lambda x : x[0] != ".", os.listdir("synthetic_dataset_G/segmentation_maps_" + PROCESS_NUMBER))))
        
    for idx, i in enumerate(masks_dirs):

        print(str(idx + 1) + "/" + str(len(masks_dirs)))
        
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # take care of the segmentation maps
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        new_dir_name = "_".join(i.split("_")[:2]) + "_" + ("L_L" if("R_R" in i) else "R_R")
        if(os.path.exists("synthetic_dataset_G/segmentation_maps_" + PROCESS_NUMBER + "/" + new_dir_name)): continue
        os.makedirs("synthetic_dataset_G/segmentation_maps_" + PROCESS_NUMBER + "/" + new_dir_name)

        masks = list(filter(lambda x : x[0] != ".", os.listdir("synthetic_dataset_G/segmentation_maps_" + PROCESS_NUMBER + "/" + i)))

        for j in masks: 
            ImageOps.mirror(Image.open("synthetic_dataset_G/segmentation_maps_" + PROCESS_NUMBER + "/" + i + "/" + j)).save("synthetic_dataset_G/segmentation_maps_" + PROCESS_NUMBER + "/" + new_dir_name + "/" + j)

        # ----------------------------------------------------------------------------------------------------------
        # take care of the actual synthetic pair
        # ----------------------------------------------------------------------------------------------------------
        synthetic_pair = np.load("synthetic_dataset_G/images_" + PROCESS_NUMBER + "/" + i + ".npy").copy()
        new_synthetic_pair = np.dstack((np.flip(synthetic_pair[:, :, :3], 1), np.flip(synthetic_pair[:, :, 3:], 1)))
        np.save("synthetic_dataset_G/images_" + PROCESS_NUMBER + "/" + new_dir_name + ".npy", new_synthetic_pair)

def create_dictionaries(): # auxiliary function, creates a structured dictionary to compile the dataset in a smarter way (thus speeding up future searches)

    centroids = np.load("synthetic_dataset_G/centroids.npy")

    # ------------------------------------------------------------------------------------------------------------------------------
    # create a structured dictionary
    # ------------------------------------------------------------------------------------------------------------------------------
    iris_position_combinations = {"L_C": {}, "R_C": {}, "C_L": {}, "C_R": {}, "L_R": {}, "R_L": {}, "L_L": {}, "R_R": {}, "C_C": {}}

    for i in iris_position_combinations.keys():
        iris_position_combinations[i] = {"L_L": [], "R_R": []}

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # determine the iris position, based on previously computed centroids
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    masks_dirs = natsorted(list(filter(lambda x : x[0] != ".", os.listdir("synthetic_dataset_G/segmentation_maps_" + PROCESS_NUMBER))))
    
    for i in range(0, len(masks_dirs), 2):

        if(MODE == "regular"): print(str(i + 1) + "/" + str(len(masks_dirs)))

        # check if we should use the entire dataset or a reduced version
        random_int = randint(1, 100)
        
        if(random_int > int(DATASET_LENGTH_PROPORTION * 100)): continue

        try:
            # determine the iris position combination of the original masks
            combination = find_closest_centroid(find_centre("synthetic_dataset_G/segmentation_maps_" + PROCESS_NUMBER + "/" + masks_dirs[i] + "/iris_A.png"), centroids)
            combination += "_" + find_closest_centroid(find_centre("synthetic_dataset_G/segmentation_maps_" + PROCESS_NUMBER + "/" + masks_dirs[i] + "/iris_B.png"), centroids)
            
            # determine the iris position combination of the mirrored masks
            combination_mirrored = find_closest_centroid(find_centre("synthetic_dataset_G/segmentation_maps_" + PROCESS_NUMBER + "/" + masks_dirs[i + 1] + "/iris_A.png"), centroids)
            combination_mirrored += "_" + find_closest_centroid(find_centre("synthetic_dataset_G/segmentation_maps_" + PROCESS_NUMBER + "/" + masks_dirs[i + 1] + "/iris_B.png"), centroids)

            # add everything to the dictionary
            iris_position_combinations[combination]["_".join(masks_dirs[i].split("_")[2:])].append(masks_dirs[i] + ".npy")
            iris_position_combinations[combination_mirrored]["_".join(masks_dirs[i + 1].split("_")[2:])].append(masks_dirs[i + 1] + ".npy")
        
        except Exception as e: 
            if(MODE == "regular"): print(e)
            continue

    # dump the dictionary to a file
    with open("synthetic_dataset_G/structured_dataset_" + PROCESS_NUMBER + ".pkl", "wb") as file:
        dump(iris_position_combinations, file)

if(__name__ == "__main__"):

    create_256_version()
    create_mirrored_copies()
    create_dictionaries()