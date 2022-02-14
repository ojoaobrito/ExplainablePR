import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import numpy as np
from natsort import natsorted
from PIL import Image, ImageEnhance

np.seterr("raise")

####################################################################################################
# CONTROL VARIABLES
####################################################################################################
# the values are the default ones (NOTE: if the values are to be changed, do so in this "if" branch)
if(len(sys.argv) == 1):
    IMAGE_SIZE = 128
    MAP_RANGE = (-1, 0)
    FIRST_OR_SECOND_WAY = "first" # either "first" or "second"

# the values are coming from the "explain_pair.py" master script
else:
    IMAGE_SIZE = int(sys.argv[1])
    MAP_RANGE = (float(sys.argv[2].replace(",", ".")), float(sys.argv[3].replace(",", ".")))
    FIRST_OR_SECOND_WAY = sys.argv[4]

def compute_weighted_average(image_B, neighbours_np, neighbours_distances_np): # auxiliary function, computes a difference mask that is weighted by the distance of each neighbour to image B
    
    if(sum(neighbours_distances_np) == 0.0): sys.exit(10)

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # compute the neighbours' weights based on their distances to the test pair
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if(len(neighbours_distances_np) == 1): weights = [1.0]
    else:
        inverted_distances = [((sum(neighbours_distances_np) - i) / i) for i in neighbours_distances_np]
        
        if(FIRST_OR_SECOND_WAY == "first"):
            weights = [(i / sum(inverted_distances)) for i in inverted_distances]

        else:
            # map the values in the range [a1, a2] to the range [MAP_RANGE[0], MAP_RANGE[1]]
            map_function = np.vectorize(lambda x : MAP_RANGE[0] + ((((x - min(inverted_distances)) * (MAP_RANGE[1] - MAP_RANGE[0]))) / (max(inverted_distances) - min(inverted_distances))))
            rescaled_inverted_distances = map_function(inverted_distances)

            # actually compute the weights
            exponential_growth_function = np.vectorize(lambda x : np.math.exp(1 * x))
            weights = exponential_growth_function(rescaled_inverted_distances)
            weights = [i / (sum(weights)) for i in weights]
            
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # get the difference masks for every neighbour
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    difference_masks = np.zeros((neighbours_np.shape[0], IMAGE_SIZE, IMAGE_SIZE))
    for idx, i in enumerate(neighbours_np):
        
        # compute the pixel difference between the test pair's image B and this neighbour's image B
        difference = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
        for j in range(IMAGE_SIZE):
            for k in range(IMAGE_SIZE):
                difference[j][k] = (np.sqrt((image_B[j][k][0] - i[j][k][0]) ** 2 + (image_B[j][k][1] - i[j][k][1]) ** 2 + (image_B[j][k][2] - i[j][k][2]) ** 2) / np.sqrt(195075)) * 255

        difference_masks[idx, :, :] = difference.copy()
    
    # -----------------------------------------------------------------------------------------------
    # compute the average difference mask (weighted by how close the neighbours are to the test pair)
    # -----------------------------------------------------------------------------------------------
    final_difference_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    for i in range(IMAGE_SIZE):
        for j in range(IMAGE_SIZE):
            weighted_average_aux = 0.0
            for kdx, k in enumerate(difference_masks):
                weighted_average_aux += (k[i][j] * weights[kdx])

            final_difference_mask[i][j] = int(weighted_average_aux)

    # ------------------------------------------------------------------------------------
    # increase the brightness of the difference mask whithin the iris just a little bit
    # ------------------------------------------------------------------------------------
    if(os.path.exists("test_pair_masks/iris_B.png")):

        iris_mask = np.asarray(Image.open("test_pair_masks/iris_B.png"))
        iris = final_difference_mask.copy()

        iris[iris_mask[:, :, 0] != 255] = 0

        img = Image.fromarray(iris.astype(np.uint8))
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.5)
        iris = np.asarray(img.convert("L"))
        
        final_difference_mask[iris_mask[:, :, 0] == 255] = iris[iris_mask[:, :, 0] == 255]
    
    final_difference_mask = Image.fromarray(final_difference_mask.astype(np.uint8))
    enhancer = ImageEnhance.Brightness(final_difference_mask)
    final_difference_mask = enhancer.enhance(1.3)

    final_difference_mask = np.asarray(final_difference_mask)

    return(final_difference_mask)

if(__name__ == "__main__"):

    directory = natsorted(list(filter(lambda x : x[0] != ".", os.listdir("../images"))))

    # load everything that is necessary
    image_B = np.asarray(Image.open("../images/" + directory[1]).resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)).copy()
    neighbours_np = np.load("best_neighbours.npy")
    neighbours_distances_np = np.load("best_neighbours_distances.npy")

    # compute the difference mask
    final_difference_mask = compute_weighted_average(image_B, neighbours_np, neighbours_distances_np)

    # save the difference mask
    np.save("difference_mask.npy", final_difference_mask)