import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import cv2
import imutils
import numpy as np
from PIL import Image
from pickle import load
from natsort import natsorted
from skimage.metrics import mean_squared_error as mse

#################################################################
# CONTROL VARIABLES
#################################################################
IMAGE_SIZE = int(sys.argv[1])
SOURCE_DIR = sys.argv[2]
START_STOP = sys.argv[3]
SIDE_CONFIGURATION = sys.argv[4]
K_NEIGHBOURS = 2 * int(sys.argv[5])
MODE = sys.argv[6]
IOU_OR_IMAGE_REGISTRATION = sys.argv[7]
USE_SEGMENTATION_DATA = True if(sys.argv[8] == "True") else False
IRIS_COMBINATION = sys.argv[9]

def compute_mse(img_1, img_2): # auxiliary function, computes the MSE distance between images "img_1" and "img_2"

    return(mse(img_1, img_2))

def compute_image_registration_cost(base_image, other_image, max_features = 50000, keep_percentage = 0.9, debug = False): # auxiliary function, computes the image registration cost and returns the transformed image

    # --------------------------------------------------------------------------------------------------
    # apply the image registration algorithm
    # --------------------------------------------------------------------------------------------------
    # convert both the neighbour's image and base image to greyscale
    base_image_grey = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    other_image_grey = cv2.cvtColor(other_image, cv2.COLOR_BGR2GRAY)

    # use ORB to detect keypoints and extract (binary) local invariant features
    orb = cv2.ORB_create(max_features)
    (kpsA, descsA) = orb.detectAndCompute(other_image_grey, None)
    (kpsB, descsB) = orb.detectAndCompute(base_image_grey, None)

    # match the features
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)

    # sort the matches by their distance (the smaller the distance, the "more similar" the features are)
    matches = sorted(matches, key = lambda x : x.distance)

    # keep only the top matches
    keep = int(len(matches) * keep_percentage)
    matches = matches[:keep]

    # check to see if we should visualize the matched keypoints
    if(debug):
        matchedVis = cv2.drawMatches(other_image, kpsA, base_image, kpsB, matches, None)
        matchedVis = imutils.resize(matchedVis, width = 1000)
        cv2.imshow("Matched Keypoints", matchedVis)
        cv2.waitKey(0)

    # allocate memory for the keypoints (x, y)-coordinates from the top matches
    ptsA = np.zeros((len(matches), 2), dtype = "float")
    ptsB = np.zeros((len(matches), 2), dtype = "float")

    # loop over the top matches
    for (i, m) in enumerate(matches):
        # indicate that the two keypoints in the respective images map to each other
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    # compute the homography matrix between the two sets of matched points
    (H, _) = cv2.findHomography(ptsA, ptsB, method = cv2.RANSAC)

    # use the homography matrix to align the images
    (h, w) = base_image.shape[:2]
    aligned = cv2.warpPerspective(other_image, H, (w, h))

    # --------------------------------------------------------------------------
    # compute the image registration cost
    # --------------------------------------------------------------------------
    other_image = other_image.astype("float") / 63.75
    aligned = aligned.astype("float") / 63.75

    score = np.sum((other_image.astype("float") - aligned.astype("float")) ** 2)
    score /= float(other_image.shape[0] * other_image.shape[1])

    return(score, aligned)

def compute_IoU_score(base_image, other_image): # auxiliary function, computes the IoU score

    def rgb2grey(rgb):

        return(0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2])

    base_image_grey = rgb2grey(base_image)
    other_image_grey = rgb2grey(other_image)

    intersection = np.logical_and(base_image_grey, other_image_grey)
    union = np.logical_or(base_image_grey, other_image_grey)
    score = np.sum(intersection) / np.sum(union)

    return(score)

def compute_masks_weight(masks_test, masks_neighbour): # auxiliary function, computes the masks' weight

    if(IOU_OR_IMAGE_REGISTRATION == "IoU"):
        iris_mask_weight = compute_IoU_score(masks_test["iris"], masks_neighbour["iris"])

        if(masks_test["eyebrow"] is None):  masks_weight = 1 - iris_mask_weight
        else: 
            eyebrow_mask_weight = compute_IoU_score(masks_test["eyebrow"], masks_neighbour["eyebrow"])

            masks_weight = 1 - ((iris_mask_weight + eyebrow_mask_weight) / 2)
            
    else:
        try:
            iris_mask_weight, _ = compute_image_registration_cost(masks_test["iris"], masks_neighbour["iris"])
        except: iris_mask_weight = None

        try:
            #eyebrow_mask_weight, _ = compute_image_registration_cost(masks_test["eyebrow"], masks_neighbour["eyebrow"])
            eyebrow_mask_weight = None
        except: eyebrow_mask_weight = None

        if((iris_mask_weight is None) and (eyebrow_mask_weight is None)): masks_weight = 1.0
        else: masks_weight = ((iris_mask_weight if(iris_mask_weight is not None) else 0.0) + (eyebrow_mask_weight if(eyebrow_mask_weight is not None) else 0.0)) / (2 if((iris_mask_weight is not None) and (eyebrow_mask_weight is not None)) else 1)
        
    return(masks_weight)

if(__name__ == "__main__"):

    ######################################################################################################################################################################################################
    # INITIAL SETUP
    ######################################################################################################################################################################################################
    directory = natsorted(list(filter(lambda x : x[0] != ".", os.listdir("../images"))))

    image_A = np.asarray(Image.open("../images/" + directory[0]).resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS))
    image_B = np.asarray(Image.open("../images/" + directory[1]).resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS))

    if(IOU_OR_IMAGE_REGISTRATION == "IoU"):
        masks_test_A = {"iris": (np.asarray(Image.open("test_pair_masks/iris_A.png")) if(os.path.exists("test_pair_masks/iris_A.png")) else None),
                        "eyebrow": (np.asarray(Image.open("test_pair_masks/eyebrow_A.png")) if(os.path.exists("test_pair_masks/eyebrow_A.png")) else None)}

        masks_test_B = {"iris": (np.asarray(Image.open("test_pair_masks/iris_B.png")) if(os.path.exists("test_pair_masks/iris_B.png")) else None),
                        "eyebrow": (np.asarray(Image.open("test_pair_masks/eyebrow_B.png")) if(os.path.exists("test_pair_masks/eyebrow_B.png")) else None)}

    else:
        masks_test_A = {"iris": (cv2.resize(cv2.imread("test_pair_masks/iris_A.png"), (256, 256), interpolation = cv2.INTER_NEAREST) if(os.path.exists("test_pair_masks/iris_A.png")) else None),
                        "eyebrow": (cv2.resize(cv2.imread("test_pair_masks/eyebrow_A.png"), (256, 256), interpolation = cv2.INTER_NEAREST) if(os.path.exists("test_pair_masks/eyebrow_A.png")) else None)}

        masks_test_B = {"iris": (cv2.resize(cv2.imread("test_pair_masks/iris_B.png"), (256, 256), interpolation = cv2.INTER_NEAREST) if(os.path.exists("test_pair_masks/iris_B.png")) else None),
                        "eyebrow": (cv2.resize(cv2.imread("test_pair_masks/eyebrow_B.png"), (256, 256), interpolation = cv2.INTER_NEAREST) if(os.path.exists("test_pair_masks/eyebrow_B.png")) else None)}
    
    # load the synthetic dataset
    with open("stylegan2/synthetic_dataset_G/structured_dataset_" + SOURCE_DIR[-1] + ".pkl", "rb") as file:
        aux = load(file)
        dataset = aux[IRIS_COMBINATION][SIDE_CONFIGURATION]
    
    ################################################################################################################################################################
    # FIND THE "K_NEIGHBOURS" CLOSEST NEIGHBOURS (USING THE CHOSEN DISTANCE METRIC)
    ################################################################################################################################################################
    best_neighbours = []
    neighbours_configurations = {}
    for idx, i in enumerate(dataset):
        
        #print(str(idx + 1) + "/" + str(len(dataset)))

        # get the path to this pair
        path = SOURCE_DIR + "/" + i

        try:
            aux = np.load(path)
        except Exception as e: 
            print(e)
            continue
        
        possible_neighbour_A = aux[:, :, :3]
        possible_neighbour_B = aux[:, :, 3:]
        
        configuration = [possible_neighbour_A, possible_neighbour_B]
        
        # --------------------------------------------------------------------------------------------------
        # if required, check whether the periocular components align properly
        # --------------------------------------------------------------------------------------------------
        if(USE_SEGMENTATION_DATA):

            segmentation_maps_path = path.replace("images_", "segmentation_maps_").replace(".npy", "") + "/"
            segmentation_maps = list(filter(lambda x : x[0] != ".", os.listdir(segmentation_maps_path)))
            
            # load the specific masks
            if(IOU_OR_IMAGE_REGISTRATION == "IoU"):
                masks_neighbour_A = {"iris": np.asarray(Image.open(segmentation_maps_path + "iris_A.png")),
                                "eyebrow": np.asarray(Image.open(segmentation_maps_path + "eyebrow_A.png"))}
            else:
                masks_neighbour_A = {"iris": cv2.imread(segmentation_maps_path + "iris_A_256.png"),
                                "eyebrow": cv2.imread(segmentation_maps_path + "eyebrow_A_256.png")}

            if(IOU_OR_IMAGE_REGISTRATION == "IoU"):
                masks_neighbour_B = {"iris": np.asarray(Image.open(segmentation_maps_path + "iris_B.png")),
                                "eyebrow": np.asarray(Image.open(segmentation_maps_path + "eyebrow_B.png"))}
            else:
                masks_neighbour_B = {"iris": cv2.imread(segmentation_maps_path + "iris_B_256.png"),
                                "eyebrow": cv2.imread(segmentation_maps_path + "eyebrow_B_256.png")}

            # compute the iris difference penalty
            iris_A = image_A.copy()
            iris_A[masks_test_A["iris"][:, :, 0] != 255] = np.asarray([0, 0, 0])

            iris_neighbour = configuration[0].copy()
            iris_neighbour[masks_neighbour_A["iris"][:, :, 0] != 255] = np.asarray([0, 0, 0])

            iris_penalty_A = mse(iris_A, iris_neighbour)

            # compute the iris difference penalty
            iris_B = image_B.copy()
            iris_B[masks_test_B["iris"][:, :, 0] != 255] = np.asarray([0, 0, 0])

            iris_neighbour = configuration[1].copy()
            iris_neighbour[masks_neighbour_B["iris"][:, :, 0] != 255] = np.asarray([0, 0, 0])

            iris_penalty_B = mse(iris_B, iris_neighbour)
            
            # compute the weight that should be attributed to the masks
            masks_weight_A = compute_masks_weight(masks_test_A, masks_neighbour_A)
            masks_weight_B = compute_masks_weight(masks_test_B, masks_neighbour_B)

        else:
            masks_weight_A = 0.0
            masks_weight_B = 1.0
        
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------
        # compare the first image of the test pair with the first image of the neighbour pair and, only after that, compare the second images (outside the for loop)
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------
        if(MODE == "elementwise_comparison"):
            
            weighted_mse_value = (masks_weight_A + masks_weight_B) * compute_mse(image_A, configuration[0])
            #weighted_mse_value = iris_penalty_A + (masks_weight_A + masks_weight_B) * compute_mse(image_A, configuration[0])
            
            # if we have few neighbours, we have to keep this one (for now)
            if(len(best_neighbours) < K_NEIGHBOURS):

                # save the best configuration for this neighbour
                neighbours_configurations[path] = configuration
                
                best_neighbours.append([path, weighted_mse_value, masks_weight_B, iris_penalty_B])
                continue
            
            # store this neighbour, if it is better than what we currently have
            worst_neighbour = max(best_neighbours, key = lambda x : x[1])
            
            if(weighted_mse_value < worst_neighbour[1]):
                
                # save the best configuration for this neighbour
                neighbours_configurations[path] = configuration

                # update the best neighbours list
                best_neighbours.remove(worst_neighbour)
                best_neighbours.append([path, weighted_mse_value, masks_weight_B, iris_penalty_B])

        # -------------------------------------------------------------------------------------------------------------------------
        # compare the pairs as if they were one image
        # -------------------------------------------------------------------------------------------------------------------------
        else:
            config_column_stack = np.column_stack((configuration[0], configuration[1]))
            image_A_image_B_column_stack = np.column_stack((image_A, image_B))

            weighted_mse_value = (masks_weight_A + masks_weight_B) * compute_mse(image_A_image_B_column_stack, config_column_stack)

            # if we have few neighbours, we have to keep this one (for now)
            if(len(best_neighbours) < K_NEIGHBOURS):

                # save the best configuration for this neighbour
                neighbours_configurations[path] = configuration

                best_neighbours.append([path, weighted_mse_value])
                continue

            # store this neighbour, if it is better than what we currently have
            worst_neighbour = max(best_neighbours, key = lambda x : x[1])

            if(weighted_mse_value < worst_neighbour[1]):

                # save the best configuration for this neighbour
                neighbours_configurations[path] = configuration

                # update the best neighbours list
                best_neighbours.remove(worst_neighbour)
                best_neighbours.append([path, weighted_mse_value])

    # -----------------------------------------------------------------------
    # special step in case the mode was set to "elementwise_comparison"
    # -----------------------------------------------------------------------
    if(MODE == "elementwise_comparison"):
        best_neighbours_aux = []

        for i in best_neighbours:

            # load the neighbour's configuration (i.e. AB or BA)
            neighbour_np = neighbours_configurations[i[0]]

            # compute the differences with regards to the second images
            weighted_mse_value = i[2] * compute_mse(image_B, neighbour_np[1])
            best_neighbours_aux.append([i[0], weighted_mse_value])
    
        best_neighbours = best_neighbours_aux

    #################################################################################################
    # SAVE THE NEIGHBOURS AND THEIR DISTANCES
    #################################################################################################
    best_neighbours_np = np.zeros((K_NEIGHBOURS, IMAGE_SIZE, IMAGE_SIZE * 2, 3))
    best_neighbours_distances_np = np.zeros((K_NEIGHBOURS))
    
    with open("neighbour_names.txt", "a") as file:
        for idx, i in enumerate(best_neighbours):
            file.write(i[0] + "\n")
            
            # load the neighbour's configuration (i.e. AB or BA)
            neighbour_np = neighbours_configurations[i[0]]
            
            best_neighbours_np[idx, :, :, :] = np.column_stack((neighbour_np[0], neighbour_np[1]))
            best_neighbours_distances_np[idx] = i[1]
            
    np.save("results/best_neighbours_" + START_STOP + ".npy", best_neighbours_np)
    np.save("results/best_neighbours_distances_" + START_STOP + ".npy", best_neighbours_distances_np)