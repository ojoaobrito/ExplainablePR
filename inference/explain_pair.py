import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import time
import numpy as np
from PIL import Image
from shutil import copyfile, rmtree
from natsort import natsorted

from code.utils import find_centre, find_closest_centroid

##############################################################################################################################
# CONTROL VARIABLES
##############################################################################################################################
# ------------------------------
# common variables
# ------------------------------
IMAGE_SIZE = 128
REMOVE_TRASH = True
EXPLAIN_A_AND_B = True
ADD_INTERPRETABLE_COLOUR = True

# -------------------------------------------------------------------
# variables targeting "classify_pair.py"
# -------------------------------------------------------------------
CNN_TYPE = "densenet161"
CNN_PATH = "../../trained_models/densenet_161/models/densenet_161.pt"

# -------------------------------------------------------------------------------------------------------------------
# variables targeting "segment_periocular_components.py"
# -------------------------------------------------------------------------------------------------------------------
MRCNN_IRIS_WEIGHTS_PATH = "../../trained_models/mask_rcnn/periocular_iris/mask_rcnn_periocular_0030_iris.h5"
MRCNN_EYEBROW_WEIGHTS_PATH = "../../trained_models/mask_rcnn/periocular_eyebrow/mask_rcnn_periocular_0030_eyebrow.h5"
MRCNN_SCLERA_WEIGHTS_PATH = "../../trained_models/mask_rcnn/periocular_sclera/mask_rcnn_periocular_0030_sclera.h5"

# ----------------------------------------------------------------------------------------------------------------------------
# variables targeting "find_neighbours_master.py"
# ----------------------------------------------------------------------------------------------------------------------------
SAVE_NEIGHBOURS = True
K_NEIGHBOURS = 200 if(len(sys.argv) != 6) else int(sys.argv[1])
MODE = "elementwise_comparison" if(len(sys.argv) != 6) else sys.argv[2] # either "elementwise_comparison" or "full_comparison"
CNN_SIDE_TYPE = "resnet18"
CNN_SIDE_PATH = "../../trained_models/resnet_18/models/resnet_18.pt"
IOU_OR_IMAGE_REGISTRATION = "IoU" if(len(sys.argv) != 6) else sys.argv[3] # either "IoU" or "image_registration"
USE_SEGMENTATION_DATA = True if(len(sys.argv) != 6) else (True if(sys.argv[4] == "True") else False)

# -------------------------------------------------------------------------------------------------
# variables targeting "compute_difference_mask.py"
# -------------------------------------------------------------------------------------------------
MAP_RANGE = (-1, 0)
FIRST_OR_SECOND_WAY = "first" if(len(sys.argv) != 6) else sys.argv[5] # either "first" or "second"

# ---------------------------------------------
# variables targeting "assemble_explanation.py"
# ---------------------------------------------
TRANSPARENT_BACKGROUND = True

def check_for_errors(r_value): # auxiliary function, checks for errors when running the main scripts

    if(r_value != 0):
        print("ERROR FOUND, STOPPING NOW...")
        sys.exit()

if(__name__ == "__main__"):

    print("\nEXPLAINING THE GIVEN PAIR...")
    print("[INFO] " + ("USING SEGMENTATION DATA!" if(USE_SEGMENTATION_DATA) else "NOT USING SEGMENTATION DATA!"))
    t0 = time.time()

    ###############################################################################################
    # PREPARE THE PAIR
    ###############################################################################################
    directory = natsorted(list(filter(lambda x : x[0] != ".", os.listdir("images"))))
    
    if(not ("a_" in directory[0])): os.rename("images/" + directory[0], "images/a_" + directory[0])
    if(not ("b_" in directory[1])): os.rename("images/" + directory[1], "images/b_" + directory[1])

    #######################################################################################################################################################################
    # EXPLAIN THE PAIR
    #######################################################################################################################################################################
    os.chdir("./code")
    
    # ---------------------------------------------------------------------------------------------------------------------------
    # use the trained CNN to determine if the pair we want to explain is genuine or impostor (the first part of the final answer)
    # ---------------------------------------------------------------------------------------------------------------------------
    if(not ("G" in directory[0] or "I" in directory[0])):
        print("\nCLASSIFYING THE GIVEN PAIR...")
        args = [str(IMAGE_SIZE), CNN_TYPE, CNN_PATH]
        r_value = os.system("python3 classify_pair.py " + " ".join(args))
        check_for_errors(r_value)
        print("DONE CLASSIFYING THE GIVEN PAIR!")

    directory = natsorted(list(filter(lambda x : x[0] != ".", os.listdir("../images"))))
    genuine_or_impostor = "G" if("G" in directory[0]) else "I"

    # ---------------------------------------------------------------------------------------------------------------------------------------------------------
    # if required, use the trained Mask-RCNN model to segment the iris, eyebrow, sclera and skin from images A and B
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------
    if(genuine_or_impostor == "I" and USE_SEGMENTATION_DATA):

        # we have to get masks for the iris and eyebrow
        if(not os.path.exists("pair_name.txt")):
            print("\nSEGMENTING THE GIVEN PAIR...")
            args = [str(IMAGE_SIZE), MRCNN_IRIS_WEIGHTS_PATH, MRCNN_EYEBROW_WEIGHTS_PATH, MRCNN_SCLERA_WEIGHTS_PATH]
            r_value = os.system("python3 segment_periocular_components.py " + " ".join(args))
            check_for_errors(r_value)
            print("DONE SEGMENTING THE GIVEN PAIR!")

        # we can use the previously computed masks
        else:
            print("\nSEGMENTING THE GIVEN PAIR...")
            if(os.path.exists("test_pair_masks")): rmtree("test_pair_masks")
            os.makedirs("test_pair_masks")

            with open("pair_name.txt", "r") as file:
                pair_name = file.read().splitlines()[0]
            
            masks = list(filter(lambda x : (x[0] != ".") and (x != "1.jpg") and (x != "2.jpg"), os.listdir("performance_evaluation_pairs_paper/" + pair_name)))

            for i in masks:
                copyfile("performance_evaluation_pairs_paper/" + pair_name + "/" + i, "test_pair_masks/" + i)
                Image.open("test_pair_masks/" + i).resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS).save("test_pair_masks/" + i)

            # -------------------------------------------------------------------------------------------------------------------------------------------------------------
            # check if any of the masks is completely black (it can happen, especially with the eyebrow)
            # -------------------------------------------------------------------------------------------------------------------------------------------------------------
            masks = list(filter(lambda x : x[0] != ".", os.listdir("test_pair_masks")))

            for i in masks:
                if(np.array_equal(np.asarray(Image.open("test_pair_masks/" + i).convert("RGB")), np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3)))): os.remove("test_pair_masks/" + i)

            print("DONE SEGMENTING THE GIVEN PAIR!")

    # make sure the images have a proper size
    Image.open("../images/" + directory[0]).resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS).save("../images/" + directory[0])
    Image.open("../images/" + directory[1]).resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS).save("../images/" + directory[1])

    # --------------------------------------------------------------------------------------------------------------------------------------------------
    # run the main pipeline
    # --------------------------------------------------------------------------------------------------------------------------------------------------
    if(genuine_or_impostor == "I"):
        for i in range(2):
            
            directory = natsorted(list(filter(lambda x : x[0] != ".", os.listdir("../images"))))

            # load the test images
            if(EXPLAIN_A_AND_B and i == 1):
                
                # change the order of the images
                os.rename("../images/" + directory[0], "../images/" + directory[0].replace("a", "c"))
                os.rename("../images/" + directory[1], "../images/" + directory[1].replace("b", "a"))
                os.rename("../images/" + directory[0].replace("a", "c"), "../images/" + directory[0].replace("a", "c").replace("c", "b"))

                # rename the segmentation masks
                if(USE_SEGMENTATION_DATA):
                    for j in list(filter(lambda x : x[0] != ".", os.listdir("test_pair_masks"))): 
                        os.rename("test_pair_masks/" + j, "test_pair_masks/" + j.replace("A", "C"))

                    for j in list(filter(lambda x : x[0] != ".", os.listdir("test_pair_masks"))): 
                        os.rename("test_pair_masks/" + j, "test_pair_masks/" + j.replace("B", "A"))

                    for j in list(filter(lambda x : x[0] != ".", os.listdir("test_pair_masks"))): 
                        os.rename("test_pair_masks/" + j, "test_pair_masks/" + j.replace("C", "B"))
                
                directory = natsorted(list(filter(lambda x : x[0] != ".", os.listdir("../images"))))
                image_A = np.asarray(Image.open("../images/" + directory[0]))
                image_B = np.asarray(Image.open("../images/" + directory[1]))

            else:
                image_A = np.asarray(Image.open("../images/" + directory[0]))
                image_B = np.asarray(Image.open("../images/" + directory[1]))

            if(genuine_or_impostor == "I"):

                # ---------------------------------------------------------------------------------------------------
                # determine the iris position combination of the test pair
                # ---------------------------------------------------------------------------------------------------
                centroids = np.load("stylegan2/synthetic_dataset_G/centroids.npy")

                iris_combination = find_closest_centroid(find_centre("test_pair_masks/iris_A.png"), centroids)
                iris_combination += "_" + find_closest_centroid(find_centre("test_pair_masks/iris_B.png"), centroids)
                print(iris_combination)
                # -----------------------------------------------------------------------------------------
                # find the closest neighbours to the pair we want to explain
                # -----------------------------------------------------------------------------------------
                print("\nFINDING NEIGHBOURS...")
                args = [str(IMAGE_SIZE), str(SAVE_NEIGHBOURS), str(K_NEIGHBOURS), MODE, CNN_SIDE_TYPE, 
                    CNN_SIDE_PATH, IOU_OR_IMAGE_REGISTRATION, str(USE_SEGMENTATION_DATA), iris_combination]
                r_value = os.system("python3 find_neighbours_master.py " + " ".join(args))
                check_for_errors(r_value)
                print("DONE FINDING NEIGHBOURS!")
                
                # ---------------------------------------------------------------------------------------------------------------------
                # compute a difference mask between image B and its closest neighbours (the second part of the final answer)
                # ---------------------------------------------------------------------------------------------------------------------
                print("\nCOMPUTING A DIFFERENCE MASK...")
                args = [str(IMAGE_SIZE), str(MAP_RANGE[0]).replace(".", ","), str(MAP_RANGE[1]).replace(".", ","), FIRST_OR_SECOND_WAY]
                r_value = os.system("python3 compute_difference_mask.py " + " ".join(args))

                print("DONE COMPUTING A DIFFERENCE MASK!")

            if(i == 0): os.rename("difference_mask.npy", "difference_mask_1.npy")
            else: os.rename("difference_mask.npy", "difference_mask_2.npy")
            
            if(not EXPLAIN_A_AND_B): break

    # if they were inverted, put the images in the original order
    if(genuine_or_impostor == "I" and EXPLAIN_A_AND_B):
        directory = natsorted(list(filter(lambda x : x[0] != ".", os.listdir("../images"))))
        
        os.rename("../images/" + directory[0], "../images/" + directory[0].replace("a", "c"))
        os.rename("../images/" + directory[1], "../images/" + directory[1].replace("b", "a"))
        os.rename("../images/" + directory[0].replace("a", "c"), "../images/" + directory[0].replace("a", "c").replace("c", "b"))

    # ----------------------------------------------------------------------
    # assemble the final explanation
    # ----------------------------------------------------------------------
    print("\nASSEMBLING THE FINAL EXPLANATION...")
    args = [str(IMAGE_SIZE), str(TRANSPARENT_BACKGROUND)]
    r_value = os.system("python3 assemble_explanation.py " + " ".join(args))
    check_for_errors(r_value)
    print("DONE ASSEMBLING THE FINAL EXPLANATION!")

    # ------------------------------------------------------------------------------
    # if required, recolour the explanation to make it more interpretable
    # ------------------------------------------------------------------------------
    if(ADD_INTERPRETABLE_COLOUR):
        print("\nRECOLOURING THE FINAL EXPLANATION...")
        args = [str(IMAGE_SIZE), "../explanation.png"]
        r_value = os.system("python3 add_interpretable_colour.py " + " ".join(args))
        check_for_errors(r_value)
        print("DONE RECOLOURING THE FINAL EXPLANATION!")

    os.chdir("..")

    #################################################################################################################
    # REMOVE SOME TRASH
    #################################################################################################################
    if(REMOVE_TRASH):
        if(genuine_or_impostor == "I"): 
            if(os.path.exists("code/name.txt")): os.remove("code/name.txt")
            if(os.path.exists("code/neighbour_names.txt")): os.remove("code/neighbour_names.txt")
            if(os.path.exists("code/best_neighbours_distances.npy")): os.remove("code/best_neighbours_distances.npy")
            if(os.path.exists("code/difference_mask_1.npy")): os.remove("code/difference_mask_1.npy")
            if(os.path.exists("code/difference_mask_2.npy")): os.remove("code/difference_mask_2.npy")
            if(os.path.exists("code/best_neighbours.npy")): os.remove("code/best_neighbours.npy")
            if(os.path.exists("code/test_pair.npy")): os.remove("code/test_pair.npy")
            if(os.path.exists("code/test_pair_latent_code.npy")): os.remove("code/test_pair_latent_code.npy")
            if(os.path.exists("code/results")): rmtree("code/results")
            if(os.path.exists("code/test_pair_masks")): rmtree("code/test_pair_masks")
            if(SAVE_NEIGHBOURS):
                if(os.path.exists("code/neighbours")): rmtree("code/neighbours")

    print("\nDONE EXPLAINING THE GIVEN PAIR!\n")
    print("[INFO] ELAPSED TIME: %.2fs\n" % (time.time() - t0))