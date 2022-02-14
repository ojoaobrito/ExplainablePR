import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import numpy as np
from PIL import Image
from random import sample
from shutil import copyfile, rmtree
from itertools import combinations, permutations

######################
# CONTROL VARIABLES
######################
AMOUNT_OF_IMAGES = 130

def make_combinations(number_of_IDs, number_of_sets): # auxiliary function, determines how the pair combinations should be made

    list_1 = [k + 1 for k in range(number_of_IDs)]
    list_2 = [k + 1 for k in range(number_of_IDs)]

    all_combinations = []
    for each_permutation in permutations(list_1, len(list_2)): # iterate over all possible permutations
        permutation_aux = list(zip(each_permutation, list_2))

        if(permutation_aux[0] != permutation_aux[1]): # if the numbers are different, we can proceed

            if(len(permutation_aux) == number_of_IDs): # if the length is correct, we can proceed

                permutation_aux = sample(permutation_aux, len(permutation_aux))
                
                if([(i[1], i[0]) not in permutation_aux for i in permutation_aux] == [True for _ in range(len(permutation_aux))]): # if there are no repetitions, we can proceed

                    # after filtering it, the permutation is considered valid
                    all_combinations.append(permutation_aux)

        if(len(all_combinations) == number_of_sets): break

    # save the permutations in a usable format
    final_combinations = []
    for i in all_combinations:
        final_combinations_aux = []
        for j in i:
            final_combinations_aux.append(j[0])
            final_combinations_aux.append(j[1])
        final_combinations.append(final_combinations_aux)

    return(final_combinations)

def create_genuine_pairs(images_per_ID): # auxiliary function, creates genuine pairs

    global dir_count

    ##################################################################################################################
    # CREATE GENUINE PAIRS
    ##################################################################################################################
    for k, v in images_per_ID.items():
        
        # ------------------------------------------------------------------------------------------------------------
        # create a genuine pair with left side images from this ID
        # ------------------------------------------------------------------------------------------------------------
        # choose a combination of images from the left side
        possible_combinations_L = list(filter(lambda x : ("_L_" in x[0] and "_L_" in x[1]), list(combinations(v, 2))))
        chosen_combination_L = sample(possible_combinations_L, ((AMOUNT_OF_IMAGES // 2) // NUM_IDS) // 2)[0]

        # put the chosen images in an appropriate folder
        dir_name = "performance_evaluation_pairs/" + str(dir_count) + "_G_L"
        os.makedirs(dir_name)
        copyfile(chosen_combination_L[0], dir_name + "/1.jpg")
        copyfile(chosen_combination_L[1], dir_name + "/2.jpg")

        dir_count += 1

        # ------------------------------------------------------------------------------------------------------------
        # create a genuine pair with right side images from this ID
        # ------------------------------------------------------------------------------------------------------------
        # choose a combination of images from the right side
        possible_combinations_R = list(filter(lambda x : ("_R_" in x[0] and "_R_" in x[1]), list(combinations(v, 2))))
        chosen_combination_R = sample(possible_combinations_R, ((AMOUNT_OF_IMAGES // 2) // NUM_IDS) // 2)[0]

        # put the chosen images in an appropriate folder
        dir_name = "performance_evaluation_pairs/" + str(dir_count) + "_G_R"
        os.makedirs(dir_name)
        copyfile(chosen_combination_R[0], dir_name + "/1.jpg")
        copyfile(chosen_combination_R[1], dir_name + "/2.jpg")

        dir_count += 1

def create_impostor_pairs(images_per_ID, impostor_ID_combinations): # auxiliary function, creates impostor pairs

    global dir_count

    #########################################################################################################################
    # CREATE IMPOSTOR PAIRS
    #########################################################################################################################
    images_per_ID_list = list(images_per_ID.items())
    side = "_L_"
    for i in impostor_ID_combinations:
        for j in range(0, len(i), 2):
            
            # get two images (one from each ID) that have the same side
            image_ID_1 = sample(list(filter(lambda x : side in x, images_per_ID_list[i[j] - 1][1])), 1)[0]
            image_ID_2 = sample(list(filter(lambda x : side in x, images_per_ID_list[i[j + 1] - 1][1])), 1)[0]

            # put the chosen images in an appropriate folder
            dir_name = "performance_evaluation_pairs/" + str(dir_count) + "_I" + side[:-1]
            os.makedirs(dir_name)
            copyfile(image_ID_1, dir_name + "/1.jpg")
            copyfile(image_ID_2, dir_name + "/2.jpg")

            # ---------------------------------------------------------------------------------------------------------------
            # move the masks for the first image
            # ---------------------------------------------------------------------------------------------------------------
            base_mask = Image.open("performance_evaluation_pairs_masks/" + image_ID_1.split("/")[-1].replace(".jpg", ".png"))
            iris_mask_np = np.asarray(base_mask).copy()
            iris_mask_np[iris_mask_np[:, :, 1] != 255] = np.asarray([0, 0, 0, 255])
            iris_mask =  Image.fromarray(iris_mask_np.astype(np.uint8))

            eyebrow_mask_np = np.asarray(base_mask).copy()
            eyebrow_mask_np[eyebrow_mask_np[:, :, 1] != 50] = np.asarray([0, 0, 0, 255])
            eyebrow_mask_np[eyebrow_mask_np[:, :, 1] == 50] = np.asarray([255, 255, 255, 255])
            eyebrow_mask =  Image.fromarray(eyebrow_mask_np.astype(np.uint8))

            # copy the masks
            iris_mask.save(dir_name + "/iris_A.png")
            eyebrow_mask.save(dir_name + "/eyebrow_A.png")

            # ---------------------------------------------------------------------------------------------------------------
            # move the masks for the second image
            # ---------------------------------------------------------------------------------------------------------------
            base_mask = Image.open("performance_evaluation_pairs_masks/" + image_ID_2.split("/")[-1].replace(".jpg", ".png"))
            iris_mask_np = np.asarray(base_mask).copy()
            iris_mask_np[iris_mask_np[:, :, 1] != 255] = np.asarray([0, 0, 0, 255])
            iris_mask =  Image.fromarray(iris_mask_np.astype(np.uint8))

            eyebrow_mask_np = np.asarray(base_mask).copy()
            eyebrow_mask_np[eyebrow_mask_np[:, :, 1] != 50] = np.asarray([0, 0, 0, 255])
            eyebrow_mask_np[eyebrow_mask_np[:, :, 1] == 50] = np.asarray([255, 255, 255, 255])
            eyebrow_mask =  Image.fromarray(eyebrow_mask_np.astype(np.uint8))

            # copy the masks
            iris_mask.save(dir_name + "/iris_B.png")
            eyebrow_mask.save(dir_name + "/eyebrow_B.png")

            dir_count += 1
            
            # choose a side
            side = "_R_" if(side == "_L_") else "_L_"

if(__name__ == "__main__"):

    #####################################################################################################################################################################################
    # INITIAL SETUP
    #####################################################################################################################################################################################
    if(os.path.exists("performance_evaluation_pairs")): rmtree("performance_evaluation_pairs")
    os.makedirs("performance_evaluation_pairs")

    all_IDs = list(filter(lambda x : x[0] != ".", os.listdir("../../../dataset/dataset_test_ids")))
    NUM_IDS = len(all_IDs)

    # divide all the test IDs into 2 sets (one with 10 IDs and another with 5 IDs)
    directory_10 = sample(all_IDs, 10)
    directory_5 = [i for i in all_IDs if(i not in directory_10)]

    images_per_ID_10 = {}
    images_per_ID_5 = {}

    # get the images that each ID has, in the set with 10 IDs
    for i in directory_10:
        if(i == "not_used"): continue
        images_aux = list(map(lambda x : "../../../dataset/dataset_test_ids/" + i + "/" + x, list(filter(lambda x : x[0] != ".", os.listdir("../../../dataset/dataset_test_ids/" + i)))))
        images_aux_L = list(filter(lambda x : "_L_" in x, images_aux))
        images_aux_R = list(filter(lambda x : "_R_" in x, images_aux))
        
        images_per_ID_10[i] = sample(images_aux_L, 5) + sample(images_aux_R, 5)

    # get the images that each ID has, in the set with 5 IDs
    for i in directory_5:
        if(i == "not_used"): continue
        images_aux = list(map(lambda x : "../../../dataset/dataset_test_ids/" + i + "/" + x, list(filter(lambda x : x[0] != ".", os.listdir("../../../dataset/dataset_test_ids/" + i)))))
        images_aux_L = list(filter(lambda x : "_L_" in x, images_aux))
        images_aux_R = list(filter(lambda x : "_R_" in x, images_aux))
        
        images_per_ID_5[i] = sample(images_aux_L, 5) + sample(images_aux_R, 5)

    #######################################################################################
    # CREATE THE PERFORMANCE EVALUATION PAIRS
    #######################################################################################
    dir_count = 1
    
    # determine how the impostor pairs will be formed
    impostor_ID_combinations_10 = make_combinations(number_of_IDs = 10, number_of_sets = 7)
    impostor_ID_combinations_5 = make_combinations(number_of_IDs = 5, number_of_sets = 6)

    # actually create the genuine pairs
    create_genuine_pairs(images_per_ID_10)
    create_genuine_pairs(images_per_ID_5)

    # actually create the impostor pairs
    create_impostor_pairs(images_per_ID_10, impostor_ID_combinations_10)
    create_impostor_pairs(images_per_ID_5, impostor_ID_combinations_5)