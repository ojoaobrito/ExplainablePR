import os, sys, csv

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import pickle
import itertools
import numpy as np
from shutil import rmtree
from operator import itemgetter
from PIL import Image, ImageOps
from random import sample, choice
from skimage.metrics import structural_similarity as ssim

####################################################################################
# CONTROL VARIABLES
####################################################################################
MODE = sys.argv[1] if(len(sys.argv) == 3) else "gan"
GENUINE_OR_IMPOSTOR = sys.argv[2] if(len(sys.argv) == 3) else "G"
TRAIN_PERCENTAGE = 0.9 if(MODE != "gan") else 1.0
VALIDATION_PERCENTAGE = 0.1 if(MODE != "gan") else 0.0
TEST_PERCENTAGE = 0.0 if(MODE != "gan") else 0.0
GAN_IMAGE_SIZE = 256
CNN_LIME_SHAP_IMAGE_SIZE = 256 # 224
NUM_GOOD_PAIRS = 4 # any value between [12, 432] that is divisible by both 2 and 3

def get_attributes(image_name): # auxiliary function, retrieves the attributes of a given image

    atts = []

    with open("../../dataset/annotations.csv", "r") as file:
        content = list(csv.reader(file, delimiter = ','))
        header = content[0][1:]
        
        for i in content[1:]:
            if(i[0] == image_name): 
                for jdx, j in enumerate(i[1:]):
                    if(j == "1"): atts.append(header[jdx])
    
    return(atts)

def initialize_master_ids_dict(subset): # auxiliary function, initializes the dictionary with every image per id (with additional information about the annotations)

    master_ids_dict = {}

    for i in ids[subset]:
        master_ids_dict.update({i[0]: {}}) # example: master_ids_dict = {C1: {"atts_1": [images], "atts_1_og_size": 30, "atts_1_round": 1, ...}}

        for j in i[1]:
            atts = get_attributes(j)
            if(not ((", ".join(atts)) in master_ids_dict[i[0]])): master_ids_dict[i[0]].update({", ".join(atts): [j], (", ".join(atts) + "_og_size"): 1, (", ".join(atts) + "_round"): 1})
            else: master_ids_dict[i[0]].update({", ".join(atts): master_ids_dict[i[0]][", ".join(atts)] + [j], (", ".join(atts) + "_og_size"): master_ids_dict[i[0]][", ".join(atts) + "_og_size"] + 1})

    return(master_ids_dict)

def get_score(base_id_atts, other_id_atts): # auxiliary function, computes the difference between both sets of attributes

    score = 0.0

    for idx, att in enumerate(other_id_atts):
        if(base_id_atts[idx] != att):

            try: score += ranking_dict[(base_id_atts[idx], att)]
            except: score += ranking_dict[(att, base_id_atts[idx])]

    return(score)

def get_eligible_ids(base_id, base_id_atts): # auxiliary function, retrieves the IDs that can be used to make impostor pairs with "base_id"

    eligible_ids = []
    
    for k, v in master_ids_dict.items():

        if(k == base_id): continue

        possible = []
        atts_aux = [i for i in v.keys() if((not ("_og_size" in i)) and (not ("_round" in i)))]
        
        for i in atts_aux:
            score = get_score(base_id_atts, i.split(", "))
            possible.append([k, i])

        if(possible == []): continue

        if(len(possible) != 1): # this ID has more than one type of attribute configuration
            chosen = ["", "", 0.0]
            for i in possible:
                if((len(v[i[1]]) / v[i[1] + "_og_size"]) > chosen[2]): chosen = [i[0], i[1], (len(v[i[1]]) / v[i[1] + "_og_size"])]

        else: chosen = [possible[0][0], possible[0][1], (len(v[possible[0][1]]) / v[possible[0][1] + "_og_size"])]

        eligible_ids.append(chosen)

    return(eligible_ids)

if(__name__ == "__main__"): 

    #############################################################################################################
    # GATHER THE IDS AND SOME IMAGES FOR EACH ONE
    #############################################################################################################
    ids_aux = {}

    # get every ID
    for i in list(filter(lambda x : x[0] != ".", os.listdir("../../dataset/dataset_one_folder"))):
        if(not (i.split("_")[0] in ids_aux)): ids_aux.update({i.split("_")[0]: [i]})
        else: ids_aux.update({i.split("_")[0]: ids_aux[i.split("_")[0]] + [i]})

    IDS_BASE = []
    for k, v in ids_aux.items(): IDS_BASE.append([k, v])

    # separate the IDs into 3 subsets: train, validation and test
    train = sample(IDS_BASE, int(TRAIN_PERCENTAGE * len(IDS_BASE)))
    validation = sample([i for i in IDS_BASE if(not (i in train))], int(VALIDATION_PERCENTAGE * len(IDS_BASE)))
    test = [i for i in IDS_BASE if(not (i in train) and not (i in validation))]
    
    if(MODE == "gan"): ids = {"train_pairs": train + validation + test, "validation_pairs": [], "test_pairs": []}
    else: ids = {"train_pairs": train, "validation_pairs": validation, "test_pairs": test}
    
    # load the annotations ranking
    with open("attribute_similarity_ranking.csv", "r") as file:
        ranking = list(csv.reader(file, delimiter = ','))[1:]

    ranking_dict = {}
    for i in ranking: ranking_dict.update({(i[0], i[1]): float(i[2])})

    ##############################################################################################
    # CREATE SOME DIRECTORIES
    ##############################################################################################
    if(os.path.exists("../../dataset/data")): rmtree("../../dataset/data")
    os.makedirs("../../dataset/data")

    if((GENUINE_OR_IMPOSTOR != "G") or (MODE != "gan")): os.makedirs("../../dataset/data/train/0")
    if((GENUINE_OR_IMPOSTOR != "I") or (MODE != "gan")): os.makedirs("../../dataset/data/train/1")

    if(MODE != "gan"):
        os.makedirs("../../dataset/data/validation/0")
        os.makedirs("../../dataset/data/validation/1")
        os.makedirs("../../dataset/data/test/0")
        os.makedirs("../../dataset/data/test/1")
        
    #####################################################################################################################################################################################################################################################
    # MAKE THE IMAGE PAIRS
    #####################################################################################################################################################################################################################################################
    for subset in ["train_pairs", "validation_pairs"]: # for each of the 3 subsets (training, validation and test)
        
        print("Preparing the IDs...")

        if(GENUINE_OR_IMPOSTOR != "G"):

            master_ids_dict = initialize_master_ids_dict(subset)

            with open("dict.pickle", "wb") as file:
                pickle.dump(master_ids_dict, file, protocol = pickle.HIGHEST_PROTOCOL)

        print("Done preparing the IDs!")

        for idx, i in enumerate(ids[subset]): # for every ID in that subset
            id_pairs = []

            if(len(i[1]) < 4): continue
            NUM_GOOD_PAIRS = 4 if(len(i[1]) == 4) else 204
            
            # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # if required, create genuine pairs (same ID)
            # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
            if(GENUINE_OR_IMPOSTOR != "I"):

                possible_good_pairs = list(itertools.permutations(i[1], 2))
                ssim_scores = {"left_left": [], "right_right": []}
                side_that_flips = "L"

                # for every possible good pair, store it according to the side configuration (L + L, R + R or otherwise) and the corresponding SSIM score
                for j in possible_good_pairs:
                    img_a = Image.open("../../dataset/dataset_one_folder/" + j[0]).convert("L")
                    img_b = Image.open("../../dataset/dataset_one_folder/" + j[1]).convert("L")

                    if((("L" in j[0]) and ("R" in j[1])) or (("R" in j[0]) and ("L" in j[1]))): # this pair has images from different sides

                        index_image_keep_side = 0 if((side_that_flips == "L" and ("R" in j[0])) or (side_that_flips == "R" and ("L" in j[0]))) else 1
                        index_image_change_side = 0 if(index_image_keep_side == 1) else 1
                        
                        img_a = np.asarray(ImageOps.mirror(img_a)) if(index_image_change_side == 0) else np.asarray(img_a)
                        img_b = np.asarray(ImageOps.mirror(img_b)) if(index_image_change_side == 1) else np.asarray(img_b)

                        score = ssim(img_a, img_b, data_range = (img_b.max() - img_b.min()))

                        if(side_that_flips == "R"): ssim_scores.update({"left_left": ssim_scores["left_left"] + [(j[0], j[1], score, index_image_change_side)]})
                        else: ssim_scores.update({"right_right": ssim_scores["right_right"] + [(j[0], j[1], score, index_image_change_side)]})

                        current_flip = "R" if(side_that_flips == "L") else "L"
                    
                    else: # this pair has images from the same side

                        img_a = np.asarray(img_a)
                        img_b = np.asarray(img_b)

                        score = ssim(img_a, img_b, data_range = (img_b.max() - img_b.min()))

                        # store this pair and its SSIM score
                        if(("L" in j[0]) and ("L" in j[1])): ssim_scores.update({"left_left": ssim_scores["left_left"] + [(j[0], j[1], score)]})
                        elif(("R" in j[0]) and ("R" in j[1])): ssim_scores.update({"right_right": ssim_scores["right_right"] + [(j[0], j[1], score)]})
                        
                # sort the pairs by their SSIM scores (ascending order)
                ssim_scores["left_left"] = sorted(ssim_scores["left_left"], key = lambda x : x[2])
                ssim_scores["right_right"] = sorted(ssim_scores["right_right"], key = lambda x : x[2])

                # choose equidistant pairs (to ensure an even distribution of side configurations and scores)
                l_l_chosen = list(itemgetter(*(np.linspace(0, len(ssim_scores["left_left"]) - 1, num = (NUM_GOOD_PAIRS // 2)).astype(int).tolist()))(ssim_scores["left_left"]))
                r_r_chosen = list(itemgetter(*(np.linspace(0, len(ssim_scores["right_right"]) - 1, num = (NUM_GOOD_PAIRS // 2)).astype(int).tolist()))(ssim_scores["right_right"]))
                
                # bring it all together
                good_pairs = l_l_chosen + r_r_chosen

                id_pairs = good_pairs

            # ----------------------------------------------------------------------------------------------------------------------------------------------
            # if required, create impostor pairs (different IDs)
            # ----------------------------------------------------------------------------------------------------------------------------------------------
            if(GENUINE_OR_IMPOSTOR != "G"):

                bad_pairs = []
                current_image_index = 0
                
                count = 0
                while(count < NUM_GOOD_PAIRS):
                    
                    # -------------------------------------------------------
                    # choose an image from the base ID and get its attributes
                    # -------------------------------------------------------
                    chosen_base_id_image = i[1][current_image_index]

                    # get the attributes of the base ID's image
                    atts = get_attributes(chosen_base_id_image)

                    # --------------------------------------------------------------------------------------------------------------------------------------
                    # get every ID that can be matched with this ID to create an impostor pair with image at index "current_image_index"
                    # --------------------------------------------------------------------------------------------------------------------------------------
                    eligible_ids = get_eligible_ids(i[0], atts)
                    chosen_id = choice(sorted(eligible_ids, key = lambda x : x[2], reverse = True))
                    
                    try:
                        base_id_image_side = "L" if("L" in chosen_base_id_image) else "R"
                        other_id_image = list(filter(lambda x : ("_" + base_id_image_side + "_") in x, (master_ids_dict[chosen_id[0]][chosen_id[1]]))).pop()
                    
                    # there are no images left to create bad pairs with
                    except Exception as e:
                        with open("dict.pickle", "rb") as file:
                            master_ids_dict = pickle.load(file)
                        continue

                    # update the image counter
                    current_image_index = 0 if(current_image_index == (len(i[1]) - 1)) else (current_image_index + 1)

                    # ----------------------------------------------------------------------------------------
                    # save the generated pair
                    # ----------------------------------------------------------------------------------------
                    pair = (chosen_base_id_image, other_id_image, get_score(atts, (chosen_id[1].split(", "))))

                    bad_pairs.append(pair)
                    count += 1

                # save all the pairs we've created
                for j in bad_pairs: id_pairs.append(j)
            
            print("ID " + str(idx + 1) + "/" + str(len(ids[subset])))
            
            # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # actually make the image pairs
            # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            for j in id_pairs:
                    
                # wrong image pairs
                if(((j[0].split("_")[0]) != (j[1].split("_")[0]))): dir_name = "0"

                # right image pairs
                else: dir_name = "1"

                # concatenate the images depthwise
                img1 = Image.open("../../dataset/dataset_one_folder/" + j[0])
                img2 = Image.open("../../dataset/dataset_one_folder/" + j[1])
                
                flipped = ""
                if(len(j) == 4): # one of the images needs to be flipped
                    if(j[3] == 0): 
                        img1 = ImageOps.mirror(img1)
                        flipped = "+flipped0"
                    else: 
                        img2 = ImageOps.mirror(img2)
                        flipped = "+flipped1"

                # build the final filename
                if(dir_name == "0"): final_name = "../../dataset/data/" + subset.split("_")[0] + "/" + dir_name + "/" + j[0].replace(".jpg", "") + "_+_" + j[1].replace(".jpg", "+score" + str(j[2]) + flipped)
                else: final_name = "../../dataset/data/" + subset.split("_")[0] + "/" + dir_name + "/" + j[0].replace(".jpg", "") + "_+_" + j[1].replace(".jpg", "+ssim-score" + str(round(j[2], 2)) + flipped)
                
                if(MODE != "cnn_lime_shap"): 
                    # build the pair
                    pair = np.dstack((np.asarray(img1), np.asarray(img2)))
                    
                    # save the pair
                    np.save(final_name + ".npy", pair)

                else: 
                    # build the pair
                    pair = np.zeros((CNN_LIME_SHAP_IMAGE_SIZE, CNN_LIME_SHAP_IMAGE_SIZE, 3))
                    pair[
                        (CNN_LIME_SHAP_IMAGE_SIZE // 4):((CNN_LIME_SHAP_IMAGE_SIZE // 4) + (CNN_LIME_SHAP_IMAGE_SIZE // 2)), 
                        :, 
                        :
                    ] = np.column_stack((np.asarray(img1.resize((CNN_LIME_SHAP_IMAGE_SIZE // 2, CNN_LIME_SHAP_IMAGE_SIZE // 2), Image.LANCZOS)), np.asarray(img2.resize((CNN_LIME_SHAP_IMAGE_SIZE // 2, CNN_LIME_SHAP_IMAGE_SIZE // 2), Image.LANCZOS))))
                    
                    # save the pair
                    Image.fromarray(pair.astype(np.uint8)).save(final_name + ".jpg")
        
        # the GAN only needs training data
        if(MODE == "gan"): break
    
    if(os.path.exists("dict.pickle")): os.remove("dict.pickle")