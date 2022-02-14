import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import numpy as np
from PIL import Image
from random import sample
from shutil import rmtree
import scikitplot as skplt
from sklearn import metrics
from statistics import stdev
from natsort import natsorted
import matplotlib.pyplot as plt
from random import shuffle
from itertools import combinations
from scipy import interp
from random import sample
np.set_printoptions(suppress = True)
np.set_printoptions(threshold = sys.maxsize)

###########################################################
# CONTROL VARIABLES
###########################################################
BOOTSTRAP_ITERATIONS = 10
BOOTSTRAP_KEEP_PERCENTAGE = 0.9
TRAINING_PERCENTAGE = 0.8
TEST_PERCENTAGE = 0.2
DATA_PATH = "../../../dataset/ubiris_v2/dataset_one_folder"
NUM_GOOD_PAIRS = 85
WORLD_SETTING = "open" # either "closed" or "open"

def check_for_errors(r_value): # auxiliary function, checks for errors when running the main scripts

    if(r_value != 0):
        print("ERROR FOUND, STOPPING NOW...")
        sys.exit()

if(__name__  == "__main__"):

    min_len = sys.maxsize

    ################################################################################################################################################################
    # GATHER THE DATASET
    ################################################################################################################################################################
    images = natsorted(list(filter(lambda x : x[0] != ".", os.listdir(DATA_PATH))))
    ids_images = {}
    id_counter = 0
    current_id = "0"

    for i in images:
        if(i.split(".")[-1] != "tiff"): continue
        if(int(i.split("_")[0][1:]) % 2 == 1 and (i.split("_")[0][1:] != current_id)): # the ID is odd and has changed, so let's increment the "id_counter" variable
            id_counter += 1
            current_id = i.split("_")[0][1:]

        if(id_counter in ids_images.keys()): ids_images[id_counter] += [i]
        else: ids_images[id_counter] = [i]

    AUCs = []
    EERs = []

    ###########################################################################################################################################################################################################################################################################
    # PERFORM THE BOOTSTRAP STRATEGY
    ###########################################################################################################################################################################################################################################################################
    fprs = []
    tprs = []
    for i in range(BOOTSTRAP_ITERATIONS):
        '''chosen_ids = sample(ids_images.keys(), int(BOOTSTRAP_KEEP_PERCENTAGE * len(ids_images.keys())))

        if(WORLD_SETTING == "closed"):
            training_chosen_ids = chosen_ids.copy()
            test_chosen_ids = sample(chosen_ids, int(TEST_PERCENTAGE * len(chosen_ids)))

        else:
            training_chosen_ids = sample(chosen_ids, int(TRAINING_PERCENTAGE * len(chosen_ids)))
            test_chosen_ids = list(filter(lambda x : x not in training_chosen_ids, chosen_ids))

        pair_number = 1
        if(os.path.exists("/".join(DATA_PATH.split("/")[:-1]) + "/data")): rmtree("/".join(DATA_PATH.split("/")[:-1]) + "/data")
        os.makedirs("/".join(DATA_PATH.split("/")[:-1]) + "/data")
        os.makedirs("/".join(DATA_PATH.split("/")[:-1]) + "/data/train/0")
        os.makedirs("/".join(DATA_PATH.split("/")[:-1]) + "/data/train/1")
        os.makedirs("/".join(DATA_PATH.split("/")[:-1]) + "/data/test/0")
        os.makedirs("/".join(DATA_PATH.split("/")[:-1]) + "/data/test/1")

        for jdx, j in enumerate(training_chosen_ids):

            print("TRAINING IDS: " + str(jdx + 1) + "/" + str(len(training_chosen_ids)))

            # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # create the genuine pairs
            # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            possible_combinations = list(combinations(ids_images[j], 2))
            filtered_possible_combinations = list(filter(lambda x : ((int(x[0].split("_")[0][1:]) % 2 == 0 and int(x[1].split("_")[0][1:]) % 2 == 0) or (int(x[0].split("_")[0][1:]) % 2 == 1 and int(x[1].split("_")[0][1:]) % 2 == 1)), possible_combinations))
            
            chosen_combinations = sample(filtered_possible_combinations, NUM_GOOD_PAIRS)

            for k in chosen_combinations:

                image_A_np = np.asarray(Image.open(DATA_PATH + "/" + k[0]))
                image_B_np = np.asarray(Image.open(DATA_PATH + "/" + k[1]))

                pair = np.dstack((image_A_np, image_B_np))

                np.save("/".join(DATA_PATH.split("/")[:-1]) + "/data/train/1/C" + str(j) + "_+_C" + str(j) + "_" + str(pair_number) + ".npy", pair)
                pair_number += 1

            # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # create the impostor pairs
            # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            already_saved_pairs = []
            k = 0
            while(k < NUM_GOOD_PAIRS):
                image_A = sample(ids_images[j], 1)[0]
                
                random_id = sample(list(filter(lambda x : x != j, ids_images.keys())), 1)[0]
                
                image_B = sample(list(filter(lambda x : (x != image_A) and ((int(x.split("_")[0][1:]) % 2 == 0 and int(image_A.split("_")[0][1:]) % 2 == 0) or (int(x.split("_")[0][1:]) % 2 == 1 and int(image_A.split("_")[0][1:]) % 2 == 1)), ids_images[random_id])), 1)[0]

                if((image_A, image_B) in already_saved_pairs): continue
                already_saved_pairs.append((image_A, image_B))

                image_A_np = np.asarray(Image.open(DATA_PATH + "/" + image_A))
                image_B_np = np.asarray(Image.open(DATA_PATH + "/" + image_B))

                pair = np.dstack((image_A_np, image_B_np))

                np.save("/".join(DATA_PATH.split("/")[:-1]) + "/data/train/0/C" + str(j) + "_+_C" + str(random_id) + "_" + str(pair_number) + ".npy", pair)
                pair_number += 1
                k += 1

        for jdx, j in enumerate(test_chosen_ids):

            print("TEST IDS: " + str(jdx + 1) + "/" + str(len(test_chosen_ids)))

            # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # create the genuine pairs
            # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            possible_combinations = list(combinations(ids_images[j], 2))
            filtered_possible_combinations = list(filter(lambda x : ((int(x[0].split("_")[0][1:]) % 2 == 0 and int(x[1].split("_")[0][1:]) % 2 == 0) or (int(x[0].split("_")[0][1:]) % 2 == 1 and int(x[1].split("_")[0][1:]) % 2 == 1)), possible_combinations))
            
            chosen_combinations = sample(filtered_possible_combinations, NUM_GOOD_PAIRS)

            for k in chosen_combinations:

                image_A_np = np.asarray(Image.open(DATA_PATH + "/" + k[0]))
                image_B_np = np.asarray(Image.open(DATA_PATH + "/" + k[1]))

                pair = np.dstack((image_A_np, image_B_np))

                np.save("/".join(DATA_PATH.split("/")[:-1]) + "/data/test/1/C" + str(j) + "_+_C" + str(j) + "_" + str(pair_number) + ".npy", pair)
                pair_number += 1

            # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # create the impostor pairs
            # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            already_saved_pairs = []
            k = 0
            while(k < NUM_GOOD_PAIRS):
                image_A = sample(ids_images[j], 1)[0]
                
                random_id = sample(list(filter(lambda x : x != j, ids_images.keys())), 1)[0]
                
                image_B = sample(list(filter(lambda x : (x != image_A) and ((int(x.split("_")[0][1:]) % 2 == 0 and int(image_A.split("_")[0][1:]) % 2 == 0) or (int(x.split("_")[0][1:]) % 2 == 1 and int(image_A.split("_")[0][1:]) % 2 == 1)), ids_images[random_id])), 1)[0]

                if((image_A, image_B) in already_saved_pairs): continue
                already_saved_pairs.append((image_A, image_B))

                image_A_np = np.asarray(Image.open(DATA_PATH + "/" + image_A))
                image_B_np = np.asarray(Image.open(DATA_PATH + "/" + image_B))

                pair = np.dstack((image_A_np, image_B_np))

                np.save("/".join(DATA_PATH.split("/")[:-1]) + "/data/test/0/C" + str(j) + "_+_C" + str(random_id) + "_" + str(pair_number) + ".npy", pair)
                pair_number += 1
                k += 1
                
        # -----------------------------------------
        # train the CNN
        # -----------------------------------------
        print("\nTRAINING THE CNN...")
        r_value = os.system("python3 train_cnn.py")
        check_for_errors(r_value)
        print("DONE TRAINING THE CNN!")'''
        
        # --------------------------------------------------------------------------------------------------------------------------
        # compute the AUC and EER
        # --------------------------------------------------------------------------------------------------------------------------
        bootstrap_outputs = natsorted(list(filter(lambda x : (x[0] != ".") and ("densenet" in x), os.listdir("outputs/bootstrap"))))
        y = np.load("outputs/bootstrap/" + bootstrap_outputs[i] + "/ground_truth.npy")
        pred = np.load("outputs/bootstrap/" + bootstrap_outputs[i] + "/predicted.npy")

        # AUC
        '''
        print(pred[:, 1])
        print(np.where(pred[:, 1] > 0.5, 1, 0))
        sys.exit()'''
        fpr, tpr, thresholds = metrics.roc_curve(np.argmax(y, axis = 1), pred[:, 1], pos_label = 1, drop_intermediate = False)
        
        AUC = metrics.auc(fpr, tpr)
        
        fprs.append(fpr)
        base_fpr = np.linspace(0, 1, len(fpr))

        if(len(fpr) < min_len): min_len = len(fpr)

        tpr = interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)

        # EER
        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]

        EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

        # save everything
        AUCs.append(AUC)
        EERs.append(EER)
        
    #
    #
    #
    new_tprs = []
    for i in tprs:
        #new_tprs.append(sorted(sample(list(i), min_len)))
        new_tprs.append(i[np.linspace(0, len(i) - 1, min_len).astype(int)])

    tprs = np.asarray(new_tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    base_fpr = np.linspace(0, 1, len(mean_tprs))

    plt.plot(base_fpr, mean_tprs, 'b', label="Ours")
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.4)

    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.005, 1.005])
    plt.ylim([-0.005, 1.005])
    plt.legend(loc="lower right")
    plt.title("ROC Curve")
    plt.ylabel('True Positive Rate (TPR)')
    plt.xlabel('False Positive Rate (FPR)')
    plt.savefig("outputs/bootstrap/roc_curve.png", dpi = 200)

    ########################################################################################
    # COMPUTE THE MEAN AUC AND EER, AS WELL AS, THEIR RESPECTIVE STANDARD DEVIATIONS
    ########################################################################################
    mean_AUC = np.mean(np.asarray(AUCs))
    std_AUC = stdev(np.asarray(AUCs))

    mean_EER = np.mean(np.asarray(EERs))
    std_EER = stdev(np.asarray(EERs))

    with open("outputs/bootstrap/AUC_and_EER_" + WORLD_SETTING + "_world.csv", "w") as file:
        file.write("Mean AUC," + str(mean_AUC) + "\n")
        file.write("Stand. dev. AUC," + str(std_AUC) + "\n")
        file.write("Mean EER," + str(mean_EER) + "\n")
        file.write("Stand. dev. EER," + str(std_EER))