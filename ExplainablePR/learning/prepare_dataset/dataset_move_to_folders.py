import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

from random import sample
from shutil import rmtree, copyfile

####################################################################################################
# CONTROL VARIABLES
####################################################################################################
# the values are the default ones (NOTE: if the values are to be changed, do so in this "if" branch)
if(len(sys.argv) == 1):
    MODE = "regular"
    TRAIN_PERCENTAGE = 0.8
    VALIDATION_PERCENTAGE = 0.1
    TEST_PERCENTAGE = 0.1

# the values are coming from the "run_preprocessing_idea_paper.py" master script
else:
    MODE = sys.argv[1]
    TRAIN_PERCENTAGE = float(sys.argv[2])
    VALIDATION_PERCENTAGE = float(sys.argv[3])
    TEST_PERCENTAGE = float(sys.argv[4])

if(__name__ == "__main__"):

    directory = list(filter(lambda x : x[0] != ".", os.listdir("../../dataset/dataset_one_folder")))

    ###########################################################################################################
    # GATHER THE IDS AND SOME IMAGES FOR EACH ONE
    ###########################################################################################################
    ids_dict = {}

    # get every ID
    for i in list(filter(lambda x : x[0] != ".", os.listdir("../../dataset/dataset_one_folder"))):
        if(not (i.split("_")[0] in ids_dict)): ids_dict.update({i.split("_")[0]: [i]})
        else: ids_dict.update({i.split("_")[0]: ids_dict[i.split("_")[0]] + [i]})

    IDS_BASE = []
    for k, v in ids_dict.items(): IDS_BASE.append([k, v])

    # separate the IDs into 3 subsets: train, validation and test
    train = sample(IDS_BASE, int(TRAIN_PERCENTAGE * len(IDS_BASE)))
    validation = sample([i for i in IDS_BASE if(not (i in train))], int(VALIDATION_PERCENTAGE * len(IDS_BASE)))
    test = [i for i in IDS_BASE if(not (i in train) and not (i in validation))]
    
    if(MODE == "gan"): ids = {"train": train + validation + test, "validation": [], "test": []}
    else: ids = {"train": train, "validation": validation, "test": test}

    # create some directories
    if(os.path.exists("../../dataset/data")): rmtree("../../dataset/data")

    if(MODE == "gan"): os.makedirs("../../dataset/data/train/0")
    else:
        for i in ["train/", "validation/", "test/"]:
            if(MODE == "cnn_side"):
                for j in ["0", "1"]: os.makedirs("../../dataset/data/" + i + j)
            else:
                os.makedirs("../../dataset/data/" + i + "0")

    #####################################################################################################################################################################
    # MOVE THE IMAGES
    #####################################################################################################################################################################
    for subset in ["train", "validation", "test"]: # for each of the 3 subsets (training, validation and test)

        for i in ids[subset]: # for every ID in that subset
            
            for j in i[1]: # for every image of that ID
                
                if(MODE == "cnn_side"): copyfile("../../dataset/dataset_one_folder/" + j, "../../dataset/data/" + subset + "/" + ("0" if("_L_" in j) else "1") + "/" + j)
                else: copyfile("../../dataset/dataset_one_folder/" + j, "../../dataset/data/" + subset + "/0/" + j)

        # the GAN only needs training data
        if(MODE == "gan"): break