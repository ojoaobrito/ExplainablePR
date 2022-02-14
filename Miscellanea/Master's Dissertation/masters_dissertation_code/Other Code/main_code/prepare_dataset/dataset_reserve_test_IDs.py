import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

from shutil import move, rmtree

###################################################
# CONTROL VARIABLES
###################################################
TEST_IDS_FILE = "test_IDs.txt"
SOURCE_DIR = "../../dataset/dataset_images_per_id/"
DESTINATION_DIR = "../../dataset/dataset_test_ids/"

if(__name__ == "__main__"):

    # get the reserved test IDs
    with open(TEST_IDS_FILE, "r") as file:
        test_ids = list(map(lambda x : x.split("\n")[0], file.readlines()))

    if(os.path.exists(DESTINATION_DIR)): rmtree(DESTINATION_DIR)
    os.makedirs(DESTINATION_DIR)

    directory = list(filter(lambda x : x[0] != ".", os.listdir(SOURCE_DIR)))

    # move the directories
    for i in directory:
        if(i in test_ids): move(SOURCE_DIR + i, DESTINATION_DIR + i)
            