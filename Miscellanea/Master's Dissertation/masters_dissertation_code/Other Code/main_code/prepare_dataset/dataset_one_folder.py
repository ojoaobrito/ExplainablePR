import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

from shutil import copyfile, rmtree

#######################################################################################
# CONTROL VARIABLES
#######################################################################################
SOURCE_DIR = sys.argv[1] if(len(sys.argv) == 2) else "../../dataset/annotations_merged"

def dataset_one_folder(current): # main function, moves every image to the same folder

    if(os.path.isdir(current)): # it's a directory, let's explore it
        for i in list(filter(lambda x : x[0] != ".", os.listdir(current))):
            dataset_one_folder(current + "/" + i)

    else: # it's a file, let's do something with it
        # exclude the FFHQ samples
        if(len(current.split("/")[-1].split("_")) <= 3): return
        
        copyfile(current, "../../dataset/dataset_one_folder/" + current.split("/")[-1])

if(__name__ == "__main__"):

    # create a common directory (and remove an existing one, if needed)
    if(os.path.exists("../../dataset/dataset_one_folder")): rmtree("../../dataset/dataset_one_folder")
    os.makedirs("../../dataset/dataset_one_folder")

    dataset_one_folder(SOURCE_DIR)