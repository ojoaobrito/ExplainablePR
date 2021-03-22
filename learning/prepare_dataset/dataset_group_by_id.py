import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

from shutil import rmtree, copyfile

########################################################
# CONTROL VARIABLES
########################################################
SOURCE_DIR = "../../dataset/dataset_one_folder/"
DESTINATION_DIR = "../../dataset/dataset_images_per_id/"

if(__name__ == "__main__"): 

    images_per_id = {}
    directory = list(filter(lambda x : x[0] != ".", os.listdir(SOURCE_DIR)))

    # get every ID
    for i in directory:

        if(int(i.split("_")[0].split("C")[1]) > 259): continue

        if(not ((i.split("_")[0]) in images_per_id)): images_per_id.update({(i.split("_")[0]): [i]})
        else: images_per_id.update({(i.split("_")[0]): images_per_id[(i.split("_")[0])] + [i]})

    if(os.path.exists(DESTINATION_DIR)): rmtree(DESTINATION_DIR)
    os.makedirs(DESTINATION_DIR)
    
    # move the images to their respective folders
    for k in images_per_id.keys():
        os.makedirs(DESTINATION_DIR + k)
        for v in images_per_id[k]:
            copyfile(SOURCE_DIR + v, DESTINATION_DIR + k + "/" + v)