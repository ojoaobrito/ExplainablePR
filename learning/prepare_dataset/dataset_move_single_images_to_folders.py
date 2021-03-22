import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

from shutil import rmtree, copyfile
from natsort import natsorted

###################################################
# CONTROL VARIABLES
###################################################
SOURCE_DIR = "../../dataset/dataset_images_per_id/"
DESTINATION_DIR = "../../dataset/data/"

if(__name__ == "__main__"): 

    if(os.path.exists(DESTINATION_DIR)): rmtree(DESTINATION_DIR)
    os.makedirs(DESTINATION_DIR)

    grouped_ids = natsorted(list(filter(lambda x : x[0] != ".", os.listdir(SOURCE_DIR))))
    image_count = 1

    for i in grouped_ids:
        
        if(int(i[1:]) >= 260): break
        
        id_images = list(filter(lambda x : x[0] != ".", os.listdir(SOURCE_DIR + i)))

        for j in id_images:
            copyfile(SOURCE_DIR + i + "/" + j, DESTINATION_DIR + str(image_count) + "_" + str(int(i[1:]) - 1) + ".jpg")
            image_count += 1