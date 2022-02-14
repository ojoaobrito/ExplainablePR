import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import numpy as np
from PIL import Image

################################################
# CONTROL VARIABLES
################################################
SOURCE_DIR = "../../dataset/dataset_one_folder/"
FINAL_IMAGE_NAME = "../../dataset/IDs.jpg"

if(__name__ == "__main__"):   

    images_per_id = {}
    directory = list(filter(lambda x : x[0] != ".", os.listdir(SOURCE_DIR)))

    # get every ID
    for i in directory:
        if(not ((i.split("_")[0]) in images_per_id)): images_per_id.update({(i.split("_")[0]): [i]})
        else: images_per_id.update({(i.split("_")[0]): images_per_id[(i.split("_")[0])] + [i]})

    template = np.zeros((17 * 128, 16 * 128, 3))
    id_count = 1

    for i in range(17):
        for j in range(16):
            template[i * 128: ((i + 1) * 128), j * 128: ((j + 1) * 128), :] = np.asarray(Image.open(SOURCE_DIR + images_per_id["C" + str(id_count)][0]).resize((128, 128), Image.LANCZOS))
            id_count += 1
            if(id_count > len(images_per_id.keys())): break

    Image.fromarray(template.astype(np.uint8)).save(FINAL_IMAGE_NAME)