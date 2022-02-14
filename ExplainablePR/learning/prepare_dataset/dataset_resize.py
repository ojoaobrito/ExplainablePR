import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

from PIL import Image

########################################################################################
# CONTROL VARIABLES
########################################################################################
IMAGE_SIZE = (int(sys.argv[1]), int(sys.argv[1])) if(len(sys.argv) != 1) else (128, 128)

def dataset_resize(current):

    if(os.path.isdir(current)): # it's a directory, let's explore it
        for i in list(filter(lambda x : x[0] != ".", os.listdir(current))):
            dataset_resize(current + "/" + i)

    else: # it's a file, let's do something with it

        img = Image.open(current)
        img = img.resize(IMAGE_SIZE,Image.ANTIALIAS)
        img.save(current)

if(__name__ == "__main__"):

    dataset_resize("../../dataset/dataset_one_folder")