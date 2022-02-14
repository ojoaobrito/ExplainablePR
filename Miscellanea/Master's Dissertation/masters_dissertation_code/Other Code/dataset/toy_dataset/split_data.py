import os, sys
from shutil import rmtree, copyfile
from PIL import Image

######################################################################################
# HYPERPARAMETERS
######################################################################################
TRAINING_GAN = True
TRAIN_PERCENTAGE = 0.8 if(not TRAINING_GAN) else 1.0
VALIDATION_PERCENTAGE = 0.1 if(not TRAINING_GAN) else 0.0
TEST_PERCENTAGE = 0.1 if(not TRAINING_GAN) else 0.0

if __name__ == "__main__":

    directory = list(filter(lambda x : x[0] != ".", os.listdir("imgs")))

    if(os.path.exists("data")): rmtree("data")
    os.makedirs("data")

    os.makedirs("data/train/0")
    
    if(not TRAINING_GAN):
        os.makedirs("data/validation/0")
        os.makedirs("data/test/0")

    for idx, i in enumerate(directory):

        # this sample goes to the training subset
        if(TRAINING_GAN or ((idx/len(directory)) <= TRAIN_PERCENTAGE)): copyfile("imgs/" + i, "data/train/0/" + i)

        # this sample goes to the validation subset
        elif((idx/len(directory)) > TRAIN_PERCENTAGE and ((idx/len(directory)) <= (TRAIN_PERCENTAGE + VALIDATION_PERCENTAGE))): copyfile("imgs/" + i, "data/validation/0/" + i)

        # this sample goes to the test subset
        else: copyfile("imgs/" + i, "data/test/0/" + i)