import os, sys, csv

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import argparse
import numpy as np
from PIL import Image
from shutil import rmtree
from random import shuffle
from natsort import natsorted

#####################################################################
# CONTROL VARIABLES
#####################################################################
IMAGE_SIZE = 224
ADD_BLACK_BARS = True
IMAGES_PATH = "our_data/images/"
PROCESSED_IMAGES_PATH = "our_data/processed_images/"
TEXT_CAPTIONS_PATH = "our_data/text_captions.csv"
PROCESSED_TEXT_CAPTIONS_PATH = "our_data/processed_text_captions.csv"

if(__name__ == "__main__"):

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='depthwise' , choices=["depthwise", "side_by_side"] , help='image mode')
    args = parser.parse_args()

    if(os.path.exists(PROCESSED_IMAGES_PATH)): rmtree(PROCESSED_IMAGES_PATH)
    os.makedirs(PROCESSED_IMAGES_PATH)

    # load the original images
    images = natsorted(list(filter(lambda x : x[0] != ".", os.listdir(IMAGES_PATH))))

    # load the original captions
    with open(TEXT_CAPTIONS_PATH, "r") as file:
        captions = file.read().splitlines()

    with open(PROCESSED_TEXT_CAPTIONS_PATH, "w") as file:
        
        captions = captions[1:]
        shuffle(captions)

        image_name_counter = 1
        for idx, i in enumerate(captions):

            print(str(idx + 1) + "/" + str(len(images)))
            
            # load the original image
            img_np = np.asarray(Image.open(IMAGES_PATH + i.split(",")[0])).copy()
            img_A_np = img_np[:, :256, :]
            img_B_np = img_np[:, 256:, :]

            if(args.mode == "side_by_side"):

                # create an inverted copy of the original image (i.e., A and B switched)
                img_reversed_np = img_np.copy()
                img_reversed_np[:, :256, :] = img_B_np
                img_reversed_np[:, 256:, :] = img_A_np

                if(ADD_BLACK_BARS):
                    black_template = np.zeros((512, 512, 3))
                    black_template[128:384, :, :] = img_np
                    img_np = black_template.copy()

                    black_template = np.zeros((512, 512, 3))
                    black_template[128:384, :, :] = img_reversed_np
                    img_reversed_np = black_template.copy()

                # convert the numpy arrays into regular images
                img = Image.fromarray(img_np.astype(np.uint8)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
                img_reversed = Image.fromarray(img_reversed_np.astype(np.uint8)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

                # AB + first caption
                img.save(PROCESSED_IMAGES_PATH + str(image_name_counter) + ".jpg")
                file.write(captions[idx].split(",")[2].replace("<COMMA>", ",") + "\n")
                image_name_counter += 1

                '''# AB + second caption
                img.save(PROCESSED_IMAGES_PATH + str(image_name_counter) + ".jpg")
                file.write(captions[idx].split(",")[3].replace("<COMMA>", ",") + "\n")
                image_name_counter += 1'''

                '''# BA + first caption
                img_reversed.save(PROCESSED_IMAGES_PATH + str(image_name_counter) + ".jpg")
                file.write(captions[idx].split(",")[2].replace("<COMMA>", ",") + "\n")
                image_name_counter += 1'''

                # BA + second caption
                img_reversed.save(PROCESSED_IMAGES_PATH + str(image_name_counter) + ".jpg")
                file.write(captions[idx].split(",")[3].replace("<COMMA>", ",") + "\n")
                image_name_counter += 1

            else:
                img_A_np = Image.fromarray(img_A_np.astype(np.uint8)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
                img_B_np = Image.fromarray(img_B_np.astype(np.uint8)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

                img_np = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 6))
                img_np[:, :, :3] = img_A_np
                img_np[:, :, 3:] = img_B_np

                img_reversed_np = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 6))
                img_reversed_np[:, :, :3] = img_B_np
                img_reversed_np[:, :, 3:] = img_A_np

                # AB + first caption
                np.save(PROCESSED_IMAGES_PATH + str(image_name_counter) + ".npy", img_np)
                file.write(captions[idx].split(",")[2].replace("<COMMA>", ",") + "\n")
                image_name_counter += 1

                # AB + second caption
                np.save(PROCESSED_IMAGES_PATH + str(image_name_counter) + ".npy", img_np)
                file.write(captions[idx].split(",")[3].replace("<COMMA>", ",") + "\n")
                image_name_counter += 1

                # BA + first caption
                np.save(PROCESSED_IMAGES_PATH + str(image_name_counter) + ".npy", img_reversed_np)
                file.write(captions[idx].split(",")[2].replace("<COMMA>", ",") + "\n")
                image_name_counter += 1

                # BA + second caption
                np.save(PROCESSED_IMAGES_PATH + str(image_name_counter) + ".npy", img_reversed_np)
                file.write(captions[idx].split(",")[3].replace("<COMMA>", ",") + "\n")
                image_name_counter += 1