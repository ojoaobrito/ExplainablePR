import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import numpy as np
from PIL import Image
from shutil import rmtree
from natsort import natsorted

if(__name__ == "__main__"):

    colors = {"sclera": np.asarray([255, 204, 51, 255]), "eyelid": np.asarray([255, 106, 77, 255]), "eyeglasses": np.asarray([89, 134, 179, 255]), 
            "to_ignore": np.asarray([38, 179, 83, 255]), "spot": np.asarray([184, 61, 244, 255]), "eyebrow": np.asarray([250, 50, 83, 255]),
            "skin": np.asarray([153, 102, 51, 200]), "iris": np.asarray([255, 255, 255, 255]), "nothing": np.asarray([0, 0, 0, 255])}

    if(os.path.exists("masks_applied")): rmtree("masks_applied")
    os.makedirs("masks_applied")

    masks = natsorted(list(filter(lambda x : x[0] != ".", os.listdir("test_masks"))))

    ###################################################################################
    # APPLY THE MASKS TO THE TEST IMAGES
    ###################################################################################
    for idx, i in enumerate(masks):

        print(str(idx + 1) + "/" + str(len(masks)))

        base_image = Image.open("test_masks/" + i + "/" + i + ".jpg").convert("RGBA")

        masks_aux = list(filter(lambda x : x[0] != ".", os.listdir("test_masks/" + i)))

        for j in masks_aux:

            if((".jpg" in j)): continue

            # load the mask
            mask = Image.open("test_masks/" + i + "/" + j).convert("RGBA")
            mask_np = np.asarray(mask).copy()

            # replace the white pixels with the actual mask color
            mask_np[mask_np[:, :, 0] == 255] = colors[j.replace(".png", "")]

            mask = Image.fromarray(mask_np.astype(np.uint8))
            mask = mask.resize(base_image.size, Image.BILINEAR)
            
            # make the black pixels transparent
            mask_np = np.asarray(mask).copy()
            mask_np[mask_np[:, :, 0] == 0] = np.asarray([0, 0, 0, 0])

            mask = Image.fromarray(mask_np.astype(np.uint8))
            
            # actually apply the mask
            base_image.paste(mask, (0, 0), mask)

        # save the test image with the masks applied to it
        base_image.save("masks_applied/" + i + ".png")