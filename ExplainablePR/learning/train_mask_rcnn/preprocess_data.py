import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import numpy as np
from random import choice
from natsort import natsorted
from PIL import Image, ImageOps, ImageEnhance

##################################################################
# CONTROL VARIABLES
##################################################################
IMAGE_SIZE = 512
RESIZE_IMAGES_AND_MASKS = True
APPLY_AUGMENTATION = False
CLASS_TO_DETECT = "iris" # either "iris", "eyebrow" or "sclera"
AUGMENTATION_STEPS_PER_IMAGE = 1
ONLY_INCLUDE_IRIS_EYEBROW_AND_SCLERA_MASKS = False
SHRINK_THEN_UPSCALE_IMAGES = True

def rotate_and_crop(img, angle): # auxiliary function, creates a rotated and cropped version of the input image

    # rotate and crop the image accordingly
    img = img.rotate(angle)
    width, height = img.size
    new_width = int(width - (width * (abs(angle) * 0.035)))
    new_height = int(height - (height * (abs(angle) * 0.035)))
    img = img.crop((int((width - new_width) / 2), int((height - new_height) / 2), new_width + int((width - new_width) / 2), new_height + int((height - new_height) / 2)))

    return(img)

if(__name__ == "__main__"):

    angles = [-1.25, -1, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1, 1.25]
    factors = [0.9, 0.95, 0.97, 1.03, 1.05, 1.1]

    colors = {"sclera": np.asarray([255, 204, 51, 255]), "eyelid": np.asarray([255, 106, 77, 255]), "eyeglasses": np.asarray([89, 134, 179, 255]), 
            "to_ignore": np.asarray([38, 179, 83, 255]), "spot": np.asarray([184, 61, 244, 255]), "eyebrow": np.asarray([250, 50, 83, 255]),
            "skin": np.asarray([153, 102, 51, 255]), "iris": np.asarray([255, 255, 255, 255]), "nothing": np.asarray([0, 0, 0, 255])}

    images = natsorted(list(filter(lambda x : x[0] != ".", os.listdir("images"))))
    images_path = "images"
    masks_path = "masks"
    
    ##############################################################################################
    # PREPROCESS THE TRAINING DATA
    ##############################################################################################
    for idx, i in enumerate(images):
        
        print(str(idx + 1) + "/" + str(len(images)))
        try:
            img = Image.open(images_path + "/" + i)
        except: 
            os.remove(images_path + "/" + i)
            continue

        # if required, resize everything
        if(RESIZE_IMAGES_AND_MASKS):
            if(SHRINK_THEN_UPSCALE_IMAGES):
                img = img.resize((256, 256), Image.LANCZOS)
            
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            img.save(images_path + "/" + i)

        try:
            mask = Image.open(masks_path + "/" + i.replace(".jpg", ".png"))
            prefix = ""
        except: 
            os.remove(images_path + "/" + i)
            continue

        if(ONLY_INCLUDE_IRIS_EYEBROW_AND_SCLERA_MASKS):
            mask_np = np.asarray(mask).copy()
            
            excluded_masks = ["eyelid", "eyeglasses", "to_ignore", "spot", "nothing", "skin"]
            for j in excluded_masks:
                mask_np[mask_np[:, :, 1] == colors[j][1]] = np.asarray([0, 0, 0, 255])

            mask = Image.fromarray(mask_np.astype(np.uint8))

        else:
            mask_np = np.asarray(mask).copy()
            
            for j in colors.keys():
                if(j != CLASS_TO_DETECT):
                    mask_np[mask_np[:, :, 1] == colors[j][1]] = np.asarray([0, 0, 0, 255])

            mask = Image.fromarray(mask_np.astype(np.uint8))  
            
        if(RESIZE_IMAGES_AND_MASKS):
            mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
            mask.save(masks_path + "/" + prefix + i.replace(".jpg", ".png"))
        
        # ----------------------------------------------------------------------------------------
        # if required, apply augmentation to the source image and corresponding mask
        # ----------------------------------------------------------------------------------------
        if(APPLY_AUGMENTATION):
            for j in range(AUGMENTATION_STEPS_PER_IMAGE):
                chosen_angle = choice(angles)
                chosen_factor = choice(factors)

                # apply augmentation to the source image
                if(choice([0, 1]) == 0):
                    img = ImageOps.mirror(img)
                
                img = rotate_and_crop(img, chosen_angle)

                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(chosen_factor)

                # apply augmentation to the corresponding mask
                mask = ImageOps.mirror(mask)
                mask = rotate_and_crop(mask, chosen_angle)

                # save everything
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
                mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)

                img.save(images_path + "/" + i.replace(".jpg", "+aug" + str(j) + ".jpg"))
                mask.save(masks_path + "/" + prefix + i.replace(".jpg", "+aug" + str(j) + ".png"))

    #########################################################################################
    # IF IT HAPPENS, FIX A SMALL PROBLEM WHERE SOME IMAGES/MASKS ARE AUGMENTED TOO MANY TIMES
    #########################################################################################
    images = natsorted(list(filter(lambda x : x[0] != ".", os.listdir(images_path))))

    for idx, i in enumerate(images):
        if("+aug" in i):
            if(len(i.split("+aug")) > 2): os.remove(images_path + "/" + i)

    masks = natsorted(list(filter(lambda x : x[0] != ".", os.listdir(masks_path))))

    for idx, i in enumerate(masks):
        if("+aug" in i):
            if(len(i.split("+aug")) > 2): os.remove(masks_path + "/" + i)
    
    ############################################################################################
    # PREPROCESS THE TEST DATA
    ############################################################################################
    if(RESIZE_IMAGES_AND_MASKS == True):

        test_images = natsorted(list(filter(lambda x : x[0] != ".", os.listdir("test_images"))))

        for idx, i in enumerate(test_images):
        
            print(str(idx + 1) + "/" + str(len(test_images)))

            img = Image.open("test_images/" + i)
            
            if(SHRINK_THEN_UPSCALE_IMAGES):
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            img.save("test_images/" + i)