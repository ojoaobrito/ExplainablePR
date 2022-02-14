import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import numpy as np
from PIL import Image
from random import sample
from skimage.metrics import structural_similarity as ssim
from dataset_augmentation import rotate_and_crop, add_noise, change_brightness, add_blur

################################################
# CONTROL VARIABLES
################################################
TARGET_UBI_PR = 30
TARGET_FFHQ = 6
SOURCE_DIR = "../../dataset/dataset_one_folder/"

def perform_augmentation(image_name, needed): # auxiliary function, brings everything together

    img = Image.open(SOURCE_DIR + image_name)

    # rotate and crop the image
    img = rotate_and_crop(img = img, image_name = None, new_image_name = None)
    
    # change the image's brightness
    img = change_brightness(img = img, image_name = None, new_image_name = None)

    # add blur to the image
    img.save("temp.jpg")
    add_blur(image_name = "temp.jpg", new_image_name = (SOURCE_DIR + image_name).replace(".jpg", "+aug+RO+BR+BL" + str(needed) + ".jpg"))
    
    os.remove("temp.jpg")

def find_equidistant_image(images): # auxiliary function, finds the image that is the most equidistant to the others (based on SSIM distances)

    ssim_scores = {}
    equidistant_image = (None, 0.0)

    for i in images:
        img_a = np.asarray(Image.open(SOURCE_DIR + i).convert("L"))
        aux = 0.0

        for j in images:
            img_b = np.asarray(Image.open(SOURCE_DIR + j).convert("L"))

            if(i == j): continue

            score = ssim(img_a, img_b, data_range = (img_b.max() - img_b.min()))

            ssim_scores.update({(i, j): score})

            aux += score

        # compute the mean SSIM score
        aux /= (len(images) - 1)

        # make this image the equidistant one, if its SSIM score is the best so far
        if(equidistant_image[1] < aux): equidistant_image = (i, aux)

    for i in list(ssim_scores.items()):

        # this score does not belong to the equidistant image we found earlier, so let's remove it
        if(equidistant_image[0] != i[0][0]): del ssim_scores[i[0]]
        
    return(ssim_scores, equidistant_image)

def get_images_to_keep(ssim_scores, images, equidistant_image, num_images): # auxiliary function, selects just "num_images" images (after being sorted by their SSIM distances to "equidistant_image")

    images_to_keep = sorted(images, key = lambda x : ssim_scores[(equidistant_image, x)] if(equidistant_image != x) else 1.1)[:num_images]
        
    return(images_to_keep)

if(__name__ == "__main__"):   

    images_per_id = {}
    directory = list(filter(lambda x : x[0] != ".", os.listdir(SOURCE_DIR)))

    # get every ID
    for i in directory:
        if(not ((i.split("_")[0]) in images_per_id)): images_per_id.update({(i.split("_")[0]): [i]})
        else: images_per_id.update({(i.split("_")[0]): images_per_id[(i.split("_")[0])] + [i]})

    #################################################################################################################################################
    # RUN THROUGH EVERY ID AND ENSURE FAIR PLAY
    #################################################################################################################################################
    for k, v in images_per_id.items():
        if(len(v[0].split("/")[-1].split("_")) <= 3): target = TARGET_FFHQ
        else: target = TARGET_UBI_PR

        if(len(v) == target): continue
        
        needed = target - len(v)

        # ----------------------------------------------
        # we have too little images, let's add more
        # ----------------------------------------------
        while(needed > 0): 
            
            images_per_id.update({k: sample(v, len(v))})

            for i in images_per_id[k]:
                perform_augmentation(i, needed)
                needed -= 1
                if(needed == 0): break

        # -------------------------------------------------------------------------------------------------------------------------------------------
        # we have too many images, lets remove some of them
        # -------------------------------------------------------------------------------------------------------------------------------------------
        if(needed < 0): 

            left_side_images = list(filter(lambda x : "_L_" in x, v))
            right_side_images = list(filter(lambda x : "_R_" in x, v))
            
            # find the most equidistant images (for both sides)
            left_ssim_scores, left_side_equidistant_image = find_equidistant_image(left_side_images)
            right_ssim_scores, right_side_equidistant_image = find_equidistant_image(right_side_images)

            # keep the images (from both sides) that are the most different (when compared to the equidistant one via SSIM scores)
            left_side_images_to_keep = get_images_to_keep(left_ssim_scores, left_side_images, left_side_equidistant_image[0], (target // 2) - 1 )
            right_side_images_to_keep = get_images_to_keep(right_ssim_scores, right_side_images, right_side_equidistant_image[0], (target // 2) - 1)
            
            #images_to_keep = sample(v, TARGET)
            images_to_keep = left_side_images_to_keep + right_side_images_to_keep + [left_side_equidistant_image[0], right_side_equidistant_image[0]]
            
            #images_to_augment = sample(images_to_keep, 10)

            # remove the unwanted images
            for i in v:
                if(not (i in images_to_keep)): os.remove(SOURCE_DIR + i)
                else: 
                    '''if(i in images_to_augment):
                        perform_augmentation(i, needed)
                        os.remove(SOURCE_DIR + i)
                        needed -= 1'''
                    continue

    if(os.path.exists(SOURCE_DIR + "temp.jpg")): os.remove(SOURCE_DIR + "temp.jpg")