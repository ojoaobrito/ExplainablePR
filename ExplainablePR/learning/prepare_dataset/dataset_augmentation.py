import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import cv2
import numpy as np
from random import choice
from skimage import io, filters
from PIL import Image, ImageEnhance

###############################
# CONTROL VARIABLES
###############################
SOURCE_DIR = "ffhq_annotations"

def rotate_and_crop(img, image_name, new_image_name): # auxiliary function, creates a rotated and cropped version of the input image
    
    rotation_angle = choice([-2.25, -2.0, -1.5, -1.0, 1.0, 1.5, 2.0, 2.25])
    
    if(img is None): img = Image.open(image_name)

    # rotate and crop the image accordingly
    img = img.rotate(rotation_angle)
    width, height = img.size
    new_width = int(width - (width * (abs(rotation_angle) * 0.020)))
    new_height = int(height - (height * (abs(rotation_angle) * 0.020)))
    img = img.crop((int((width - new_width) / 2),int((height - new_height) / 2),new_width + int((width - new_width) / 2),new_height + int((height - new_height) / 2)))

    # save the final image
    if(not (new_image_name is None)): img.save(new_image_name)
    else: return(img)

def change_brightness(img, image_name, new_image_name): # auxiliary function, creates a version of the input image with a brightness change

    factor = choice([0.80, 0.85, 0.87, 0.90, 0.93, 0.95, 0.97, 1.07, 1.1, 1.13, 1.15, 1.17, 1.20])

    if(img is None): img = Image.open(image_name)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(factor)

    # save the final image
    if(not (new_image_name is None)): img.save(new_image_name)
    else: return(img)

def add_blur(image_name, new_image_name): # auxiliary function, creates a blurred version of the input image

    sigma = choice([1.001, 1.002, 1.003])

    img = io.imread(fname = image_name)

    blurred = filters.gaussian(img, sigma = (sigma, sigma), truncate = 3.5, multichannel = True)

    # save the final image
    if(new_image_name is None): 
        io.imsave("temp.jpg", (blurred * 255).astype(np.uint8))
        temp = Image.open("temp.jpg")
        os.remove("temp.jpg")
        return(temp)

    else: io.imsave(new_image_name, (blurred * 255).astype(np.uint8))

def add_noise(image_name, new_image_name): # auxiliary function, creates a noisy version of the input image

    # gaussian noise parameters
    amount = choice([7.0, 7.25, 7.5, 7.75])
    mean = 0
    var = 0.1
    sigma = var ** 0.5 + amount

    img = cv2.imread(image_name)

    # add gaussian noise
    row, col, ch = img.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = img + gauss
    noisy = img

    # save the final image
    if(new_image_name is None): 
        cv2.imwrite("temp.jpg", noisy)
        temp = Image.open("temp.jpg")
        os.remove("temp.jpg")
        return(temp)

    else: cv2.imwrite(new_image_name, noisy)

def perform_augmentation(image_name): # main function, brings everything together

    img = Image.open(image_name)

    # rotate and crop the image
    img = rotate_and_crop(img = img, image_name = None, new_image_name = None)
    
    # change the image's brightness
    img = change_brightness(img = img, image_name = None, new_image_name = None)

    # add blur to the image
    img.save("temp.jpg")
    
    if(len(image_name.split("/")[-1].split("_")) == 2): add_blur(image_name = "temp.jpg", new_image_name = (image_name).replace(".jpg", "_2.jpg"))
    else:
        image_number = int(image_name.split("/")[-1].split("_")[2])
        image_name = "_".join(image_name.split("/")[-1].split("_")[:-1])
        add_blur(image_name = "temp.jpg", new_image_name = ("/".join(image_name.split("/")[:-1]) + image_name + "_" + str(image_number + 1) + ".jpg"))
    
    os.remove("temp.jpg")

def perform_recursive_augmentation(current):

    if(os.path.isdir(current)): # it's a directory, let's explore it
        for i in list(filter(lambda x : x[0] != ".", os.listdir(current))):
            perform_recursive_augmentation(current + "/" + i)

    else: # it's a file, let's do something with it
        perform_augmentation(current)

if(__name__ == "__main__"):

    perform_recursive_augmentation(SOURCE_DIR)