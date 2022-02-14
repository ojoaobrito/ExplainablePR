import os, sys, csv

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

############################
# CONTROL VARIABLES
############################
GENERAL_INFO = False
IMAGES_PER_ATTRIBUTE = True
IDS_PER_IMAGE_AMOUNT = True

def general_info(just_total): # auxiliary function, prints, amongst others, the total number of images and plots the most common image sizes

    # --------------------------------------------------------------------------------------------
    # compute the total number of images
    # --------------------------------------------------------------------------------------------
    # create a common directory (and remove an existing one, if needed)

    # compute the total number of images
    images = list(filter(lambda x : x[0] != ".", os.listdir("../../dataset/dataset_one_folder/")))
    num_images = len(images)

    if(just_total): return(num_images)
    else: print("\nTOTAL NUMBER OF IMAGES: " + str(num_images) + "\n")

    # compute the total number of IDs
    ids = []
    for i in images:
        if(not ((i.split("_")[0]) in ids)): ids.append(i.split("_")[0])

    print("\nTOTAL NUMBER OF IDs: " + str(len(ids)) + "\n")

    # -------------------------------------------------------------------------------------------------------
    # plot a bar chart with the number of images per size
    # -------------------------------------------------------------------------------------------------------
    size_dict = {}

    for i in images:

        # retrieve the image's dimensions
        width, heigth = Image.open("../../dataset/dataset_one_folder/" + i).size
        
        # update the dictionary
        if((width, heigth) in size_dict): size_dict.update({(width, heigth): size_dict[(width, heigth)] + 1})
        else: size_dict.update({(width, heigth): 1})

    # prepare the data
    sizes = list(size_dict.keys())
    y_pos = np.arange(len(sizes))
    amounts = list(size_dict[i] for i in sizes)

    # plot the bar chart
    plt.barh(y_pos, amounts, align = "center", alpha = 0.5)
    plt.yticks(y_pos, sizes)
    plt.ylabel("Image size")
    plt.xlabel("Number of images")
    plt.title("Distribution of image sizes")
    plt.tight_layout()

    for i, v in enumerate(amounts):
        plt.text(v + 3, i - .05, str(v), color = "blue", fontweight = "bold", fontsize = 7)
    
    plt.show()

def images_per_attribute(show_plot, return_stats): # auxiliary function, plots a bar chart with the amount of images per attribute

    attributes_names = ["iris_color_0-blue", "iris_color_1-hazel", "iris_color_2-brown", "iris_color_3-dark-brown", 
                    "eyebrow_distribution_0-sparse", "eyebrow_distribution_1-average", "eyebrow_distribution_2-dense", "eyebrow_shape_0-angled", 
                    "eyebrow_shape_1-curved", "eyebrow_shape_2-straight", "skin_color_0-light", "skin_color_1-average", "skin_color_2-dark", 
                    "skin_texture_0-average", "skin_texture_1-middle-aged", "skin_texture_2-aged", "skin_spots_0-0", "skin_spots_1-1", 
                    "skin_spots_2-2+", "eyelid_shape_0-exposed", "eyelid_shape_1-covered"]

    images_per_attribute = {i: [] for i in attributes_names}

    # ------------------------------------------------------------------------------------------------------------------------
    # read and prepare the data
    # ------------------------------------------------------------------------------------------------------------------------
    with open("../../dataset/annotations.csv", "r") as file:
        content = list(csv.reader(file, delimiter = ","))[1:]

    for i in content:
        
        image_attributes = i[1:]

        # get the image's attribute names
        image_attributes_names = list(filter(lambda x : image_attributes[attributes_names.index(x)] == "1", attributes_names))

        # update the dictionary
        for j in image_attributes_names: images_per_attribute.update({j: images_per_attribute[j] + [i]})

    # prepare the data
    y_pos = np.arange(len(attributes_names))
    amounts = list(len(images_per_attribute[i]) for i in attributes_names)

    # -----------------------------------------------------------------------------------------
    # if required, plot the bar chart
    # -----------------------------------------------------------------------------------------
    if(show_plot):    
        bar = plt.barh(y_pos, amounts, align = "center", alpha = 0.5)
        plt.yticks(y_pos, attributes_names)
        plt.ylabel("Attribute")
        plt.xlabel("Number of images")
        plt.title("Distribution of images per attribute")
        plt.tight_layout()
        
        for i, v in enumerate(amounts):
            plt.text(v + 4, i - .15, str(v), color = "blue", fontweight = "bold", fontsize = 7)
        
        plt.show()

    if(return_stats): return(images_per_attribute)

def IDs_per_image_amount(): # auxiliary function, plots a bar chart with the amount of IDs for per amount of images

    ids = {}
    IDs_per_amount = {}

    # --------------------------------------------------------------------------------------------
    # get the amount of images per ID
    # --------------------------------------------------------------------------------------------
    for i in list(filter(lambda x : x[0] != ".", os.listdir("../../dataset/dataset_one_folder"))):
        
        # update the dictionary
        if((i.split("_")[0]) in ids): ids.update({i.split("_")[0]: ids[i.split("_")[0]] + 1})
        else: ids.update({i.split("_")[0]: 1})

    # ----------------------------------------------------------------------------
    # invert the above information (i.e. get the number of IDs per image amount)
    # ----------------------------------------------------------------------------
    for k in ids.values():
        
        # update the dictionary
        if(k in IDs_per_amount): IDs_per_amount.update({k: IDs_per_amount[k] + 1})
        else: IDs_per_amount.update({k: 1})

    # prepare the data
    image_amounts = sorted(list(IDs_per_amount.keys()))
    image_amounts.reverse()
    y_pos = np.arange(len(image_amounts))
    amounts = list(IDs_per_amount[i] for i in image_amounts)

    # ---------------------------------------------------------------------------------------
    # plot the bar chart
    # ---------------------------------------------------------------------------------------
    plt.barh(y_pos, amounts, align = "center", alpha = 0.5)
    plt.yticks(y_pos, image_amounts)
    plt.ylabel("Amount of images")
    plt.xlabel("Number of IDs per image amount")
    plt.title("Distribution of IDs per image amount")
    plt.tight_layout()

    for i, v in enumerate(amounts):
        plt.text(v + 0.5, i - .25, str(v), color = "blue", fontweight = "bold", fontsize = 7)

    plt.show()

if(__name__ == "__main__"):

    if(GENERAL_INFO): general_info(just_total = False)
    if(IMAGES_PER_ATTRIBUTE): images_per_attribute(show_plot = True, return_stats = False)
    if(IDS_PER_IMAGE_AMOUNT): IDs_per_image_amount()