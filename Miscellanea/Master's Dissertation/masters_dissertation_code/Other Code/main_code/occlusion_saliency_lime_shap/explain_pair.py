import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import cv2
import time
import shap
import numpy as np
from PIL import Image
import tensorflow as tf
from vis.utils import utils
from lime import lime_image
import matplotlib.pylab as pl
from keras import activations
from natsort import natsorted
import matplotlib.pyplot as plt
from keras.models import load_model

import matplotlib.image as mpimage
from copy import deepcopy

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from matplotlib.colors import LinearSegmentedColormap
from keras.applications import inception_v3 as inc_net
from skimage.segmentation import slic, mark_boundaries
from vis.visualization import visualize_saliency

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

np.set_printoptions(threshold = sys.maxsize)

##############################################################################################################################
# CONTROL VARIABLES
##############################################################################################################################
# common variables
IMAGE_SIZE = 224
MASK_SIZE = (128, 128)
TECHNIQUE = "lime" if(len(sys.argv) != 2) else sys.argv[1] # either "occlusion_map", "saliency_map", "lime" or "shap"
TRANSPARENT_BACKGROUND = True
MODEL_PATH = "code/vgg_face_2_side_by_side_densenet121_98,04.h5"
REMOVE_TRASH = True
EXPLAIN_A_AND_B = True

# variables targeting the occlusion map explanations
NUMBER_OF_TILES = 3136 # either "16", "49", "64", "196", "256", "784", "1024" or "3136"
OCCLUSION_COLOUR = np.asarray([0, 0, 0])

# variables targeting the LIME explanations
TOP_SUPERPIXELS = 1000
NUM_SAMPLES_LIME = 20000

# variables targeting the SHAP explanations
NUM_SEGMENTS = 100
NUM_SAMPLES_SHAP = 10000

def check_for_errors(r_value): # auxiliary function, checks for errors when running the main scripts

    if(r_value != 0):
        print("ERROR FOUND, STOPPING NOW...")
        sys.exit()

def create_image_occlusions(img_path, number_of_tiles): # auxiliary function, returns a dictionary with several tiles that shall be occluded

    img = cv2.imread(img_path)
    slices = []
    occlusion_scores =  {}
    
    # ---------------------------------------------------------------------------------------------------------------------------------------------------
    # break the image into equal tiles and prepare a dictionary to save the CNN's score when each tile is occluded
    # ---------------------------------------------------------------------------------------------------------------------------------------------------
    for r in range(0, img.shape[0], int(img.shape[0] // np.math.sqrt(NUMBER_OF_TILES))):
        for c in range(0, img.shape[1], int(img.shape[1] // np.math.sqrt(NUMBER_OF_TILES))):
            slices.append(img[r:(r + int(img.shape[0] // np.math.sqrt(NUMBER_OF_TILES))), c:(c + int(img.shape[1] // np.math.sqrt(NUMBER_OF_TILES))), :])

    for i in range(len(slices)):

        line_range = (
            int(i // np.math.sqrt(number_of_tiles) * (IMAGE_SIZE // np.math.sqrt(number_of_tiles))), 
            int((i // np.math.sqrt(number_of_tiles) * (IMAGE_SIZE // np.math.sqrt(number_of_tiles))) + (IMAGE_SIZE // np.math.sqrt(number_of_tiles)))
            )

        column_range = (
            int((i % (np.math.sqrt(number_of_tiles))) * (IMAGE_SIZE // np.math.sqrt(number_of_tiles))),
            int((i % (np.math.sqrt(number_of_tiles))) * (IMAGE_SIZE // np.math.sqrt(number_of_tiles)) + (IMAGE_SIZE // np.math.sqrt(number_of_tiles)))
            )

        occlusion_scores[(line_range, column_range)] = 0.0

    return(occlusion_scores)

def transform_img_fn(path_list): # auxiliary function, prepares the given image for the LIME explainer method

    out = []
    
    for img_path in path_list:
        
        img = image.load_img(img_path, target_size = (IMAGE_SIZE, IMAGE_SIZE))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        x = inc_net.preprocess_input(x)
        out.append(x)
    
    return(np.vstack(out))

def fill_segmentation(values, segmentation): # auxiliary function, fills the given segmentation
    
    out = np.zeros(segmentation.shape)
    
    for i in range(len(values)):
        out[segmentation == i] = values[i]
    
    return(out)

def mask_image(zs, segmentation, image, background = None): # auxiliary function, applies a mask to the given image
    
    if(background is None):
        background = image.mean((0, 1))
    
    out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
    
    for i in range(zs.shape[0]):
        out[i, :, :, :] = image
        
        for j in range(zs.shape[1]):
            if zs[i, j] == 0:
                out[i][segmentation == j, :] = background
    
    return(out)

def f(z): # auxiliary function, returns the model's prediction

    return model.predict(preprocess_input(mask_image(z, segments_slic, pair_orig, 255)))

def extract_explanation(pair, mask, A_or_B): # auxiliary function, finalizes the proposed explanation

    pair[mask == 0] = [136, 136, 136]
    
    if(A_or_B == "A"): pair = pair[56:168, :224 // 2, :]
    else: pair = pair[56:168, 224 // 2:, :]

    # -------------------------------------------------------------------------------
    # detect the borders and highlight them in yellow
    # -------------------------------------------------------------------------------
    # prepare the images
    Image.fromarray((mask * 255).astype(np.uint8)).save("code/mask_temp.png")
    image = cv2.imread("code/mask_temp.png")
    mask = np.zeros(image.shape, dtype=np.uint8) * 255
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # find the contours
    cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts: cv2.drawContours(mask, [c], -1, (0, 243, 239), thickness = 1)

    cv2.imwrite("code/mask.png", mask)
    mask = np.asarray(Image.open("code/mask.png").convert("RGBA")).copy()

    # make the mask transparent
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if(np.array_equal(mask[i][j], [0, 0, 0, 255])): mask[i][j] = [0, 0, 0, 0]

    # -------------------------------------------
    # apply the mask over the original image
    # -------------------------------------------
    if(A_or_B == "A"):
        mask = mask[56:168, :224 // 2, :]
    else:
        mask = mask[56:168, 224 // 2:, :]

    mask = Image.fromarray(mask.astype(np.uint8))
    pair = Image.fromarray(pair.astype(np.uint8))
    pair.paste(mask, (0, 0), mask)

    return(pair)

def get_percentage_of_white_pixels(img_np): # auxiliary function, gets the percentage of white pixels in the given image (used for saliency map explanations)

    count = 0

    for i in range(img_np.shape[0]):
        for j in range(img_np.shape[1]):
            if(not np.array_equal(img_np[i][j], np.asarray([0, 0, 0, 255]))):
                count += 1

    return(count / (np.product(img_np.shape)))

def get_percentage_of_grey_pixels(img_np): # auxiliary function, gets the percentage of grey pixels in the given image (used for LIME explanations)

    count = 0

    for i in range(img_np.shape[0]):
        for j in range(img_np.shape[1]):
            if(np.array_equal(img_np[i][j], np.asarray([127, 127, 127, 255]))):
                count += 1

    return(count / (np.product(img_np.shape)))

def get_percentage_of_green_pixels(img_np, first_or_second_run, A_or_B, shap_values, segments_slic): # auxiliary function, gets the percentage of green pixels in the given image (used for SHAP explanations)

    count = 0

    for i in range(img_np.shape[0]):
        for j in range(img_np.shape[1]):
            if(first_or_second_run == 1):
                if(A_or_B == "A"):
                    segment_number = segments_slic[i + 102][j]
                else:
                    segment_number = segments_slic[i + 102][j + 108]
            else:
                if(A_or_B == "A"):
                    segment_number = segments_slic[i + 102][j + 108]
                else:
                    segment_number = segments_slic[i + 102][j]
            
            if(shap_values[segment_number] > 0): count += 1

    return(count / (np.product(img_np.shape)))

if(__name__ == "__main__"):

    print("\nEXPLAINING THE GIVEN PAIR...")
    t0 = time.time()

    ########################################################################################################
    # PREPARE THE PAIR
    ########################################################################################################
    directory = natsorted(list(filter(lambda x : x[0] != ".", os.listdir("images"))))
    
    if(not ("a_" in directory[0])):
        img = Image.open("images/" + directory[0]).resize((IMAGE_SIZE // 2, IMAGE_SIZE // 2), Image.LANCZOS)
        img.save("images/a_" + directory[0])
        os.remove("images/" + directory[0])

    if(not ("b_" in directory[1])):
        img = Image.open("images/" + directory[1]).resize((IMAGE_SIZE // 2, IMAGE_SIZE // 2), Image.LANCZOS)
        img.save("images/b_" + directory[1])
        os.remove("images/" + directory[1])

    ###################################################################################################################################################################################################################
    # EXPLAIN THE PAIR
    ###################################################################################################################################################################################################################
    occlusion_map_possible_explanations = {"A": [], "B": []}
    saliency_map_possible_explanations = {"A": [], "B": []}
    lime_possible_explanations = {"A": [], "B": []}
    shap_possible_explanations = {"A": [], "B": []}
    shap_values_list = []
    segments_slic_list = []

    for i in range(2):

        directory = natsorted(list(filter(lambda x : x[0] != ".", os.listdir("images"))))
        genuine_or_impostor = "G" if("G" in directory[0]) else "I"

        # load the test images
        if(EXPLAIN_A_AND_B and i == 1):

            # change the order of the images
            os.rename("images/" + directory[0], "images/" + directory[0].replace("a", "c"))
            os.rename("images/" + directory[1], "images/" + directory[1].replace("b", "a"))
            os.rename("images/" + directory[0].replace("a", "c"), "images/" + directory[0].replace("a", "c").replace("c", "b"))
            
            directory = natsorted(list(filter(lambda x : x[0] != ".", os.listdir("images"))))
            imageA = np.asarray(Image.open("images/" + directory[0]))
            imageB = np.asarray(Image.open("images/" + directory[1]))

        else:
            imageA = np.asarray(Image.open("images/" + directory[0]))
            imageB = np.asarray(Image.open("images/" + directory[1]))
        
        # build the pair
        pair = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))
        pair[(IMAGE_SIZE // 4):((IMAGE_SIZE // 4) + (IMAGE_SIZE // 2)), :, :] = np.column_stack((imageA, imageB))
        pair_orig = pair.astype(int)
        pair_shap = Image.fromarray(pair_orig.astype(np.uint8))

        # convert the pair to a PIL object and back to Numpy (it's weird but necessary)
        Image.fromarray(pair_orig.astype(np.uint8)).save("code/temp.jpg")
        pair = np.asarray(Image.open("code/temp.jpg")).copy()
        
        # ---------------------------------------------------------------------------------------------------------------------------------------------
        # load the CNN and get its prediction 
        # ---------------------------------------------------------------------------------------------------------------------------------------------
        model = load_model(MODEL_PATH)
        
        if(not ("I" in directory[0]) and not ("G" in directory[0])):
            
            prediction = np.squeeze(model.predict(pair.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))))
            genuine_or_impostor = ["I", "G"][np.argmax(prediction)]
            score = round(prediction[1], 3)

            os.rename("images/" + directory[0], "images/a_" + genuine_or_impostor + "_" + str(score).replace(".", ",") + directory[0].replace("a", ""))
            os.rename("images/" + directory[1], "images/b_" + genuine_or_impostor + "_" + str(score).replace(".", ",") + directory[1].replace("b", ""))

        print("\nCOMPUTING A DIFFERENCE MASK...")

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # explain the pair with an occlusion map
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if(genuine_or_impostor == "I" and TECHNIQUE == "occlusion_map"):

            # break the image into several tiles
            occlusion_scores = create_image_occlusions("code/temp.jpg", NUMBER_OF_TILES)

            # -----------------------------------------------------------------------
            # get the CNN's score when each tile is occluded
            # -----------------------------------------------------------------------
            pair = pair.reshape((IMAGE_SIZE, IMAGE_SIZE, 3)).copy()
            for k, v in occlusion_scores.items():
                
                # prepare the image by occluding one of the tiles with a solid colour
                pair_aux = pair.copy()
                pair_aux[k[0][0]:k[0][1], k[1][0]:k[1][1], :] = OCCLUSION_COLOUR
                pair_aux = pair_aux.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
                
                # obtain the model's prediction and invert it
                occlusion_scores[k] = model.predict(pair_aux)[0][0]
                
            # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # plot the scores and create the occlusion map
            # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            scores = np.asarray(list(occlusion_scores.values())).reshape((int(np.math.sqrt(NUMBER_OF_TILES)), int(np.math.sqrt(NUMBER_OF_TILES))))
            
            plt.imshow(scores, interpolation = "nearest", origin = "upper", cmap = "Greys")
            plt.savefig("code/occlusion_temp_" + str(i + 1) + ".png", dpi = 500)
            occlusion_map = np.asarray(Image.fromarray((np.asarray(Image.open("code/occlusion_temp_" + str(i + 1) + ".png"))[291:2133, 719:2561, :]).astype(np.uint8)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST))
            
            # --------------------------------------------------------------------------------------------------------------------------------------------------
            # save the possible explanations
            # --------------------------------------------------------------------------------------------------------------------------------------------------
            if(i == 0):
                occlusion_map_possible_explanations["A"].append(occlusion_map[(IMAGE_SIZE // 4):((IMAGE_SIZE // 4) + (IMAGE_SIZE // 2)), :(IMAGE_SIZE // 2), :])
                occlusion_map_possible_explanations["B"].append(occlusion_map[(IMAGE_SIZE // 4):((IMAGE_SIZE // 4) + (IMAGE_SIZE // 2)), (IMAGE_SIZE // 2):, :])

            else:
                occlusion_map_possible_explanations["A"].append(occlusion_map[(IMAGE_SIZE // 4):((IMAGE_SIZE // 4) + (IMAGE_SIZE // 2)), (IMAGE_SIZE // 2):, :])
                occlusion_map_possible_explanations["B"].append(occlusion_map[(IMAGE_SIZE // 4):((IMAGE_SIZE // 4) + (IMAGE_SIZE // 2)), :(IMAGE_SIZE // 2), :])

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------
        # explain the pair with a saliency map
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------
        elif(genuine_or_impostor == "I" and TECHNIQUE == "saliency_map"):

            if(i ==  0):
                model.layers[-1].activation = activations.linear
                model = utils.apply_modifications(model)

            # -------------------------------------------------------------------------------------------------------------------------------------------------------
            # create the saliency map
            # -------------------------------------------------------------------------------------------------------------------------------------------------------
            saliency_map = visualize_saliency(model, -1, filter_indices = 0, seed_input = pair.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3)), backprop_modifier = "guided")
            saliency_map = (saliency_map.reshape((IMAGE_SIZE, IMAGE_SIZE)) * 255).astype(int)
            saliency_map = np.asarray(Image.fromarray(saliency_map.astype(np.uint8)).convert("RGB"))

            # ------------------------------------------------------------------------------------------------------------------------------------------------
            # save the possible explanations
            # ------------------------------------------------------------------------------------------------------------------------------------------------
            if(i == 0):
                saliency_map_possible_explanations["A"].append(saliency_map[(IMAGE_SIZE // 4):((IMAGE_SIZE // 4) + (IMAGE_SIZE // 2)), :(IMAGE_SIZE // 2), :])
                saliency_map_possible_explanations["B"].append(saliency_map[(IMAGE_SIZE // 4):((IMAGE_SIZE // 4) + (IMAGE_SIZE // 2)), (IMAGE_SIZE // 2):, :])

            else:
                saliency_map_possible_explanations["A"].append(saliency_map[(IMAGE_SIZE // 4):((IMAGE_SIZE // 4) + (IMAGE_SIZE // 2)), (IMAGE_SIZE // 2):, :])
                saliency_map_possible_explanations["B"].append(saliency_map[(IMAGE_SIZE // 4):((IMAGE_SIZE // 4) + (IMAGE_SIZE // 2)), :(IMAGE_SIZE // 2), :])

        # ------------------------------------------------------------------------------------------------------------------------------------------------
        # explain the pair with LIME
        # ------------------------------------------------------------------------------------------------------------------------------------------------
        elif(genuine_or_impostor == "I" and TECHNIQUE == "lime"):

            pair_aux = transform_img_fn(["code/temp.jpg"])[0].astype("double")

            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(pair_aux, model.predict, top_labels = 1, hide_color = 0, num_samples = NUM_SAMPLES_LIME)

            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only = True, num_features = TOP_SUPERPIXELS, hide_rest = True)

            plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
            plt.savefig("code/lime_temp_" + str(i + 1) + ".png")
            
            # ------------------------------------------------------------------------------------------------------------------------------
            # save the possible explanations
            # ------------------------------------------------------------------------------------------------------------------------------
            if(i == 0):
                lime_possible_explanations["A"].append(np.asarray(Image.open("code/lime_temp_" + str(i + 1) + ".png"))[148:336, 144:330, :])
                lime_possible_explanations["B"].append(np.asarray(Image.open("code/lime_temp_" + str(i + 1) + ".png"))[148:336, 329:513, :])

            else:
                lime_possible_explanations["A"].append(np.asarray(Image.open("code/lime_temp_" + str(i + 1) + ".png"))[148:336, 329:513, :])
                lime_possible_explanations["B"].append(np.asarray(Image.open("code/lime_temp_" + str(i + 1) + ".png"))[148:336, 144:330, :])

        # ----------------------------------------------------------------------------------------------------------------------------------
        # explain the pair with SHAP
        # ----------------------------------------------------------------------------------------------------------------------------------
        elif(genuine_or_impostor == "I" and TECHNIQUE == "shap"):
            
            # ------------------------------------------------------------------------------------------
            # run the KernelSHAP method to get an explanation
            # ------------------------------------------------------------------------------------------
            # create random super-pixels
            segments_slic = slic(pair_shap, n_segments = NUM_SEGMENTS, compactness = 1, sigma = 3)
            segments_slic_list.append(segments_slic)
            
            # get the KernelSHAP explanation
            explainer = shap.KernelExplainer(f, np.zeros((1, NUM_SEGMENTS)))
            shap_values = explainer.shap_values(np.ones((1, NUM_SEGMENTS)), nsamples = NUM_SAMPLES_SHAP)
            shap_values_list.append(shap_values[0][0])

            # get the top predictions from the model
            preds = model.predict(preprocess_input(np.expand_dims(pair_orig.copy(), axis = 0)))[0]

            # make a color map
            colors = []
            for l in np.linspace(1, 0, 100):
                colors.append((24 / 255, 196 / 255, 93 / 255, l))
            for l in np.linspace(0, 1, 100):
                colors.append((245 / 255, 39 / 255, 87 / 255, l))
            cm = LinearSegmentedColormap.from_list("shap", colors)

            # --------------------------------------------------------------------------------------------------------------
            # plot our explanations
            # --------------------------------------------------------------------------------------------------------------
            fig, axes = pl.subplots(nrows = 1, ncols = 2, figsize = (12, 4))
            axes[0].imshow(pair_shap)
            axes[0].axis("off")
            max_val = np.max([np.max(np.abs(shap_values[i][:, :-1])) for i in range(len(shap_values))])

            for j in range(1):
                m = fill_segmentation(shap_values[0][0], segments_slic)
                axes[j + 1].set_title("I")
                axes[j + 1].imshow(pair_shap.convert("LA"), alpha = 0.6)
                im = axes[j + 1].imshow(m, cmap = cm, vmin = -max_val, vmax = max_val)
                axes[j + 1].axis("off")

            cb = fig.colorbar(im, ax = axes.ravel().tolist(), label = "SHAP value", orientation = "horizontal", aspect = 60)
            cb.outline.set_visible(False)
            pl.savefig("code/shap_temp_" + str(i + 1) + ".png")

            # ------------------------------------------------------------------------------------------------------------------------------
            # save the possible explanations
            # ------------------------------------------------------------------------------------------------------------------------------
            if(i == 0):
                shap_possible_explanations["A"].append(np.asarray(Image.open("code/shap_temp_" + str(i + 1) + ".png"))[102:210, 761:869, :])
                shap_possible_explanations["B"].append(np.asarray(Image.open("code/shap_temp_" + str(i + 1) + ".png"))[102:210, 869:977, :])

            else:
                shap_possible_explanations["A"].append(np.asarray(Image.open("code/shap_temp_" + str(i + 1) + ".png"))[102:210, 869:977, :])
                shap_possible_explanations["B"].append(np.asarray(Image.open("code/shap_temp_" + str(i + 1) + ".png"))[102:210, 761:869, :])

        if(not EXPLAIN_A_AND_B): break

    print("\nDONE COMPUTING A DIFFERENCE MASK!")

    # if they were inverted, put the images in the original order
    if(genuine_or_impostor == "I" and EXPLAIN_A_AND_B):
        directory = natsorted(list(filter(lambda x : x[0] != ".", os.listdir("images"))))
        
        os.rename("images/" + directory[0], "images/" + directory[0].replace("a", "c"))
        os.rename("images/" + directory[1], "images/" + directory[1].replace("b", "a"))
        os.rename("images/" + directory[0].replace("a", "c"), "images/" + directory[0].replace("a", "c").replace("c", "b"))
    
        if(TECHNIQUE == "occlusion_map"):
            #---------------------------------------------------------------------------------------------------
            # get the best explanation for image A
            # --------------------------------------------------------------------------------------------------
            possible_explanation_1 = occlusion_map_possible_explanations["A"][0]
            possible_explanation_1_num_white_pixels = get_percentage_of_white_pixels(possible_explanation_1)

            if(EXPLAIN_A_AND_B):
                possible_explanation_2 = occlusion_map_possible_explanations["A"][1]
                possible_explanation_2_num_white_pixels = get_percentage_of_white_pixels(possible_explanation_2)

                if(possible_explanation_1_num_white_pixels >= possible_explanation_2_num_white_pixels):
                    np.save("code/difference_mask_1.npy", possible_explanation_1)

                else:
                    np.save("code/difference_mask_1.npy", possible_explanation_2)
            
            else:
                np.save("code/difference_mask_1.npy", possible_explanation_1)

            #---------------------------------------------------------------------------------------------------
            # get the best explanation for image B
            # --------------------------------------------------------------------------------------------------
            possible_explanation_1 = occlusion_map_possible_explanations["B"][0]
            possible_explanation_1_num_white_pixels = get_percentage_of_white_pixels(possible_explanation_1)

            if(EXPLAIN_A_AND_B):
                possible_explanation_2 = occlusion_map_possible_explanations["B"][1]
                possible_explanation_2_num_white_pixels = get_percentage_of_white_pixels(possible_explanation_2)

                if(possible_explanation_1_num_white_pixels >= possible_explanation_2_num_white_pixels):
                    np.save("code/difference_mask_2.npy", possible_explanation_1)

                else:
                    np.save("code/difference_mask_2.npy", possible_explanation_2)
            
            else:
                np.save("code/difference_mask_2.npy", possible_explanation_1)

        elif(TECHNIQUE == "saliency_map"):
            #---------------------------------------------------------------------------------------------------
            # get the best explanation for image A
            # --------------------------------------------------------------------------------------------------
            possible_explanation_1 = saliency_map_possible_explanations["A"][0]
            possible_explanation_1_num_white_pixels = get_percentage_of_white_pixels(possible_explanation_1)

            if(EXPLAIN_A_AND_B):
                possible_explanation_2 = saliency_map_possible_explanations["A"][1]
                possible_explanation_2_num_white_pixels = get_percentage_of_white_pixels(possible_explanation_2)

                if(possible_explanation_1_num_white_pixels >= possible_explanation_2_num_white_pixels):
                    np.save("code/difference_mask_1.npy", possible_explanation_1)

                else:
                    np.save("code/difference_mask_1.npy", possible_explanation_2)
            
            else:
                np.save("code/difference_mask_1.npy", possible_explanation_1)

            #---------------------------------------------------------------------------------------------------
            # get the best explanation for image B
            # --------------------------------------------------------------------------------------------------
            possible_explanation_1 = saliency_map_possible_explanations["B"][0]
            possible_explanation_1_num_white_pixels = get_percentage_of_white_pixels(possible_explanation_1)

            if(EXPLAIN_A_AND_B):
                possible_explanation_2 = saliency_map_possible_explanations["B"][1]
                possible_explanation_2_num_white_pixels = get_percentage_of_white_pixels(possible_explanation_2)

                if(possible_explanation_1_num_white_pixels >= possible_explanation_2_num_white_pixels):
                    np.save("code/difference_mask_2.npy", possible_explanation_1)

                else:
                    np.save("code/difference_mask_2.npy", possible_explanation_2)
            
            else:
                np.save("code/difference_mask_2.npy", possible_explanation_1)
        
        elif(TECHNIQUE == "lime"):
            #-------------------------------------------------------------------------------------------------
            # get the best explanation for image A
            # ------------------------------------------------------------------------------------------------
            possible_explanation_1 = lime_possible_explanations["A"][0]
            possible_explanation_1_num_grey_pixels = get_percentage_of_grey_pixels(possible_explanation_1)

            if(EXPLAIN_A_AND_B):
                possible_explanation_2 = lime_possible_explanations["A"][1]
                possible_explanation_2_num_grey_pixels = get_percentage_of_grey_pixels(possible_explanation_2)

                if(possible_explanation_1_num_grey_pixels <= possible_explanation_2_num_grey_pixels):
                    np.save("code/difference_mask_1.npy", possible_explanation_1)

                else:
                    np.save("code/difference_mask_1.npy", possible_explanation_2)
            
            else:
                np.save("code/difference_mask_1.npy", possible_explanation_1)

            #-------------------------------------------------------------------------------------------------
            # get the best explanation for image B
            # ------------------------------------------------------------------------------------------------
            possible_explanation_1 = lime_possible_explanations["B"][0]
            possible_explanation_1_num_grey_pixels = get_percentage_of_grey_pixels(possible_explanation_1)

            if(EXPLAIN_A_AND_B):
                possible_explanation_2 = lime_possible_explanations["B"][1]
                possible_explanation_2_num_grey_pixels = get_percentage_of_grey_pixels(possible_explanation_2)

                if(possible_explanation_1_num_grey_pixels <= possible_explanation_2_num_grey_pixels):
                    np.save("code/difference_mask_2.npy", possible_explanation_1)

                else:
                    np.save("code/difference_mask_2.npy", possible_explanation_2)
            
            else:
                np.save("code/difference_mask_2.npy", possible_explanation_1)

        elif(TECHNIQUE == "shap"):
            #-------------------------------------------------------------------------------------------------------------------------------------------------------
            # get the best explanation for image A
            # ------------------------------------------------------------------------------------------------------------------------------------------------------
            possible_explanation_1 = shap_possible_explanations["A"][0]
            possible_explanation_1_num_green_pixels = get_percentage_of_green_pixels(possible_explanation_1, 1, "A", shap_values_list[0], segments_slic_list[0])

            if(EXPLAIN_A_AND_B):
                possible_explanation_2 = shap_possible_explanations["A"][1]
                possible_explanation_2_num_green_pixels = get_percentage_of_green_pixels(possible_explanation_2, 2, "A", shap_values_list[1], segments_slic_list[1])

                if(possible_explanation_1_num_green_pixels >= possible_explanation_2_num_green_pixels):
                    np.save("code/difference_mask_1.npy", possible_explanation_1)
                    #np.save("shap_values.npy", np.asarray(shap_values_list[0]))
                    #np.save("segments_slic.npy", np.asarray(segments_slic_list[0]))

                else:
                    np.save("code/difference_mask_1.npy", possible_explanation_2)
                    #np.save("shap_values.npy", np.asarray(shap_values_list[1]))
                    #np.save("segments_slic.npy", np.asarray(segments_slic_list[1]))
            
            else:
                np.save("code/difference_mask_1.npy", possible_explanation_1)

            #-------------------------------------------------------------------------------------------------------------------------------------------------------
            # get the best explanation for image B
            # ------------------------------------------------------------------------------------------------------------------------------------------------------
            possible_explanation_1 = shap_possible_explanations["B"][0]
            possible_explanation_1_num_green_pixels = get_percentage_of_green_pixels(possible_explanation_1, 1, "B", shap_values_list[0], segments_slic_list[0])

            if(EXPLAIN_A_AND_B):
                possible_explanation_2 = shap_possible_explanations["B"][1]
                possible_explanation_2_num_green_pixels = get_percentage_of_green_pixels(possible_explanation_2, 2, "B", shap_values_list[1], segments_slic_list[1])

                if(possible_explanation_1_num_green_pixels >= possible_explanation_2_num_green_pixels):
                    np.save("code/difference_mask_2.npy", possible_explanation_1)
                    #np.save("shap_values.npy", np.asarray(shap_values_list[0]))
                    #np.save("segments_slic.npy", np.asarray(segments_slic_list[0]))

                else:
                    np.save("code/difference_mask_2.npy", possible_explanation_2)
                    #np.save("shap_values.npy", np.asarray(shap_values_list[1]))
                    #np.save("segments_slic.npy", np.asarray(segments_slic_list[1]))
            
            else:
                np.save("code/difference_mask_2.npy", possible_explanation_1)
    
    os.chdir("./code")

    # ----------------------------------------------------------------------
    # assemble the final explanation
    # ----------------------------------------------------------------------
    print("\nASSEMBLING THE FINAL EXPLANATION...")
    args = [str(MASK_SIZE[0]), str(TRANSPARENT_BACKGROUND)]
    r_value = os.system("python3 assemble_explanation.py " + " ".join(args))
    check_for_errors(r_value)
    print("DONE ASSEMBLING THE FINAL EXPLANATION!")

    os.chdir("..")

    ###################################################################################################
    # REMOVE SOME TRASH
    ###################################################################################################
    if(REMOVE_TRASH):
        if(os.path.exists("code/temp.jpg")): os.remove("code/temp.jpg")
        
        if(genuine_or_impostor == "I"):
            if(os.path.exists("code/difference_mask_1.npy")): os.remove("code/difference_mask_1.npy")
            if(os.path.exists("code/difference_mask_2.npy")): os.remove("code/difference_mask_2.npy")

            if(TECHNIQUE == "occlusion_map"):
                if(os.path.exists("code/occlusion_temp_1.png")): os.remove("code/occlusion_temp_1.png")
                if(os.path.exists("code/occlusion_temp_2.png")): os.remove("code/occlusion_temp_2.png")

            elif(TECHNIQUE == "lime"):
                if(os.path.exists("code/mask_temp.png")): os.remove("code/mask_temp.png")
                if(os.path.exists("code/mask.png")): os.remove("code/mask.png")
                if(os.path.exists("code/lime_temp_1.png")): os.remove("code/lime_temp_1.png")
                if(os.path.exists("code/lime_temp_2.png")): os.remove("code/lime_temp_2.png")

            elif(TECHNIQUE == "shap"):
                if(os.path.exists("code/shap_temp_1.png")): os.remove("code/shap_temp_1.png")
                if(os.path.exists("code/shap_temp_2.png")): os.remove("code/shap_temp_2.png")

    print("\nDONE EXPLAINING THE GIVEN PAIR!\n")
    elapsed_time = time.time() - t0
    print("[INFO] ELAPSED TIME: %.2fs\n" % (elapsed_time))

    with open("times_" + TECHNIQUE + ".txt", "a") as file:
        file.write(str(elapsed_time) + "\n")