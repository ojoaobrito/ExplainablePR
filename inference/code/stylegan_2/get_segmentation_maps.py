import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import torch
import numpy as np
import skimage.draw
from PIL import Image
from natsort import natsorted

np.set_printoptions(threshold = sys.maxsize)

# Import Mask RCNN
sys.path.append("../../../learning/train_mask_rcnn/mask_RCNN_master/")  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib

class PeriocularConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "periocular"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + periocular component

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.93

class InferenceConfig(PeriocularConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

########################################################################################################################
# CONTROL VARIABLES
########################################################################################################################
# general
GENUINE_OR_IMPOSTOR = "G"
NUM_IMAGES = 100000
SCRIPT_ID = 1

# mask-rcnn
IMAGE_SIZE = 256
MRCNN_IRIS_WEIGHTS_PATH = "../../../trained_models/mask_rcnn/periocular_iris/mask_rcnn_periocular_0030_iris.h5"
MRCNN_EYEBROW_WEIGHTS_PATH = "../../../trained_models/mask_rcnn/periocular_eyebrow/mask_rcnn_periocular_0030_eyebrow.h5"
MRCNN_SCLERA_WEIGHTS_PATH = "../../../trained_models/mask_rcnn/periocular_sclera/mask_rcnn_periocular_0030_sclera.h5"

def get_segmentation_map_all(model, image = None): # auxiliary function, computes the segmentation maps for the given image

    class_ids_reversed = {1: "iris", 2: "eyebrow", 3: "sclera"}

    image.save("seg_temp_" + str(SCRIPT_ID) + ".jpg")
    image = skimage.io.imread("seg_temp_" + str(SCRIPT_ID) + ".jpg")
    
    r = model.detect([image], verbose=1)[0]
    masks = {}
    for idx, i in enumerate(list(r["class_ids"])):
        mask = np.dstack((np.stack((((r["masks"][:, :, idx] + 0) * 255),) * 3, axis =- 1), np.full((512, 512), 255)))

        masks[class_ids_reversed[i]] = Image.fromarray(mask.astype(np.uint8)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

    os.remove("seg_temp_" + str(SCRIPT_ID) + ".jpg")
    
    if(len(masks.keys()) != len(class_ids_reversed.keys())): return(None)
    
    masks_final = {}
    for k, v in masks.items():
        mask_np = np.asarray(v).copy()
        mask_np[mask_np[:, :, 0] == 0] = np.asarray([0, 0, 0, 0])
        mask_np[mask_np[:, :, 0] == 255] = colors[k]
        masks_final[k] = mask_np
    
    return(masks_final)

def get_segmentation_map(model, class_to_detect, image = None): # auxiliary function, computes the segmentation maps for the given image

    class_ids_reversed = {1: class_to_detect}

    image.save("seg_temp_" + str(SCRIPT_ID) + ".jpg")
    image = skimage.io.imread("seg_temp_" + str(SCRIPT_ID) + ".jpg")
    
    r = model.detect([image], verbose = 0)[0]
    mask = None
    for idx, i in enumerate(list(r["class_ids"])):
        mask = np.dstack((np.stack((((r["masks"][:, :, idx] + 0) * 255),) * 3, axis =- 1), np.full((512, 512), 255)))

        if(class_ids_reversed[i] == class_to_detect): 
            mask = Image.fromarray(mask.astype(np.uint8)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
            break

    os.remove("seg_temp_" + str(SCRIPT_ID) + ".jpg")
    
    if(mask is None): return(None)
    
    '''mask_np = np.asarray(mask).copy()
    mask_np[mask_np[:, :, 0] == 0] = np.asarray([0, 0, 0, 0])
    mask_np[mask_np[:, :, 0] == 255] = colors[class_to_detect]'''
    
    return(np.asarray(mask))

if(__name__ == "__main__"):
    
    #################################################################################################################
    # INITIAL SETUP
    #################################################################################################################
    device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

    os.makedirs("synthetic_dataset_" + GENUINE_OR_IMPOSTOR + "/segmentation_maps_" + str(SCRIPT_ID), exist_ok = True)
    os.makedirs("synthetic_dataset_" + GENUINE_OR_IMPOSTOR + "/images_revised_" + str(SCRIPT_ID), exist_ok = True)
    os.makedirs("synthetic_dataset_" + GENUINE_OR_IMPOSTOR + "/images_excluded_" + str(SCRIPT_ID), exist_ok = True)
    
    colors = {
                "iris": np.asarray([255, 255, 255, 255]), 
                "eyebrow": np.asarray([250, 50, 83, 255]), 
                #"sclera": np.asarray([255, 204, 51, 255])
            }
    
    ####################################################################################################################
    # LOAD THE MASK-RCNN MODELS
    ####################################################################################################################
    config = InferenceConfig()

    model_iris = modellib.MaskRCNN(mode = "inference", config = config, model_dir = "../../trained_models/mask_rcnn")
    model_iris.load_weights(MRCNN_IRIS_WEIGHTS_PATH, by_name = True)

    model_eyebrow = modellib.MaskRCNN(mode = "inference", config = config, model_dir = "../../trained_models/mask_rcnn")
    model_eyebrow.load_weights(MRCNN_EYEBROW_WEIGHTS_PATH, by_name = True)

    model_sclera = modellib.MaskRCNN(mode = "inference", config = config, model_dir = "../../trained_models/mask_rcnn")
    model_sclera.load_weights(MRCNN_SCLERA_WEIGHTS_PATH, by_name = True)

    ######################################################################################################################################################################################################
    # GET THE SEGMENTATION MAPS FOR THE SYNTHETIC IMAGES
    ######################################################################################################################################################################################################
    images_directory = "synthetic_dataset_" + GENUINE_OR_IMPOSTOR + "/images_" + str(SCRIPT_ID)
    images = list(filter(lambda x : x[0] != ".", os.listdir(images_directory)))

    NUM_IMAGES = len(images)

    for idx, i in enumerate(images):

        print(str(idx + 1) + "/" + str(NUM_IMAGES))

        neighbour_np = np.load("synthetic_dataset_" + GENUINE_OR_IMPOSTOR + "/images_" + str(SCRIPT_ID) + "/" + i)
        first_image = neighbour_np[:, :, :3]
        second_image = neighbour_np[:, :, 3:]
        
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # obtain the segmentation maps for both images of this synthetic pair (if we can't get all the segmentation maps needed, then forget about this pair and try again)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        mask_path = "synthetic_dataset_" + GENUINE_OR_IMPOSTOR + "/segmentation_maps_" + str(SCRIPT_ID) + "/" + i.replace(".npy", "")

        models = {"iris": model_iris, "eyebrow": model_eyebrow}
        masks = {}
        for k, v in models.items():
            # get the mask for image A
            mask_A = get_segmentation_map(model = v, class_to_detect = k, image = Image.fromarray(first_image.astype(np.uint8)).resize((512, 512), Image.LANCZOS))

            if(mask_A is None): 
                #os.remove("synthetic_dataset_" + GENUINE_OR_IMPOSTOR + "/images_" + str(SCRIPT_ID) + "/" + i)
                os.rename("synthetic_dataset_" + GENUINE_OR_IMPOSTOR + "/images_" + str(SCRIPT_ID) + "/" + i, "synthetic_dataset_" + GENUINE_OR_IMPOSTOR + "/images_excluded_" + str(SCRIPT_ID) + "/" + i)
                break
            
            # get the mask for image B
            mask_B = get_segmentation_map(model = v, class_to_detect = k, image = Image.fromarray(second_image.astype(np.uint8)).resize((512, 512), Image.LANCZOS))

            if(mask_B is None): 
                #os.remove("synthetic_dataset_" + GENUINE_OR_IMPOSTOR + "/images_" + str(SCRIPT_ID) + "/" + i)
                os.rename("synthetic_dataset_" + GENUINE_OR_IMPOSTOR + "/images_" + str(SCRIPT_ID) + "/" + i, "synthetic_dataset_" + GENUINE_OR_IMPOSTOR + "/images_excluded_" + str(SCRIPT_ID) + "/" + i)
                break

            masks[k] = [mask_A, mask_B]
            
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # save the segmentation maps for this synthetic pair
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if(len(masks.keys()) == 2):
            print("SUCCESSFUL SEGMENTATION")

            os.rename("synthetic_dataset_" + GENUINE_OR_IMPOSTOR + "/images_" + str(SCRIPT_ID) + "/" + i, "synthetic_dataset_" + GENUINE_OR_IMPOSTOR + "/images_revised_" + str(SCRIPT_ID) + "/" + i)
            os.makedirs(mask_path)

            for k in ["A", "B"]:
                index = 0 if(k == "A") else 1

                eyebrow_mask = Image.fromarray(np.asarray(masks["eyebrow"][index]))
                iris_mask = Image.fromarray(np.asarray(masks["iris"][index]))

                # finally save the masks
                eyebrow_mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST).save(mask_path + "/eyebrow_" + k + ".png")
                iris_mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST).save(mask_path + "/iris_" + k + ".png")