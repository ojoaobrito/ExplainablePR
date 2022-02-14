import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import torch
import numpy as np
import skimage.draw
from PIL import Image
from shutil import rmtree
from natsort import natsorted

np.set_printoptions(threshold = sys.maxsize)

# Import Mask RCNN
sys.path.append("../../learning/train_mask_rcnn/mask_RCNN_master/")  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

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
    STEPS_PER_EPOCH = 120

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.92

class InferenceConfig(PeriocularConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

###############################################################################################################################
# CONTROL VARIABLES
###############################################################################################################################
# the values are the default ones (NOTE: if the values are to be changed, do so in this "if" branch)
if(len(sys.argv) == 1):
    # general
    IMAGE_SIZE = 128

    # mask-rcnn
    MRCNN_IRIS_WEIGHTS_PATH = "../../trained_models/mask_rcnn/periocular_iris/mask_rcnn_periocular_0030_iris.h5"
    MRCNN_EYEBROW_WEIGHTS_PATH = "../../trained_models/mask_rcnn/periocular_eyebrow/mask_rcnn_periocular_0030_eyebrow.h5"
    MRCNN_SCLERA_WEIGHTS_PATH = "../../trained_models/mask_rcnn/periocular_sclera/mask_rcnn_periocular_0030_sclera.h5"

# the values are coming from the "run_training.py" master script
else:
    # general
    IMAGE_SIZE = int(sys.argv[1])

    # mask-rcnn
    MRCNN_IRIS_WEIGHTS_PATH = sys.argv[2]
    MRCNN_EYEBROW_WEIGHTS_PATH = sys.argv[3]
    MRCNN_SCLERA_WEIGHTS_PATH = sys.argv[4]

def get_segmentation_map(model, class_to_detect, image = None): # auxiliary function, computes the segmentation maps for the given image

    class_ids_reversed = {1: class_to_detect}

    image.save("seg_temp.jpg")
    image = skimage.io.imread("seg_temp.jpg")
    
    r = model.detect([image], verbose=1)[0]
    mask = None
    for idx, i in enumerate(list(r["class_ids"])):
        mask = np.dstack((np.stack((((r["masks"][:, :, idx] + 0) * 255),) * 3, axis =- 1), np.full((512, 512), 255)))

        if(class_ids_reversed[i] == class_to_detect): 
            mask = Image.fromarray(mask.astype(np.uint8)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
            break

    os.remove("seg_temp.jpg")

    if(mask is None): return(None)
    
    mask_np = np.asarray(mask).copy()
    
    return(mask_np)

if(__name__ == "__main__"):
    
    ####################################################################################
    # INITIAL SETUP
    ####################################################################################
    device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

    if(os.path.exists("test_pair_masks")): rmtree("test_pair_masks")
    os.makedirs("test_pair_masks")

    directory = natsorted(list(filter(lambda x : x[0] != ".", os.listdir("../images"))))
    image_A = np.asarray(Image.open("../images/" + directory[0]))
    image_B = np.asarray(Image.open("../images/" + directory[1]))

    #####################################################################################################################
    # LOAD THE MASK-RCNN MODELS
    #####################################################################################################################
    config = InferenceConfig()

    model_iris = modellib.MaskRCNN(mode = "inference", config = config, model_dir = "../../trained_models/mask_rcnn/")
    model_iris.load_weights(MRCNN_IRIS_WEIGHTS_PATH, by_name = True)

    model_eyebrow = modellib.MaskRCNN(mode = "inference", config = config, model_dir = "../../trained_models/mask_rcnn/")
    model_eyebrow.load_weights(MRCNN_EYEBROW_WEIGHTS_PATH, by_name = True)

    model_sclera = modellib.MaskRCNN(mode = "inference", config = config, model_dir = "../../trained_models/mask_rcnn/")
    model_sclera.load_weights(MRCNN_SCLERA_WEIGHTS_PATH, by_name = True)
    
    ##############################################################################################################################################################
    # GET THE SEGMENTATION MAPS FOR THE TEST PAIR
    ##############################################################################################################################################################
    models = {"iris": model_iris, "eyebrow": model_eyebrow, "sclera": model_sclera}
    masks = {}
    for k, v in models.items():
        # get the mask for image A
        mask_A = get_segmentation_map(v, k, image = Image.fromarray(image_A.astype(np.uint8)).resize((256, 256), Image.LANCZOS).resize((512, 512), Image.LANCZOS))
        if(mask_A is not None): Image.fromarray(mask_A.astype(np.uint8)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS).save("test_pair_masks/" + k + "_A.png")
        
        # get the mask for image B
        mask_B = get_segmentation_map(v, k, image = Image.fromarray(image_B.astype(np.uint8)).resize((256, 256), Image.LANCZOS).resize((512, 512), Image.LANCZOS))
        if(mask_B is not None): Image.fromarray(mask_B.astype(np.uint8)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS).save("test_pair_masks/" + k + "_B.png")