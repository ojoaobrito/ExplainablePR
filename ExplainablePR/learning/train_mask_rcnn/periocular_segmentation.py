"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from PIL import Image
import tensorflow as tf
from natsort import natsorted
from shutil import copyfile, rmtree

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

np.set_printoptions(threshold = sys.maxsize)

# Root directory of the project
ROOT_DIR = os.path.abspath("./mask_RCNN_master/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

COLORS = {"sclera": np.asarray([255, 204, 51, 255]), "eyelid": np.asarray([255, 106, 77, 255]), "eyeglasses": np.asarray([89, 134, 179, 255]), 
            "spot": np.asarray([184, 61, 244, 255]), "eyebrow": np.asarray([250, 50, 83, 255]), "skin": np.asarray([153, 102, 51, 255]), 
            "iris": np.asarray([255, 255, 255, 255])}

#CLASS_IDS = {"skin": 1, "eyebrow": 2, "eyeglasses": 3, "eyelid": 4, "iris": 5, "sclera": 6, "spot": 7}
#CLASS_IDS_REVERSED = {1: "skin", 2: "eyebrow", 3: "eyeglasses", 4: "eyelid", 5: "iris", 6: "sclera", 7: "spot"}

'''CLASS_IDS = {"iris": 1, "eyebrow": 2, "sclera": 3}
CLASS_IDS_REVERSED = {1: "iris", 2: "eyebrow", 3: "sclera"}'''

CLASS_TO_DETECT = "eyebrow"
CLASS_IDS = {CLASS_TO_DETECT: 1}
CLASS_IDS_REVERSED = {1: CLASS_TO_DETECT}

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "../../trained_models/mask_rcnn")
SOURCE_DIR_TRAINING_IMAGES = "images"
SOURCE_DIR_TRAINING_MASKS = "masks"

############################################################
#  Configurations
############################################################


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
    NUM_CLASSES = 1 + 1  # Background + skin + eyebrow + eyeglasses + eyelid + iris + sclera + spots

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.93


############################################################
#  Dataset
############################################################

class PeriocularDataset(utils.Dataset):

    def load_periocular(self, subset):
        """Load a subset of the Periocular dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes
        '''self.add_class("periocular", 1, "skin")
        self.add_class("periocular", 2, "eyebrow")
        self.add_class("periocular", 3, "eyeglasses")
        self.add_class("periocular", 4, "eyelid")
        self.add_class("periocular", 5, "iris")
        self.add_class("periocular", 6, "sclera")
        self.add_class("periocular", 7, "spots")'''

        '''self.add_class("periocular", 1, "iris")
        self.add_class("periocular", 2, "eyebrow")
        self.add_class("periocular", 3, "sclera")'''

        self.add_class("periocular", 1, CLASS_TO_DETECT)

        # Train or validation dataset?
        assert subset in ["train", "test"]
        dataset = list(filter(lambda x : x[0] != ".", os.listdir(SOURCE_DIR_TRAINING_IMAGES)))

        # Add images
        for i in dataset:
            
            img = Image.open(SOURCE_DIR_TRAINING_IMAGES + "/" + i)
            width, height = img.size

            self.add_image(
                "periocular",
                image_id = i,  # use file name as a unique image id
                path = SOURCE_DIR_TRAINING_IMAGES + "/" + i,
                width = width, 
                height = height
                )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "periocular":
            return super(self.__class__, self).load_mask(image_id)
        info = self.image_info[image_id]
        
        # ------------------------------------------------------------------------------------------------------------------
        # load the masks
        # ------------------------------------------------------------------------------------------------------------------
        '''try:
            mask_aux_np = np.asarray(Image.open(SOURCE_DIR_TRAINING_MASKS + "/j_" + info["id"].replace(".jpg", ".png")))
        except:
            try:
                mask_aux_np = np.asarray(Image.open(SOURCE_DIR_TRAINING_MASKS + "/m_" + info["id"].replace(".jpg", ".png")))
            except:
                mask_aux_np = np.asarray(Image.open(SOURCE_DIR_TRAINING_MASKS + "/n_" + info["id"].replace(".jpg", ".png")))'''

        mask_aux_np = np.asarray(Image.open(SOURCE_DIR_TRAINING_MASKS + "/" + info["id"].replace(".jpg", ".png")))

        mask_np = None
        mask_class_ids = []

        for k, v in COLORS.items():
            if(k != CLASS_TO_DETECT): continue
            class_mask = np.zeros((info["height"], info["width"]), dtype = np.uint8)
            for i in range(mask_aux_np.shape[0]):
                for j in range(mask_aux_np.shape[1]):
                    if(np.array_equal(mask_aux_np[i][j], v)): # this pixel belongs to this class
                        class_mask[i][j] = 1

            if(np.array_equal(class_mask, np.zeros((info["height"], info["width"]), dtype = np.uint8))): continue

            # update the mask
            if(mask_np is None): 
                mask_np = class_mask.copy()
                mask_class_ids.append(CLASS_IDS[k])

            else: 
                mask_np = np.dstack((mask_np, class_mask))
                mask_class_ids.append(CLASS_IDS[k])
        
        #return mask_np.astype(np.bool), np.asarray(mask_class_ids, dtype = np.int32)
        return np.reshape(mask_np, (info["height"], info["width"], 1)).astype(np.bool), np.ones([1], dtype = np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "periocular":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = PeriocularDataset()
    dataset_train.load_periocular("train")
    dataset_train.prepare()

    # Validation dataset
    '''dataset_val = PeriocularDataset()
    dataset_val.load_periocular("val")
    dataset_val.prepare()'''

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, None,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims = True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def test(model, image_path = None, mask_path = None):

    image = skimage.io.imread(image_path)
    # Detect objects
    r = model.detect([image], verbose=1)[0]

    if(not os.path.exists(mask_path)): os.makedirs(mask_path)
    if(not os.path.exists(mask_path + "/" + image_path.split("/")[-1])): copyfile(image_path, mask_path + "/" + image_path.split("/")[-1])
    for idx, i in enumerate(list(r["class_ids"])):
        mask = r["masks"][:, :, idx] + 0 

        mask_name = mask_path + "/" + CLASS_IDS_REVERSED[i] + ".png"

        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_name)

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'test'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--weights_test_path', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file, that resulted from the training process")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    '''if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"'''

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = PeriocularConfig()
    else:
        class InferenceConfig(PeriocularConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    elif args.weights.lower() == "custom":
        weights_path = args.weights
    elif args.weights.lower() == "random":
        weights_path = None
    elif args.weights.lower() == "test":
        weights_path = args.weights_test_path
    else:
        print("Wrong argument for \"weights\"")
        sys.exit()

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    elif(args.weights.lower() == "custom"):
        model.load_weights(weights_path, by_name=True)
    elif(args.weights.lower() == "random"): pass
    elif(args.weights.lower() == "test"):
        model.load_weights(weights_path, by_name=True)
    else: pass

    # Train or evaluate
    if args.command == "train":
        train(model)
        
        # ---------------------------------------------------------------------------------------------------------------
        # rename the directory
        # ---------------------------------------------------------------------------------------------------------------
        suffix_to_add = CLASS_TO_DETECT
        directory = natsorted(list(filter(lambda x : x[0] != ".", os.listdir(args.logs))))[-1]
        os.makedirs(args.logs + "/periocular_" + suffix_to_add, exist_ok = True)
        
        for i in list(filter(lambda x : x[0] != ".", os.listdir(args.logs + "/" + directory))):
            os.rename(args.logs + "/" + directory + "/" + i, args.logs + "/" + directory + "_" + suffix_to_add + "/" + i)
        
        rmtree(args.logs + "/" + directory)

    elif args.command == "test":
        '''if(os.path.exists("test_masks")): rmtree("test_masks")
        os.makedirs("test_masks")'''

        if(not os.path.exists("test_masks")): os.makedirs("test_masks")
        
        test_images = natsorted(list(filter(lambda x : x[0] != ".", os.listdir("test_images"))))

        for i in test_images:
            try:
                Image.open("test_images/" + i).resize((512, 512), Image.LANCZOS).save("test_images/" + i)
            except: continue

            test(model, image_path = "test_images/" + i, mask_path = "test_masks/" + i.replace(".jpg", ""))

    else:
        print("'{}' is not recognized. Use 'train' or 'test'".format(args.command))