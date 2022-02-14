import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

##############################################################################################################################################################################
# CONTROL VARIABLES
##############################################################################################################################################################################
# ----------------------------------------------------------------
# common variables
# ----------------------------------------------------------------
# general
IMAGE_SIZE = 256
GENUINE_OR_IMPOSTOR = "G"
TRAIN_GAN_PAIRS = True
TRAIN_GAN_SINGLE = False
TRAIN_CNN = False
TRAIN_CNN_SIDE = False
TRAIN_LATENT_MAPPER = False
GENERATE_DATASET = False
MODEL_SAVE_FREQ = 1
NUM_BATCHES_VALIDATION_SAMPLES = 5
KEEP_PLOT = False
SAVE_CONFIGS_AND_RESULTS = True
OVERWRITE_SAVED_MODELS = False

# stylegan2
DATASET_SAVE_LOCATION = "./stylegan2/outputs/dataset/dataset.lmdb"
GAN_DATA_PATH = "../../../dataset/data/train/"
CNN_DATA_PATH = "../../../dataset/data/"
LATENT_DIM = 512

# cnn side
CNN_SIDE_TYPE = "resnet18" # resnet18 / inceptionv3 / densenet161

# -------------------------------------------------------------
# variables targeting "train_cnn.py"
# -------------------------------------------------------------
# model / training
CNN_NUM_EPOCHS = 15
CNN_BATCH_SIZE = 64
CNN_LEARNING_RATE = 2e-4
CNN_BETA_1 = 0.9
CNN_BETA_2 = 0.999
CNN_TYPE = "densenet161" # resnet18 / inceptionv3 / densenet161

# ---------------------------------------
# variables targeting "train_cnn_side.py"
# ---------------------------------------
# model / training
CNN_SIDE_NUM_EPOCHS = 15
CNN_SIDE_BATCH_SIZE = 64
CNN_SIDE_LEARNING_RATE = 2e-4
CNN_SIDE_BETA_1 = 0.9
CNN_SIDE_BETA_2 = 0.999

# ----------------------------------------------------------------------
# variables targeting "train_latent_mapper.py"
# ----------------------------------------------------------------------
# general
TRAIN_PERCENTAGE = 0.8
VALIDATION_PERCENTAGE = 0.1
TEST_PERCENTAGE = 0.1
SAMPLES_SAVE_FREQ = 1
LATENT_CODES_TYPE = "w" # either "z" or "w"
G_DATASET_PATH = "stylegan2/synthetic_dataset_G/"
I_DATASET_PATH = "stylegan2/synthetic_dataset_I/"

# latent mapper
LATENT_NUM_EPOCHS = 25
LATENT_BATCH_SIZE = 32
LATENT_LEARNING_RATE = 2e-4
LATENT_BETA_1 = 0.9
LATENT_BETA_2 = 0.999
LATENT_MODEL_TYPE = "densenet161" # resnet18 / inceptionv3 / densenet161

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# variables targeting "generate_synthetic_dataset.py"
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# general
SAVE_LATENT_CODES = True
GAN_PATH = "outputs/2020-10-15_14_57_46_stylegan2_G/checkpoints/072000.pt" if(GENUINE_OR_IMPOSTOR == "G") else "outputs/2020-10-19_00_59_27_stylegan2_I/checkpoints/042000.pt"
CNN_SIDE_PATH = "../outputs/2020-10-23_15_18_13_resnet18_side/models/epoch3.pt"

# stylegan2
NUM_IMAGES = 184000
NUM_SLAVES = 4
NUM_MLP = 8
TRUNCATION = 1.0
TRUNCATION_MEAN = 4096
CHANNEL_MULTIPLIER = 2

# ----------------------------------------------------------------------------------------------------------------------------
# variables targeting "get_segmentation_maps.py"
# ----------------------------------------------------------------------------------------------------------------------------
MRCNN_IRIS_WEIGHTS_PATH = "../mask_rcnn/mask_RCNN_master/logs/periocular20210113T1504_iris/mask_rcnn_periocular_0030.h5"
MRCNN_EYEBROW_WEIGHTS_PATH = "../mask_rcnn/mask_RCNN_master/logs/periocular20210113T1658_eyebrow/mask_rcnn_periocular_0030.h5"
MRCNN_SCLERA_WEIGHTS_PATH = "../mask_rcnn/mask_RCNN_master/logs/periocular20210113T1832_sclera/mask_rcnn_periocular_0030.h5"
MRCNN_SKIN_WEIGHTS_PATH = "../mask_rcnn/mask_RCNN_master/logs/periocular20210113T2152_skin/mask_rcnn_periocular_0030.h5"

def check_for_errors(r_value): # auxiliary function, checks for errors when running the main scripts

    if(r_value != 0):
        print("ERROR FOUND, STOPPING NOW...")
        sys.exit()

if(__name__ == "__main__"):

    if(TRAIN_GAN_PAIRS):

        # prepare the data for the training of StyleGAN2
        print("\nPREPARING THE DATA FOR TRAINING...")
        r_value = os.system("python3 stylegan2/prepare_data.py --out " + DATASET_SAVE_LOCATION + " --size " + str(IMAGE_SIZE) + " " + GAN_DATA_PATH + ("1/" if(GENUINE_OR_IMPOSTOR == "G") else "0/"))
        check_for_errors(r_value)
        print("DONE PREPARING THE DATA FOR TRAINING!")

        # train the StyleGAN2 model on our data
        print("\nTRAINING STYLEGAN2...")
        r_value = os.system("python3 stylegan2/train_gan.py " + DATASET_SAVE_LOCATION + " --genuine_or_impostor " + GENUINE_OR_IMPOSTOR)
        check_for_errors(r_value)
        print("DONE TRAINING STYLEGAN2!")
    
    if(TRAIN_GAN_SINGLE):

        # prepare the data for the training of StyleGAN2
        print("\nPREPARING THE DATA FOR TRAINING...")
        r_value = os.system("python3 stylegan2/prepare_data_single.py --out " + DATASET_SAVE_LOCATION + " --size " + str(IMAGE_SIZE) + " " + GAN_DATA_PATH + "0/")
        check_for_errors(r_value)
        print("DONE PREPARING THE DATA FOR TRAINING!")

        # train the StyleGAN2 model on our data
        print("\nTRAINING STYLEGAN2...")
        r_value = os.system("python3 stylegan2/train_gan_single.py " + DATASET_SAVE_LOCATION)
        check_for_errors(r_value)
        print("DONE TRAINING STYLEGAN2!")

    if(TRAIN_CNN):
        
        # train a CNN to distinguish between genuine and impostor pairs
        print("\nTRAINING A CNN TO CLASSIFY PAIRS...")
        args = [str(MODEL_SAVE_FREQ), str(NUM_BATCHES_VALIDATION_SAMPLES), str(KEEP_PLOT), str(SAVE_CONFIGS_AND_RESULTS), str(OVERWRITE_SAVED_MODELS), CNN_DATA_PATH, str(IMAGE_SIZE), str(CNN_NUM_EPOCHS), str(CNN_BATCH_SIZE), str(CNN_LEARNING_RATE), str(CNN_BETA_1), str(CNN_BETA_2), CNN_TYPE]
        r_value = os.system("python3 train_cnn.py " + " ".join(args))
        check_for_errors(r_value)
        print("DONE TRAINING A CNN TO CLASSIFY PAIRS!")

    if(TRAIN_CNN_SIDE):
        
        # train a CNN that predicts an image's side
        print("\nTRAINING A CNN TO DISTINGUISH AN IMAGE'S SIDE...")
        args = [str(MODEL_SAVE_FREQ), str(NUM_BATCHES_VALIDATION_SAMPLES), str(KEEP_PLOT), str(SAVE_CONFIGS_AND_RESULTS), str(OVERWRITE_SAVED_MODELS), CNN_DATA_PATH, str(IMAGE_SIZE), str(CNN_SIDE_NUM_EPOCHS), str(CNN_SIDE_BATCH_SIZE), str(CNN_SIDE_LEARNING_RATE), str(CNN_SIDE_BETA_1), str(CNN_SIDE_BETA_2), CNN_SIDE_TYPE]
        r_value = os.system("python3 train_cnn_side.py " + " ".join(args))
        check_for_errors(r_value)
        print("DONE TRAINING A CNN TO DISTINGUISH AN IMAGE'S SIDE!")

    if(TRAIN_LATENT_MAPPER):
        
        # train a CNN that predicts an image's side
        print("\nTRAINING A MODEL TO MAP IMAGES TO LATENT SPACE...")
        args = [str(TRAIN_PERCENTAGE), str(VALIDATION_PERCENTAGE), str(TEST_PERCENTAGE), str(SAMPLES_SAVE_FREQ), str(MODEL_SAVE_FREQ), str(NUM_BATCHES_VALIDATION_SAMPLES), str(KEEP_PLOT), str(SAVE_CONFIGS_AND_RESULTS), str(OVERWRITE_SAVED_MODELS), str(LATENT_CODES_TYPE), G_DATASET_PATH, I_DATASET_PATH, str(IMAGE_SIZE), str(LATENT_NUM_EPOCHS), str(LATENT_BATCH_SIZE), str(LATENT_LEARNING_RATE), str(LATENT_BETA_1), str(LATENT_BETA_2), LATENT_MODEL_TYPE, str(LATENT_DIM)]
        r_value = os.system("python3 train_latent_mapper.py " + " ".join(args))
        check_for_errors(r_value)
        print("DONE TRAINING A MODEL TO MAP IMAGES TO LATENT SPACE!")

    if(GENERATE_DATASET):
        
        # generate a large synthetic dataset using the trained generator (from StyleGAN2)
        print("\nGENERATING A LARGE SYNTHETIC DATASET...")
        args = [str(SAVE_LATENT_CODES), GAN_PATH, CNN_SIDE_TYPE, CNN_SIDE_PATH, str(IMAGE_SIZE), str(NUM_IMAGES), str(NUM_SLAVES), str(LATENT_DIM), str(NUM_MLP), str(TRUNCATION), str(TRUNCATION_MEAN), str(CHANNEL_MULTIPLIER), GENUINE_OR_IMPOSTOR]
        r_value = os.system("python3 stylegan2/generate_synthetic_dataset.py " + " ".join(args))
        check_for_errors(r_value)
        print("DONE GENERATING A LARGE SYNTHETIC DATASET!")

        # compute the segmentation maps for the synthetic images
        print("\nCOMPUTING THE SEGMENTATION MAPS...")
        args = [GENUINE_OR_IMPOSTOR, str(NUM_IMAGES), str(NUM_SLAVES), str(IMAGE_SIZE), MRCNN_IRIS_WEIGHTS_PATH, MRCNN_EYEBROW_WEIGHTS_PATH, MRCNN_SCLERA_WEIGHTS_PATH, MRCNN_SKIN_WEIGHTS_PATH]
        r_value = os.system("python3 stylegan2/get_segmentation_maps.py " + " ".join(args))
        check_for_errors(r_value)
        print("DONE COMPUTING THE SEGMENTATION MAPS!")

    print("")