import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import torch
import inspect
import numpy as np
from PIL import Image
from shutil import rmtree
from model import Generator
from functools import partial
from torchvision import utils as torch_utils

np.set_printoptions(threshold = sys.maxsize)

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from find_neighbours_master import determine_side, load_cnn_side

#####################################################################################################################################################################################
# CONTROL VARIABLES
#####################################################################################################################################################################################
# the values are the default ones (NOTE: if the values are to be changed, do so in this "if" branch)
if(len(sys.argv) == 1):
    # general
    GENUINE_OR_IMPOSTOR = "G"
    SAVE_LATENT_CODES = False
    GAN_PATH = "../outputs/2021-01-11_20_52_02_stylegan2_G/checkpoints/070000.pt" if(GENUINE_OR_IMPOSTOR == "G") else "outputs/2020-10-19_00_59_27_stylegan2_I/checkpoints/042000.pt"
    CNN_SIDE_TYPE = "resnet18" # resnet18 / inceptionv3 / densenet161
    CNN_SIDE_PATH = "../outputs/2020-10-23_15_18_13_resnet18_side/models/epoch3.pt"
    
    # stylegan2
    IMAGE_SIZE = 256
    NUM_IMAGES = 1000000
    NUM_SLAVES = 4
    LATENT_DIM = 512
    NUM_MLP = 8
    TRUNCATION = 1.0
    TRUNCATION_MEAN = 4096
    CHANNEL_MULTIPLIER = 2

# the values are coming from the "run_training.py" master script
else:
    # general
    SAVE_LATENT_CODES = True if(sys.argv[1] == "True") else False
    GAN_PATH = sys.argv[2]
    CNN_SIDE_TYPE = sys.argv[3]
    CNN_SIDE_PATH = sys.argv[4]
    
    # stylegan2
    IMAGE_SIZE = int(sys.argv[5])
    NUM_IMAGES = int(sys.argv[6])
    NUM_SLAVES = int(sys.argv[7])
    LATENT_DIM = int(sys.argv[8])
    NUM_MLP = int(sys.argv[9])
    TRUNCATION = int(sys.argv[10])
    TRUNCATION_MEAN = int(sys.argv[11])
    CHANNEL_MULTIPLIER = int(sys.argv[12])
    GENUINE_OR_IMPOSTOR = sys.argv[13]

activation = {}
def get_activation(name): # auxiliary function, retrieves a layer's output

    def hook(model, input, output):
        activation[name] = output.detach()
    
    return hook

def get_pair(dataset_dir, sample): # auxiliary function, unfolds the pair

    torch_utils.save_image(
        torch.from_numpy(sample.detach().cpu().numpy()[0,:3,:,:]),
        dataset_dir + "/temp.png",
        nrow=int(64 ** 0.5),
        normalize=True,
        range=(-1, 1)
    )

    first_image = np.asarray(Image.open(dataset_dir + "/temp.png"))

    # save the second image from this pair
    torch_utils.save_image(
        torch.from_numpy(sample.detach().cpu().numpy()[0,3:,:,:]),
        dataset_dir + "/temp.png",
        nrow=int(64 ** 0.5),
        normalize=True,
        range=(-1, 1)
    )

    second_image = np.asarray(Image.open(dataset_dir + "/temp.png"))
    
    os.remove(dataset_dir + "/temp.png")

    return((first_image, second_image))

if(__name__ == "__main__"):
    
    ##################################################################################################################
    # INITIAL SETUP
    ##################################################################################################################
    device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

    if(os.path.exists("/media/socialab/HARD disc/synthetic_dataset_" + GENUINE_OR_IMPOSTOR)): rmtree("/media/socialab/HARD disc/synthetic_dataset_" + GENUINE_OR_IMPOSTOR)
    os.makedirs("/media/socialab/HARD disc/synthetic_dataset_" + GENUINE_OR_IMPOSTOR)

    if(SAVE_LATENT_CODES):
        for i in range(NUM_SLAVES):
            #os.makedirs("/media/socialab/HARD disc/synthetic_dataset_" + GENUINE_OR_IMPOSTOR + "/latent_z_codes_" + str(i + 1))
            os.makedirs("/media/socialab/HARD disc/synthetic_dataset_" + GENUINE_OR_IMPOSTOR + "/latent_w_codes_" + str(i + 1))

    for i in range(NUM_SLAVES):
        os.makedirs("/media/socialab/HARD disc/synthetic_dataset_" + GENUINE_OR_IMPOSTOR + "/images_" + str(i + 1))

    # load the CNN that predicts an image's side
    model = load_cnn_side(model_type = CNN_SIDE_TYPE, cnn_path = CNN_SIDE_PATH)

    determine_side_partial = partial(determine_side, model_type = CNN_SIDE_TYPE, model = model)

    ######################################################################################################
    # LOAD THE GENERATOR
    ######################################################################################################
    g_ema = Generator(IMAGE_SIZE, LATENT_DIM, NUM_MLP, channel_multiplier = CHANNEL_MULTIPLIER).to(device)

    ckpt = torch.load(GAN_PATH)
    g_ema.load_state_dict(ckpt["g_ema"])

    if(TRUNCATION < 1):
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(TRUNCATION_MEAN)
    else: mean_latent = None

    g_ema.eval()

    dir_number = "1"
    
    ############################################################################################################################################################################################################
    # GENERATE A BIG SYNTHETIC DATASET
    ############################################################################################################################################################################################################
    synthetic_counter = 1
    while(synthetic_counter <= NUM_IMAGES):
        with torch.no_grad():

            print(str(synthetic_counter) + "/" + str(NUM_IMAGES))

            if((synthetic_counter % (NUM_IMAGES // NUM_SLAVES)) == 0): dir_number = str(int(dir_number) + 1)
            
            # --------------------------------------------------------------------------------------
            # generate a synthetic pair
            # --------------------------------------------------------------------------------------
            # generate random noise
            sample_z = torch.randn(1, LATENT_DIM, device = device)

            # obtain a synthetic sample based on the given noise
            g_ema.style.register_forward_hook(get_activation("8"))
            sample, _ = g_ema([sample_z], truncation = TRUNCATION, truncation_latent = mean_latent)
            
            sample_w = activation["8"]

            first_image, second_image = get_pair("/media/socialab/HARD disc/synthetic_dataset_" + GENUINE_OR_IMPOSTOR, sample)
        
            # -------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # determine the side configuration of the generated pair
            # -------------------------------------------------------------------------------------------------------------------------------------------------------------------
            first_image_side_confidence = determine_side_partial(img = first_image)
            second_image_side_confidence = determine_side_partial(img = second_image)
            
            # if the CNN got confused and doesn't "agree" on the images' sides, we opt to use the side the CNN is most confidant of
            if(first_image_side_confidence[0] != second_image_side_confidence[0]):

                if(first_image_side_confidence[1] >= second_image_side_confidence[1]): side_configuration = first_image_side_confidence[0] + "_" + first_image_side_confidence[0]
                
                else: side_configuration = second_image_side_confidence[0] + "_" + second_image_side_confidence[0]

            else: side_configuration = first_image_side_confidence[0] + "_" + second_image_side_confidence[0]

            # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # save the generated pair and, if required, its corresponding latent code
            # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # save the generated pair as a ".npy" file
            pair_save_path = "/media/socialab/HARD disc/synthetic_dataset_" + GENUINE_OR_IMPOSTOR + "/images_" + dir_number + "/" + f"{str(synthetic_counter).zfill(len(str(NUM_IMAGES)))}" + "_" + side_configuration
            pair = np.dstack((first_image, second_image))
            np.save(pair_save_path, pair)
            
            # save the generated pair as an actual image
            #pair = np.column_stack((first_image, second_image)).astype(np.uint8)
            #pair_save_path = "/media/socialab/HARD disc/synthetic_dataset_" + GENUINE_OR_IMPOSTOR + "/images_" + dir_number + "/" + f"{str(synthetic_counter).zfill(len(str(NUM_IMAGES)))}" + "_" + side_configuration
            #Image.fromarray(pair).save(pair_save_path + ".jpg")

            if(SAVE_LATENT_CODES): 
                #np.save("/media/socialab/HARD disc/synthetic_dataset_" + GENUINE_OR_IMPOSTOR + "/latent_z_codes_" + dir_number + "/" + f"{str(synthetic_counter).zfill(len(str(NUM_IMAGES)))}" + "_" + side_configuration, sample_z.cpu())
                np.save("/media/socialab/HARD disc/synthetic_dataset_" + GENUINE_OR_IMPOSTOR + "/latent_w_codes_" + dir_number + "/" + f"{str(synthetic_counter).zfill(len(str(NUM_IMAGES)))}" + "_" + side_configuration, sample_w.cpu())

            synthetic_counter += 1