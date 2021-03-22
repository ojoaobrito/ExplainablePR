import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from model import Generator
from functools import partial
from natsort import natsorted
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import utils as torch_utils

np.set_printoptions(threshold = sys.maxsize)

##################################################################################################################################################################################################
# CONTROL VARIABLES
##################################################################################################################################################################################################
# general
GENUINE_OR_IMPOSTOR = "G"
SAVE_LATENT_CODES = False
GAN_PATH = "../../../trained_models/stylegan_2_G/checkpoints/" + natsorted(list(filter(lambda x : x[0] != ".", os.listdir("../../../trained_models/stylegan_2_G/checkpoints/"))))[-1]
CNN_SIDE_TYPE = "resnet_18" # either "resnet_18", "inception_v3" or "densenet_161"
CNN_SIDE_PATH = "../../../trained_models/" + CNN_SIDE_TYPE + "/models/" + natsorted(list(filter(lambda x : x[0] != ".", os.listdir("../../../trained_models/" + CNN_SIDE_TYPE + "/models/"))))[-1]

# stylegan2
IMAGE_SIZE = 256
NUM_IMAGES = 1000000
NUM_SLAVES = 4
LATENT_DIM = 512
NUM_MLP = 8
TRUNCATION = 1.0
TRUNCATION_MEAN = 4096
CHANNEL_MULTIPLIER = 2

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

def load_cnn_side(model_type, cnn_path): # auxiliary function, loads the CNN responsible for determining the side (left or right) of a given image

    if(model_type == "resnet_18"):

        model = models.resnet18()

        model.fc = nn.Sequential(
            nn.Linear(in_features = 512, out_features = 2, bias = True),
            nn.Softmax(dim = 1),
        )
        model = model.cuda()

    elif(model_type == "inception_v3"):

        model = models.inception_v3()
        
        model.fc = nn.Sequential(
            nn.Linear(in_features = 2048, out_features = 2, bias = True),
            nn.Softmax(dim = 1),
        )
        model = model.cuda()

    else:

        model = models.densenet161()
            
        model.classifier = nn.Sequential(
            nn.Linear(in_features = 2208, out_features = 2, bias = True),
            nn.Softmax(dim = 1),
        )
        model = model.cuda()

    model.load_state_dict(torch.load(cnn_path))
    model.eval()

    return(model)

def determine_side(model_type, model, img): # auxiliary function, determines the side of the input image

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    img = img.copy()
    img_torch = torch.reshape(transform(img), (1, 3, img.shape[0], img.shape[1])).cuda()

    if(model_type == "inceptionv3"): outputs = torch.squeeze(model(img_torch)[0]).detach().cpu().numpy()
    else: outputs = torch.squeeze(model(img_torch)).detach().cpu().numpy()

    confidence = np.max(outputs)
    side = ["L", "R"][np.argmax(outputs)]

    return((side, confidence))

if(__name__ == "__main__"):
    
    ##################################################################################################################
    # INITIAL SETUP
    ##################################################################################################################
    device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

    os.makedirs("synthetic_dataset_" + GENUINE_OR_IMPOSTOR, exist_ok = True)

    if(SAVE_LATENT_CODES):
        for i in range(NUM_SLAVES):
            os.makedirs("synthetic_dataset_" + GENUINE_OR_IMPOSTOR + "/latent_w_codes_" + str(i + 1))

    for i in range(NUM_SLAVES):
        os.makedirs("synthetic_dataset_" + GENUINE_OR_IMPOSTOR + "/images_" + str(i + 1))

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

            first_image, second_image = get_pair("synthetic_dataset_" + GENUINE_OR_IMPOSTOR, sample)
        
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
            pair_save_path = "synthetic_dataset_" + GENUINE_OR_IMPOSTOR + "/images_" + dir_number + "/" + f"{str(synthetic_counter).zfill(len(str(NUM_IMAGES)))}" + "_" + side_configuration
            pair = np.dstack((first_image, second_image))
            np.save(pair_save_path, pair)
            
            # save the generated pair as an actual image
            #pair = np.column_stack((first_image, second_image)).astype(np.uint8)
            #pair_save_path = "synthetic_dataset_" + GENUINE_OR_IMPOSTOR + "/images_" + dir_number + "/" + f"{str(synthetic_counter).zfill(len(str(NUM_IMAGES)))}" + "_" + side_configuration
            #Image.fromarray(pair).save(pair_save_path + ".jpg")

            if(SAVE_LATENT_CODES): 
                np.save("synthetic_dataset_" + GENUINE_OR_IMPOSTOR + "/latent_w_codes_" + dir_number + "/" + f"{str(synthetic_counter).zfill(len(str(NUM_IMAGES)))}" + "_" + side_configuration, sample_w.cpu())

            synthetic_counter += 1