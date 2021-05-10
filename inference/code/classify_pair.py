import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from natsort import natsorted
import torchvision.models as models
import torchvision.transforms as transforms

####################################################################################################
# CONTROL VARIABLES
####################################################################################################
# the values are the default ones (NOTE: if the values are to be changed, do so in this "if" branch)
if(len(sys.argv) == 1):
    IMAGE_SIZE = 128
    CNN_TYPE = "densenet161"
    CNN_PATH = "../../trained_models/densenet_161/models/densenet_161.pt"

# the values are coming from the "explain_pair.py" master script
else:
    IMAGE_SIZE = int(sys.argv[1])
    CNN_TYPE = sys.argv[2]
    CNN_PATH = sys.argv[3]

if(__name__ == "__main__"):

    ####################################################################################
    # INITIAL SETUP
    ####################################################################################
    directory = natsorted(list(filter(lambda x : x[0] != ".", os.listdir("../images"))))
    
    # the images were already classified
    if(("I" in directory[0]) or ("G" in directory[0])): sys.exit()
    
    device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
    
    ###############################################################################################################################################
    # PREPARE THE MODEL
    ###############################################################################################################################################
    if(CNN_TYPE == "resnet18"):

        model = models.resnet18()

        model.conv1 = nn.Conv2d(in_channels = 6, out_channels = 64, kernel_size = (7, 7), stride = (2, 2), padding = (3, 3), bias = False)

        model.fc = nn.Sequential(
            nn.Linear(in_features = 512, out_features = 2, bias = True),
            nn.Softmax(dim = 1),
        )
        model = model.to(device)

    elif(CNN_TYPE == "inceptionv3"):

        model = models.inception_v3()
        
        model.Conv2d_1a_3x3.conv = nn.Conv2d(in_channels = 6, out_channels = 32, kernel_size = (3, 3), stride = (2, 2), bias = False)

        model.fc = nn.Sequential(
            nn.Linear(in_features = 2048, out_features = 2, bias = True),
            nn.Softmax(dim = 1),
        )
        model = model.to(device)

    else:
        model = models.densenet161()
        
        model.features.conv0 = nn.Conv2d(in_channels = 6, out_channels = 96, kernel_size = (7, 7), stride = (2, 2), padding = (3, 3), bias = False)

        model.classifier = nn.Sequential(
            nn.Linear(in_features = 2208, out_features = 2, bias = True),
            nn.Softmax(dim = 1),
        )
        model = model.to(device)

    model.load_state_dict(torch.load(CNN_PATH))
    model.eval()

    ####################################################################################################################################################################
    # CLASSIFY THE UNLABELED PAIR
    ####################################################################################################################################################################
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image_A = transform(Image.open("../images/" + directory[0]).resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS))
    image_B = transform(Image.open("../images/" + directory[1]).resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS))

    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------
    # get the output from the CNN (by giving it the pair in the original and inversed order, to see which of them gets the highest degree of confidance from the CNN)
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------
    outputs_dict = {"A_B": ["", 0.0], "B_A": ["", 0.0]}

    for i in ["A_B", "B_A"]:
        if(i == "A_B"):
            unlabeled_pair = torch.reshape(torch.cat((image_A, image_B), 0), (1, 6, IMAGE_SIZE, IMAGE_SIZE)).cuda()

        else:
            unlabeled_pair = torch.reshape(torch.cat((image_B, image_A), 0), (1, 6, IMAGE_SIZE, IMAGE_SIZE)).cuda()
        
        # retrieve the predicted output
        if(CNN_TYPE == "inceptionv3"): outputs = torch.squeeze(model(unlabeled_pair)[0]).detach().cpu().numpy()
        else: outputs = torch.squeeze(model(unlabeled_pair)).detach().cpu().numpy()
        
        genuine_or_impostor = ["I", "G"][np.argmax(outputs)]

        outputs_dict[i] = [genuine_or_impostor, outputs[1]]

    # -----------------------------------------------------
    # keep the output with the highest degree of confidence
    # -----------------------------------------------------
    if(outputs_dict["A_B"][1] <= outputs_dict["B_A"][1]): 
        genuine_or_impostor = outputs_dict["A_B"][0]
        outputs = outputs_dict["A_B"][1]
    
    else:
        genuine_or_impostor = outputs_dict["B_A"][0]
        outputs = outputs_dict["B_A"][1]
    
    # add the predicted output to the images' names
    os.rename("../images/" + directory[0], "../images/a_" + genuine_or_impostor + "_" + str(round(outputs, 3)).replace(".", ",") + "_" + directory[0].replace("a_", ""))
    os.rename("../images/" + directory[1], "../images/b_" + genuine_or_impostor + "_" + str(round(outputs, 3)).replace(".", ",") + "_" + directory[1].replace("b_", ""))