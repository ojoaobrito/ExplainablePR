import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import time
import torch
import subprocess
import numpy as np
import torch.nn as nn
from PIL import Image
from shutil import rmtree
from functools import partial 
from natsort import natsorted
import torchvision.models as models
import torchvision.transforms as transforms
from skimage.metrics import mean_squared_error as mse

####################################################################################################
# CONTROL VARIABLES
####################################################################################################
# the values are the default ones (NOTE: if the values are to be changed, do so in this "if" branch)
if(len(sys.argv) == 1):
    IMAGE_SIZE = 128
    SAVE_NEIGHBOURS = True
    K_NEIGHBOURS = 15
    MODE = "elementwise_comparison" # either "elementwise_comparison" or "full_comparison"
    CNN_SIDE_TYPE = "resnet18" # resnet18 / inceptionv3 / densenet161
    CNN_SIDE_PATH = "outputs/2020-10-23_15_18_13_resnet18_side/models/epoch3.pt"
    IOU_OR_IMAGE_REGISTRATION = "IoU"
    USE_SEGMENTATION_DATA = False
    IRIS_COMBINATION = "C_C"

# the values are coming from the "explain_pair.py" master script
else:
    IMAGE_SIZE = int(sys.argv[1])
    SAVE_NEIGHBOURS = True if(sys.argv[2] == "True") else False
    K_NEIGHBOURS = int(sys.argv[3])
    MODE = sys.argv[4]
    CNN_SIDE_TYPE = sys.argv[5]
    CNN_SIDE_PATH = sys.argv[6]
    IOU_OR_IMAGE_REGISTRATION = sys.argv[7]
    USE_SEGMENTATION_DATA = True if(sys.argv[8] == "True") else False
    IRIS_COMBINATION = sys.argv[9]

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

def compute_mse(img1, img2): # auxiliary function, computes the MSE distance between images "img1" and "img2"

    return(mse(img1, img2))

if(__name__ == "__main__"):

    if(os.path.exists("results")): rmtree("results")
    os.makedirs("results")

    if(SAVE_NEIGHBOURS):
        if(os.path.exists("neighbours")): rmtree("neighbours")
        os.makedirs("neighbours")
    
    device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

    directory = natsorted(list(filter(lambda x : x[0] != ".", os.listdir("../images"))))

    # load the test images
    image_A = np.asarray(Image.open("../images/" + directory[0]).resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS))
    image_B = np.asarray(Image.open("../images/" + directory[1]).resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS))
    
    genuine_or_impostor = "G" if("I" in directory[0]) else "I"

    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    # get the images' side configuration
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    if(("_L" in directory[0]) or ("_R" in directory[0])): side_configuration = directory[0].split("_")[-2] + "_" + directory[0].split("_")[-2]
    else:
        # load the CNN that predicts an image's side
        model = load_cnn_side(model_type = CNN_SIDE_TYPE, cnn_path = CNN_SIDE_PATH)

        determine_side_partial = partial(determine_side, model_type = CNN_SIDE_TYPE, model = model)
        
        image_A_side_confidence = determine_side_partial(img = image_A)
        image_B_side_confidence = determine_side_partial(img = image_B)
        
        # if the CNN got confused and doesn't "agree" on the images' sides, we opt to use the side that deserved the highest amount of confidence from the CNN
        if(image_A_side_confidence[0] != image_B_side_confidence[0]):

            if(image_A_side_confidence[1] >= image_B_side_confidence[1]): side_configuration = image_A_side_confidence[0] + "_" + image_A_side_confidence[0]
            else: side_configuration = image_B_side_confidence[0] + "_" + image_B_side_confidence[0]

        else: side_configuration = image_A_side_confidence[0] + "_" + image_B_side_confidence[0]

        # just for documentation, add the side configuration to both images' names
        os.rename("../images/" + directory[0], "../images/" + "_".join(directory[0].split("_")[:-1]) + "_" + side_configuration[0] + "_" + directory[0].split("_")[-1])
        os.rename("../images/" + directory[1], "../images/" + "_".join(directory[1].split("_")[:-1]) + "_" + side_configuration[2] + "_" + directory[1].split("_")[-1])

    # retrieve the total number of slaves and synthetic images at our disposal
    synthetic_diretories = list(filter(lambda x : "images_" in x, os.listdir("stylegan2/synthetic_dataset_" + genuine_or_impostor)))
    num_slaves = len(synthetic_diretories)
    num_images = sum([len(list(filter(lambda x : x[0] != ".", os.listdir("stylegan2/synthetic_dataset_" + genuine_or_impostor + "/" + i)))) for i in synthetic_diretories])
    
    milestones = np.linspace(0, num_images + 1, num_slaves + 1).astype(int)

    ###############################################################################################
    # TELL THE SLAVES TO FIND NEIGHBOURS IN A SMALLER PORTION OF THE SEARCH SPACE
    ###############################################################################################
    with open("neighbour_names.txt", "w") as file:
        file.write("")
    
    processes = []
    t0 = time.time()
    for i in range(num_slaves):
        
        # prepare some arguments
        source_dir = "stylegan2/synthetic_dataset_" + genuine_or_impostor + "/images_" + str(i + 1)
        start_stop = str(milestones[i]) + "_" + str(milestones[i + 1])

        args = [str(IMAGE_SIZE), source_dir, start_stop, side_configuration, str(K_NEIGHBOURS), 
            MODE, IOU_OR_IMAGE_REGISTRATION, str(USE_SEGMENTATION_DATA), IRIS_COMBINATION]

        # add this process to the process list
        processes.append(
            subprocess.Popen(
                ["python3", "find_neighbours_slave.py"] + args, 
                stdout = subprocess.PIPE, 
                stderr = subprocess.STDOUT
            )
        )
        #os.system("python3 find_neighbours_slave.py " + " ".join(args))
        #sys.exit(10)
    
    # wait for the slaves to finish
    for p in processes: p.wait()
    
    ##################################################################################################################
    # BRING EVERYTHING TOGETHER
    ##################################################################################################################
    # ----------------------------------------------------------------------------------------------------------------
    # get the "K_NEIGHBOURS" best neighbours
    # ----------------------------------------------------------------------------------------------------------------
    distances = natsorted(list(filter(lambda x : "distances" in x, os.listdir("results"))))
    neighbours = natsorted(list(filter(lambda x : ((x[0] != ".") and ("distances" not in x)), os.listdir("results"))))

    with open("neighbour_names.txt", "r") as file:
        names = file.read().splitlines()

    neighbours_images = []
    neighbours_names = []
    best_neighbours = []
    count = 1
    index_counter = -1
    for i in range(num_slaves):
        
        slave_distances = np.load("results/" + distances[i])
        slave_neighbours = np.load("results/" + neighbours[i])
        
        # go through this slave's distances and neighbours
        for jdx, j in enumerate(slave_distances):
            
            # the slave couldn't find that many neighbours, so let's skip this part
            if(np.array_equal(slave_neighbours[jdx], np.zeros((IMAGE_SIZE, IMAGE_SIZE * 2, 3)))): continue

            neighbours_images.append(slave_neighbours[jdx])
            index_counter += 1
            neighbours_names.append(names[index_counter])

            if(len(best_neighbours) < K_NEIGHBOURS):
                best_neighbours.append([index_counter, slave_distances[jdx]])
                continue
            
            # ---------------------------------------------------------------
            # in case we used the "MSE" distance metric
            # ---------------------------------------------------------------
            worst_neighbour = max(best_neighbours, key = lambda x : x[1])

            if(slave_distances[jdx] < worst_neighbour[1]):
                best_neighbours.remove(worst_neighbour)
                best_neighbours.append([index_counter, slave_distances[jdx]])

    # -------------------------------------------------------------------------------
    # save the "K_NEIGHBOURS" best neighbours
    # -------------------------------------------------------------------------------
    best_neighbours_np = np.zeros((K_NEIGHBOURS, IMAGE_SIZE, IMAGE_SIZE, 3))
    best_neighbours_distances_np = np.zeros((K_NEIGHBOURS))
    best_neighbours_names = []
    
    for idx, i in enumerate(best_neighbours):
        
        best_neighbours_np[idx, :, :, :] = neighbours_images[i[0]][:, IMAGE_SIZE:, :]
        best_neighbours_distances_np[idx] = i[1]
        best_neighbours_names.append(i[0])
    
    # ---------------------------------------------------------------------------------------
    # sort the "K_NEIGHBOURS" best neighbours with regards to their MSE/SSIM/latent distances
    # ---------------------------------------------------------------------------------------
    best_neighbours_final_list_aux = []
    
    # sort the neighbours' indexes with regards to the MSE/latent distances
    while(len(best_neighbours_final_list_aux) != K_NEIGHBOURS):

        minimum = (0, sys.maxsize)
        for idx, i in enumerate(best_neighbours_distances_np):
            if((i < minimum[1]) and (idx not in best_neighbours_final_list_aux)):
                minimum = (idx, i)

        best_neighbours_final_list_aux.append(minimum[0])

    best_neighbours_final_list = []
    best_neighbours_names_final_list = []

    # we only sorted the indexes, so let's put the actual neighbours in the final list
    for i in best_neighbours_final_list_aux:
        best_neighbours_final_list.append(best_neighbours_np[i])
        best_neighbours_names_final_list.append(names[best_neighbours_names[i]])
    
    # convert to numpy and sort the distances
    best_neighbours_np = np.asarray(best_neighbours_final_list)
    best_neighbours_distances_np.sort()

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # save everything
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    np.save("best_neighbours.npy", best_neighbours_np)
    np.save("best_neighbours_distances.npy", best_neighbours_distances_np)
    
    if(SAVE_NEIGHBOURS): 
        for idx, i in enumerate(best_neighbours_np):
            Image.fromarray(i.astype(np.uint8)).save("neighbours/" + str(idx + 1) + "_" + str(round(best_neighbours_distances_np[idx], 3)).replace(".", ",") + "_" + best_neighbours_names_final_list[idx].split("/")[-1].replace(".npy", ".jpg"))
    
    #print("\n[INFO] ELAPSED TIME: %.2fs\n" % (time.time() - t0))
    #rmtree("results")