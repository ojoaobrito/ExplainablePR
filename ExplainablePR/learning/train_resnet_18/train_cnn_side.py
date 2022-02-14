import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import time
import torch
import datetime
import numpy as np
import torch.nn as nn
from plot import Plot
from random import sample
from shutil import rmtree
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix

np.set_printoptions(threshold = sys.maxsize)

#####################################################################################
# CONTROL VARIABLES
#####################################################################################
# general
MODEL_SAVE_FREQ = 1
NUM_BATCHES_VALIDATION_SAMPLES = 5
KEEP_PLOT = False
SAVE_CONFIGS_AND_RESULTS = True
DATA_PATH = "../../../dataset/data/"

# model / training
IMAGE_SIZE = 128
NUM_EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 2e-4
BETA_1 = 0.9
BETA_2 = 0.999
CNN_SIDE_TYPE = "resnet_18" # either "resnet_18", "inception_v3" or "densenet_161"

def save_confusion_matrix(dir_name, ground_truth, predicted_labels): # auxiliary function, saves a confusion matrix

    _ = plt.figure()

    matrix = confusion_matrix(ground_truth, predicted_labels, labels = [0, 1])
    
    plt.matshow(matrix, cmap = "gray_r")
    plt.title("Confusion Matrix")
    plt.ylabel("Ground Truth")
    plt.xlabel("Predicted Label")
    plt.savefig(dir_name + "/confusion_matrix.jpg")

def remove_old_model(dir_name): # auxiliary function, removes an older version of the trained model

    rmtree(dir_name)
    os.makedirs(dir_name)

def prepare_data(): # auxiliary function, prepares the data

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train = datasets.ImageFolder(root = DATA_PATH + "train/", transform = transform)
    validation = datasets.ImageFolder(root = DATA_PATH + "validation/", transform = transform)
    test = datasets.ImageFolder(root = DATA_PATH + "test/", transform = transform)

    train_loader = DataLoader(train, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4, drop_last = True)
    validation_loader = DataLoader(validation, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4, drop_last = True)
    test_loader = DataLoader(test, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4, drop_last = True)

    return(train_loader, validation_loader, test_loader)

if(__name__ == "__main__"):

    device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

    #############################################################################
    # INITIAL STEPS
    #############################################################################
    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d_%H_%M_%S")

    # create some directories
    SAVE_DIR = "../../trained_models/" + CNN_SIDE_TYPE
    os.makedirs(SAVE_DIR, exist_ok = True)
    os.makedirs(SAVE_DIR + "/models")

    # load the data
    train_loader, validation_loader, test_loader = prepare_data()

    plot = None

    #####################################################################
    # PREPARE THE MODEL
    #####################################################################
    if(CNN_SIDE_TYPE == "resnet_18"):

        model = models.resnet18()

        model.fc = nn.Sequential(
            nn.Linear(in_features = 512, out_features = 2, bias = True),
            nn.Softmax(dim = 1),
        )
        model = model.to(device)

    elif(CNN_SIDE_TYPE == "inception_v3"):

        model = models.inception_v3()
        
        model.fc = nn.Sequential(
            nn.Linear(in_features = 2048, out_features = 2, bias = True),
            nn.Softmax(dim = 1),
        )
        model = model.to(device)

    else:

        model = models.densenet161()
        
        model.classifier = nn.Sequential(
            nn.Linear(in_features = 2208, out_features = 2, bias = True),
            nn.Softmax(dim = 1),
        )
        model = model.to(device)

    # define the loss function and optimizer
    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE, betas = (BETA_1, BETA_2))
    
    # ---------------------------------------------------------------------------------------------------------
    # select some validation samples to assess the model's performance
    # ---------------------------------------------------------------------------------------------------------
    validation_samples_aux = sample([i for i in range(len(validation_loader))], NUM_BATCHES_VALIDATION_SAMPLES)
    validation_samples = []

    for idx, data_batch in enumerate(validation_loader):
        if(idx in validation_samples_aux):
            
            img_batch, img_labels = data_batch
            validation_samples.append((img_batch, img_labels))

    ############################################################################################################################################
    # TRAIN THE MODEL
    ############################################################################################################################################
    for i in range(NUM_EPOCHS):
        loss_aux = 0.0
        model.train()
        for idx, data_batch in enumerate(train_loader):
            
            # read a batch of images
            img_batch, img_labels = data_batch
            img_batch = img_batch.to(device)

            # compute the loss
            if(CNN_SIDE_TYPE == "inceptionv3"): outputs = model(img_batch)[0]
            else: outputs = model(img_batch)
            optimizer.zero_grad()
            loss = criterion(outputs, torch.nn.functional.one_hot(img_labels).float().to(device))

            # update the model's weights
            loss.backward()
            optimizer.step()

            loss_aux += loss.item()

            print("EPOCH " + str(i + 1) + "/" + str(NUM_EPOCHS) + " (" + str(idx + 1) + "/" + str(len(train_loader)) + "): " + str(loss.item()))
        
        # ------------------------------------------------------------------------------------------------------
        # assess the performance of the model on the validation set
        # ------------------------------------------------------------------------------------------------------
        accuracy = 0.0
        model.eval()
        for img_batch, img_labels in validation_samples: # go through the previously selected validation samples

            # retrieve the ground truth labels and the model's predictions
            img_batch = img_batch.to(device)
            
            if(CNN_SIDE_TYPE == "inceptionv3"): outputs = model(img_batch)[0]
            else: outputs = model(img_batch)

            _, outputs = torch.max(outputs.data, 1)

            accuracy_aux = (outputs.cpu() == img_labels).float().mean()

            accuracy += accuracy_aux.detach().cpu().numpy()
            
        print("\nVALIDATION ACCURACY: %.2f\n" % (accuracy / len(validation_samples)))
        
        # ----------------------------------------------------------------------------------------------------------
        # update the plot
        # ---------------------------------------------------------------------------------------------------------- 
        if(plot is None):
            plot = Plot([
                [ [i + 1], [loss_aux / len(train_loader)], "-x", "r", "train loss" ],
                [ [i + 1], [accuracy / len(validation_samples)], "-x", "g", "validation accuracy" ]
            ], "Training", "Epochs", "Loss")

        else: plot.update_plot([(i + 1, loss_aux / len(train_loader)), (i + 1, accuracy / len(validation_samples))])

        # ----------------------------------------------------------------------------------
        # save the model trained up until this point
        # ----------------------------------------------------------------------------------
        if((((i + 1) % MODEL_SAVE_FREQ) == 0) or ((i + 1) == NUM_EPOCHS)):
            
            if(OVERWRITE_SAVED_MODELS): # if there is already a saved model, let's remove it
                remove_old_model(SAVE_DIR + "/models")

            # save the new model
            torch.save(model.state_dict(), SAVE_DIR + "/models/epoch" + str(i + 1) + ".pt")

    ################################################################################################
    # FINAL STEPS
    ################################################################################################
    # either keep the plot on the screen or save it
    if(KEEP_PLOT): plot.keep_plot()
    else: plot.save_plot(SAVE_DIR + "/training.png")

    plt.clf()

    # ---------------------------------------------------------------------------------------------
    # assess the performance of the model on the validation set
    # ---------------------------------------------------------------------------------------------
    accuracy = 0.0
    ground_truth = None
    predicted_labels = None
    model.eval()    
    for data_batch in test_loader: # go through the previously selected validation samples

        # read a batch of images
        img_batch, img_labels = data_batch
        img_batch = img_batch.to(device)
        
        outputs = model(img_batch)
        _, outputs = torch.max(outputs.data, 1)

        if(ground_truth is None): ground_truth = img_labels.detach().cpu().numpy()
        else: ground_truth = np.concatenate((ground_truth, img_labels.detach().cpu().numpy()))

        if(predicted_labels is None): predicted_labels = outputs.detach().cpu().numpy()
        else: predicted_labels = np.concatenate((predicted_labels, outputs.detach().cpu().numpy()))
        
        accuracy_aux = (outputs.cpu() == img_labels).float().mean()

        accuracy += accuracy_aux.detach().cpu().numpy()

    print("\nTEST ACCURACY: %.2f\n" % (accuracy / len(test_loader)))
        
    save_confusion_matrix(SAVE_DIR, ground_truth, predicted_labels)

    # ----------------------------------------------------------------------------------------------
    # if required, save a .csv file with the hyperparameters, the final loss value and test accuracy
    # ----------------------------------------------------------------------------------------------
    if(SAVE_CONFIGS_AND_RESULTS):
        with open(SAVE_DIR + "/configs_and_results.csv", "w") as file:
            
            # save the hyperparameters
            file.write("NUM EPOCHS," + str(NUM_EPOCHS) + "\n")
            file.write("LEARNING RATE," + str(LEARNING_RATE) + "\n")
            file.write("BETA 1," + str(BETA_1) + "\n")
            file.write("BETA 2," + str(BETA_2) + "\n")
            file.write("BATCH SIZE," + str(BATCH_SIZE) + "\n")
            file.write("IMAGE SIZE," + str(IMAGE_SIZE) + "\n")
            file.write("MODEL TYPE," + CNN_SIDE_TYPE + "\n")
            file.write("\nFINAL LOSS,%.2f" % (loss_aux / len(train_loader)))
            file.write("\nFINAL TEST ACCURACY,%.2f" % (accuracy / len(test_loader)))