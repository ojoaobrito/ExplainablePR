import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import cv2
import torch
import numpy as np
from PIL import Image
from random import shuffle
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class DepthWiseDataset(Dataset): # custom Dataset class (useful when loading 6 channels of data - i.e. 2 images on top of each other)

    def __init__(self, two_classes, img_path, latent_path, transform): # class constructor
        
        self.img_path = img_path
        self.latent_path = latent_path
        self.transform = transform
        self.two_classes = two_classes

        # determine if we'll have to store the latent codes
        if(self.latent_path is None):
            self.latent_codes = None
            
        else:
            self.latent_codes = list(filter(lambda x : x[0] != ".", os.listdir(self.latent_path + "0/")))

        # determine if we need the class information        
        if(self.two_classes):
            self.images = list(filter(lambda x : x[0] != ".", os.listdir(self.img_path + "0/"))) + list(filter(lambda x : x[0] != ".", os.listdir(self.img_path + "1/"))) 
            
        else:
            self.images = list(filter(lambda x : x[0] != ".", os.listdir(self.img_path + "0/")))

        shuffle(self.images)

    def __len__(self): # auxiliary method, retrieves the length of the dataset
        return(len(self.images))

    def __getitem__(self, idx): # auxiliary method, retrieves a sample and its label

        if(self.two_classes):
            
            ids = (self.images[idx].split("_+_")[0].split("_")[0], self.images[idx].split("_+_")[1].split("_")[0])

            label = 1 if(ids[0] == ids[1]) else 0
            image = np.load(self.img_path + str(label) + "/" + self.images[idx])

            final_imgs = torch.cat((self.transform(Image.fromarray(image[:, :, :3].astype(np.uint8))), self.transform(Image.fromarray(image[:, :, 3:].astype(np.uint8)))), 0)
            
            sample = (final_imgs, label)

        else:
            image = np.load(self.img_path + "0/" + self.images[idx])
            
            final_imgs = torch.cat((self.transform(Image.fromarray(image[:, :, :3].astype(np.uint8))), self.transform(Image.fromarray(image[:, :, 3:].astype(np.uint8)))), 0)
            
            if(self.latent_path is None): 
                sample = (final_imgs, 0) # we define the label as 0, but, in practice, it is never used

            else:
                sample = (final_imgs, np.load(self.latent_path + "0/" + self.latent_codes[self.latent_codes.index(self.images[idx])])[0])

        return(sample)

def prepare_data(two_classes, img_path, latent_path, image_size, batch_size): # auxiliary function, prepares the data

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train = DepthWiseDataset(two_classes = two_classes, img_path = img_path + "train/", latent_path = None if(latent_path is None) else (latent_path + "train/"), transform = transform)
    #validation = DepthWiseDataset(two_classes = two_classes, img_path = img_path + "validation/", latent_path = None if(latent_path is None) else (latent_path + "validation/"), transform = transform)
    test = DepthWiseDataset(two_classes = two_classes, img_path = img_path + "test/", latent_path = None if(latent_path is None) else (latent_path + "test/"), transform = transform)

    train_loader = DataLoader(train, batch_size = batch_size, shuffle = True, num_workers = 4, drop_last = True)
    #validation_loader = DataLoader(validation, batch_size = batch_size, shuffle = True, num_workers = 4, drop_last = True)
    test_loader = DataLoader(test, batch_size = batch_size, shuffle = True, num_workers = 4, drop_last = True)

    return(train_loader, None, test_loader)

def find_centre(img_path): # auxiliary function, determines the iris' centre of the given image

        img = cv2.imread(img_path)

        thresh = 132
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
        edges = cv2.Canny(blur, thresh, thresh * 2)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        cv2.drawContours(img, contours, -1, (0, 255, 0), -1)

        M = cv2.moments(cnt)
        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])

        return((x, y))

def find_closest_centroid(iris_centre, centroids): # auxiliary function, determines closest centroid to the given iris centre

    distances = [
        np.math.sqrt((centroids[0][0] - iris_centre[0])**2 + (centroids[0][1] - iris_centre[1])**2),
        np.math.sqrt((centroids[1][0] - iris_centre[0])**2 + (centroids[1][1] - iris_centre[1])**2),
        np.math.sqrt((centroids[2][0] - iris_centre[0])**2 + (centroids[2][1] - iris_centre[1])**2)
    ]

    if(np.argmin(distances) == 0): return("L")
    if(np.argmin(distances) == 1): return("C")
    return("R")