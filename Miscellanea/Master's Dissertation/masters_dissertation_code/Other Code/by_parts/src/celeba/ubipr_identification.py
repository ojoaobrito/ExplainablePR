import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import numpy as np
from PIL import Image
from random import shuffle
from torch.utils.data import Dataset
from pickle import load

class UBIPr_Identification(Dataset): # custom Dataset class

    def __init__(self, img_path, split, transform, num_classes): # class constructor
        
        self.img_path = img_path + "/" + split
        self.transform = transform
        self.split = split
        self.num_classes = num_classes

        self.images = list(filter(lambda x : x[0] != ".", os.listdir(self.img_path)))

        with open("/".join(self.img_path.split("/")[:-2]) + "/ids_one_hot_encoding.pkl", "rb") as file:
            self.ids_one_hot = load(file)

        shuffle(self.images)

    def __len__(self): # auxiliary method, retrieves the length of the dataset
        return(len(self.images))

    def __getitem__(self, idx): # auxiliary method, retrieves a sample and its label
        
        label_aux = self.images[idx].split(".")[0].split("_")[1]
        label = self.ids_one_hot[label_aux]

        image = Image.open(self.img_path + "/" + self.images[idx])

        if(self.transform is not None):
            image = self.transform(image)

        return(image, label, np.zeros((1,)))