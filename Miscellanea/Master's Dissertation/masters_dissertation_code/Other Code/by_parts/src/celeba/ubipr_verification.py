import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import numpy as np
from PIL import Image
from random import shuffle
from torch.utils.data import Dataset

class UBIPr_Verification(Dataset): # custom Dataset class

    def __init__(self, img_path, split, transform): # class constructor
        
        self.img_path = img_path + "/" + split
        self.transform = transform
        self.split = split

        self.images = list(filter(lambda x : x[0] != ".", os.listdir(self.img_path + "/0"))) + list(filter(lambda x : x[0] != ".", os.listdir(self.img_path + "/1")))

        shuffle(self.images)

    def __len__(self): # auxiliary method, retrieves the length of the dataset
        return(len(self.images))

    def __getitem__(self, idx): # auxiliary method, retrieves a sample and its label
        
        label_aux = 1 if(self.images[idx].split("_+_")[0].split("_")[0] == self.images[idx].split("_+_")[1].split("_")[0]) else 0
        label = [0, 1] if(label_aux == 1) else [1, 0]

        image = Image.open(self.img_path + "/" + str(label_aux) + "/" + self.images[idx])

        if(self.transform is not None):
            image = self.transform(image)

        return(image, np.asarray(label), np.zeros((1,)))