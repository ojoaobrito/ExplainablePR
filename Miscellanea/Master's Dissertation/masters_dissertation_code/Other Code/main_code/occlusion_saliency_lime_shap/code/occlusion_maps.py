import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import numpy as np
from PIL import Image
import tensorflow as tf
import keras.backend as K
from keras.models import load_model
import matplotlib.pyplot as plt 
import matplotlib.image as mpimage
from copy import deepcopy
from keras.preprocessing import image
import pickle

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

np.set_printoptions(threshold = sys.maxsize)

###########################################################
# CONTROL VARIABLES
###########################################################
IMAGE_SIZE = 224
NUM_EPOCHS = 2000
IMAGE_TO_EXPLAIN = "temp.jpg"
MODEL_PATH = "vgg_face_2_side_by_side_densenet121_98,04.h5"

class OcclusionGenerator(object):
    
    def __init__(self, img, boxsize=10, step=10, prepocess=True):
        """ Initializations """
        self.img = img
        self.boxsize = boxsize
        self.step = step 
        self.i = int(IMAGE_SIZE / 4)
        self.j = 0
    

    def flow(self):
        """ Return a single occluded image and its location """
        if self.i + self.boxsize > self.img.shape[0]:
            self.i = int(IMAGE_SIZE / 4)
            self.j = 0
        
        retImg = np.copy(self.img)

        retImg[self.i:self.i+self.boxsize, self.j:self.j+self.boxsize] = 0.0 

        old_i = deepcopy(self.i) 
        old_j = deepcopy(self.j)
        
        # update indices
        self.j = self.j + self.step
        if self.j+self.boxsize>self.img.shape[1]: #reached end
            self.j = 0 # reset j
            self.i = min(self.i + self.step, IMAGE_SIZE - int(IMAGE_SIZE / 4) - self.boxsize) # go to next row
        
        return retImg, old_i, old_j

    def gen_minibatch(self, batchsize=10):
        """ Returns a minibatch of images of size <=batchsize """
        
        # list of occluded images
        occ_imlist = []
        locations = []
        for i in range(batchsize):
            occimg, i, j = self.flow()
            if occimg is not None:
                occ_imlist.append(occimg)
                locations.append([i,j])

        if len(occ_imlist)==0: # no data
            return None,None
        else:
            # convert list to numpy array and pre-process input (0 mean centering)
            return np.asarray(occ_imlist), locations

def post_process(heatmap, num_heatmaps):
    print(len(heatmap))

    # postprocessing
    total = heatmap[0]
    
    for val in heatmap[1:]:
        total = total + val
    
    for i in range(IMAGE_SIZE):
        for j in range(IMAGE_SIZE):
            if(num_heatmaps[(i, j)] != 0):
                total[i][j] = total[i][j] / num_heatmaps[(i, j)]
            else:
                total[i][j] = 1.0

    return(total)

def gen_heatmap(fileloc, boxsize, step, verbose=True, savetodisk=False):

    # load up image 
    img = mpimage.imread(fileloc)
    if verbose:
        plt.imshow(img); plt.axis("off")
        plt.show()

    img = img.astype(np.float64).copy()

    # classify image (w/o occlusions)
    model = load_model(MODEL_PATH)
    preds = model.predict(img.reshape((1, 224, 224, 3)))
    correct_class_index = np.argmax(preds[0])

    correct_class_label = ["I", "G"][correct_class_index]

    # generate occluded images and location of mask
    occ = OcclusionGenerator(img, boxsize, step, True)

    # scores of occluded image
    heatmap = []
    index = 0
    epoch = 1
    num_heatmaps = {}

    for i in range(IMAGE_SIZE):
        for j in range(IMAGE_SIZE):
            num_heatmaps[(i, j)] = 0

    while(epoch <= NUM_EPOCHS):

        print("EPOCH " + str(epoch))

        # get minibatch of data
        x, locations = occ.gen_minibatch(batchsize=1)
        
        if(x is not None):

            #predict 
            op = model.predict(x)

            #unpack prediction values 
            for i in range(x.shape[0]):
                score = op[i][correct_class_index]
                r,c = locations[i]
                scoremap = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
                scoremap[r : r+occ.boxsize, c : c+occ.boxsize] = score
                
                for j in range(IMAGE_SIZE):
                    for k in range(IMAGE_SIZE):
                        if((j >= r and j < (r+occ.boxsize)) and (k >= c and k < (c+occ.boxsize))):
                            num_heatmaps[(j, k)] += 1

                heatmap.append(scoremap)

            epoch += 1

            if verbose:
                print("..minibatch completed")
        else:
            occ = OcclusionGenerator(img, boxsize, step, True)

    if savetodisk:
        f = open("heatmap", "wb")
        pickle.dump(heatmap, f)
        f.close()

    return heatmap, correct_class_index, correct_class_label, num_heatmaps

if(__name__ == "__main__"):

    heatmapList, index, label, num_heatmaps = gen_heatmap(IMAGE_TO_EXPLAIN, 30, 3, False)
    processed = post_process(heatmapList, num_heatmaps)
    print(processed)

    img = mpimage.imread(IMAGE_TO_EXPLAIN)
    plt.subplot(121)
    plt.imshow(img); plt.axis("off")
    plt.title("Pred: " + label)
    plt.subplot(122)
    plt.imshow(processed, interpolation = "nearest", origin = "upper", cmap = "Greys"); plt.axis("off")
    plt.colorbar()
    plt.savefig("aux.png", dpi = 250)

    """img = img = mpimage.imread("temp.jpg")
    occ = OcclusionGenerator(img, 40, 15, False)
    occList = []
    plt.rcParams["figure.figsize"] = (10, 10)
    for i in range(100):
        occList.append(occ.flow()[0])
    for i in range(30):
        plt.subplot(5,6,i + 1)
        plt.imshow(occList[np.random.randint(len(occList))]); plt.axis("off")
    plt.show()"""