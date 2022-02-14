import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import numpy as np
from PIL import Image
import tensorflow as tf
import keras.backend as K
from keras.models import load_model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

np.set_printoptions(threshold = sys.maxsize)

###########################################################
# CONTROL VARIABLES
###########################################################
IMAGE_SIZE = 224
IMAGE_TO_EXPLAIN = "temp.jpg"
MODEL_PATH = "vgg_face_2_side_by_side_densenet121_98,04.h5"

if(__name__ == "__main__"):

    ###################################
    # RELOAD THE SESSION AND THE MODEL
    ###################################
    model = load_model(MODEL_PATH)

    '''
    saver = tf.train.Saver()
    sess = K.get_session()
    saver.restore(sess, "session.ckpt")
    '''

    ##############################################################################################
    # MAKE A PREDICTION
    ##############################################################################################
    test = np.asarray(Image.open(IMAGE_TO_EXPLAIN))
    prediction = model.predict(test.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3)))[0]

    print("Predicted class: " + str(np.argmax(prediction)) + " (" + str(np.max(prediction)) + ")")