import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import time
import datetime
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.engine import  Model
from keras_vggface.vggface import VGGFace
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense, GlobalAveragePooling2D, Dropout, BatchNormalization

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

np.set_printoptions(threshold = sys.maxsize)

#######################################################################################################################################################################################################################
# CONTROL VARIABLES
#######################################################################################################################################################################################################################
MODE = "pairs" # "pairs" / "single"
WEIGHTS = "random" # "random" / "vggface"
TRAIN_FROM_SCRATCH_MODEL = "densenet121" # "resnet50" / "densenet121"
LEARNING_RATE = 2e-4
NUM_CLASSES = 246 if(MODE == "single") else 2
BATCH_SIZE = 32
NUM_EPOCHS = 15
IMAGE_SIZE = 224
DROPOUT = 0.8
TOT_TRAINING_IMAGES = len(os.listdir("../../../dataset/data/train/0")) if(MODE == "single") else (len(os.listdir("../../../dataset/data/train/0")) + len(os.listdir("../../../dataset/data/train/1")))
TOT_VALIDATION_IMAGES = len(os.listdir("../../../dataset/data/validation/0")) if(MODE == "single") else (len(os.listdir("../../../dataset/data/validation/0")) + len(os.listdir("../../../dataset/data/validation/1")))
CLASS_MODE = "categorical"
SAVE_DIR = "outputs/"

if(__name__ == "__main__"):

    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d_%H_%M_%S")

    if(not (os.path.exists(SAVE_DIR))): os.makedirs(SAVE_DIR)

    os.makedirs(SAVE_DIR + timestamp)

    #######################################################################################################################################################################################
    # LOAD AND PREPARE THE DATASET
    #######################################################################################################################################################################################
    datagen = ImageDataGenerator()

    # training dataset
    train_it = datagen.flow_from_directory("../../../dataset/data/train/", class_mode = CLASS_MODE, target_size = (IMAGE_SIZE, IMAGE_SIZE), color_mode = "rgb", batch_size = BATCH_SIZE)
    # validation dataset
    val_it = datagen.flow_from_directory("../../../dataset/data/validation/", class_mode = CLASS_MODE, target_size = (IMAGE_SIZE, IMAGE_SIZE), color_mode = "rgb", batch_size = BATCH_SIZE)
    # test dataset
    test_it = datagen.flow_from_directory("../../../dataset/data/test/", class_mode = CLASS_MODE, target_size = (IMAGE_SIZE, IMAGE_SIZE), color_mode = "rgb", batch_size = BATCH_SIZE)

    ###################################################################################################################################
    # SAVE THE KEY OF IDS TO A SEPARATE FILE (THE GENERATOR RETRIEVES THE IDS IN THE WRONG ORDER, SO WE HAVE TO TAKE THAT INTO ACCOUNT)
    ###################################################################################################################################
    if(MODE == "single"):
        it_order = train_it.filenames
        ids_key = []

        for i in it_order:
            aux = int((i.split("/")[0]).split("C")[1])
            if(not (aux in ids_key)): ids_key.append(aux)

        np.save("ids_key.npy", np.asarray(ids_key))

    ####################################################################################################################
    # PREPARE THE MODEL
    ####################################################################################################################
    if(WEIGHTS == "vggface"):
        resnet_model = VGGFace(model = "resnet50", include_top = False, input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3))
        x = resnet_model.get_layer("avg_pool").output
        x = Flatten(name = "flatten")(x)
        out = Dense(NUM_CLASSES, activation = "softmax", name = "classifier")(x)
        model = Model(resnet_model.input, out)

    else:
        # use a ResNet50 model
        if(TRAIN_FROM_SCRATCH_MODEL == "resnet50"):
            resnet_model = ResNet50(weights = None, include_top = False, input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3))
            x = resnet_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dropout(DROPOUT)(x)
            out = Dense(NUM_CLASSES, activation = "softmax")(x)
            model = Model(inputs = resnet_model.input, outputs = out)

        # use a Densenet121 model
        else:
            densenet_model = DenseNet121(weights = None, include_top = False, input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3))
            x = densenet_model.output
            x = GlobalAveragePooling2D()(x)
            x = BatchNormalization()(x)
            x = Dense(1024, activation = "relu")(x) 
            x = Dropout(DROPOUT)(x)
            x = Dense(512, activation = "relu")(x) 
            x = BatchNormalization()(x)
            x = Dropout(DROPOUT)(x)
            out = Dense(NUM_CLASSES, activation = "softmax")(x)
            model = Model(inputs = densenet_model.input, outputs = out)

    #################################################################################################################################################################################################################
    # COMPILE, FINE-TUNE/TRAIN AND EVALUATE THE MODEL
    #################################################################################################################################################################################################################
    model.compile(loss = CLASS_MODE + "_crossentropy", optimizer = "adam", metrics = ["accuracy"])
    K.set_value(model.optimizer.lr, LEARNING_RATE)
    model.fit_generator(train_it, shuffle = True, epochs = NUM_EPOCHS, steps_per_epoch = int(TOT_TRAINING_IMAGES / BATCH_SIZE), validation_data = val_it, validation_steps = int(TOT_VALIDATION_IMAGES / BATCH_SIZE))

    scores = model.evaluate_generator(test_it, steps = 24)

    print("Accuracy: %.2f%%" % (scores[1] * 100))

    ################################################################################################################################################################
    # SAVE THE CURRENT SESSION AND THE FINAL MODEL
    ################################################################################################################################################################
    saver = tf.train.Saver()
    sess = K.get_session()
    saver.save(sess, SAVE_DIR + timestamp + "/session.ckpt")

    if(WEIGHTS == "vggface"): model.save(SAVE_DIR + timestamp + "/vgg_face_2_side_by_side_resnet50_" + str(round(scores[1] * 100, 2)).replace(".", ",") + ".h5")
    else: model.save(SAVE_DIR + timestamp + "/vgg_face_2_side_by_side_" + TRAIN_FROM_SCRATCH_MODEL + "_" + str(round(scores[1] * 100, 2)).replace(".", ",") + ".h5")
    print("Saved model!")