import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import time
import datetime
from natsort import natsorted
from shutil import rmtree, copyfile

####################################################################
# CONTROL VARIABLES
####################################################################
SAVE_DIR = "evaluation_results/"
CONFIGURATIONS = [ # NOTE: choose any of the following combinations:

    #"occlusion_map",
    "saliency_map",
    "lime",
    "shap"
]

def check_for_errors(r_value): # auxiliary function, checks for errors when running the main scripts

    if(r_value != 0):
        print("ERROR FOUND, STOPPING NOW...")
        sys.exit()

if(__name__ == "__main__"):

    if(CONFIGURATIONS == []): 
        print("\nNO CONFIGURATIONS FOUND, LEAVING NOW...\n")
        sys.exit()

    directory = natsorted(list(filter(lambda x : x[0] != ".", os.listdir("performance_evaluation_pairs"))))
    
    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d_%H_%M_%S")

    if(not (os.path.exists(SAVE_DIR))): os.makedirs(SAVE_DIR)

    os.makedirs(SAVE_DIR + timestamp)

    ####################################################################################################
    # EXPLAIN SEVERAL PAIRS TO EVALUATE THE METHOD'S PERFORMANCE (USING THE CONFIGURATIONS CHOSEN ABOVE)
    ####################################################################################################
    for idx, i in enumerate(CONFIGURATIONS):
        print("CONFIG " + str(idx + 1) + "/" + str(len(CONFIGURATIONS)))

        for jdx, j in enumerate(directory):

            #if("G" in j): continue
            #if((j != "31_I_L") and (j != "43_I_L") and (j != "102_I_R") and (j != "121_I_R")): continue

            # -------------------------------------------------------------------------------
            # move the pair's images to the "images" folder
            # -------------------------------------------------------------------------------
            rmtree("../images")
            os.makedirs("../images")
            copyfile("performance_evaluation_pairs/" + j + "/1.jpg", "../images/1.jpg")
            copyfile("performance_evaluation_pairs/" + j + "/2.jpg", "../images/2.jpg")
            
            # ------------------------------------------------------
            # call the "explain_pair.py" script to explain this pair
            # ------------------------------------------------------
            os.chdir("..")
            
            r_value = os.system("python3 explain_pair.py " + i)
            check_for_errors(r_value)

            os.chdir("code")

            # --------------------------------------------------------------------------------
            # move the explanation to the performance evaluation directory
            # --------------------------------------------------------------------------------
            os.rename("../explanation.png", SAVE_DIR + timestamp + "/" + i + "_" + j + ".png")
