import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import time
import datetime
from natsort import natsorted
from shutil import rmtree, copyfile

#######################################################################################
# CONTROL VARIABLES
#######################################################################################
SAVE_DIR = "evaluation_results/"
CONFIGURATIONS = [ # NOTE: choose any of the following combinations:

    # NOTE: without considering the segmentation information
    #(15, "elementwise_comparison", "IoU", False, "first"),

    # NOTE: considering the segmentation information

    # general results
    (100, "elementwise_comparison", "IoU", True, "first"),
    #(50, "elementwise_comparison", "IoU", True, "first"),

    # ablation study (value of "K")
    #(1, "elementwise_comparison", "IoU", True, "first"),
    #(5, "elementwise_comparison", "IoU", True, "first"),
    #(15, "elementwise_comparison", "IoU", True, "first"),
    #(50, "elementwise_comparison", "IoU", True, "first"),
    #(100, "elementwise_comparison", "IoU", True, "first"),
    #(150, "elementwise_comparison", "IoU", True, "first"),

    # ablation study (dataset length)
    #(15, "elementwise_comparison", "IoU", True, "first"), # 5% of the original dataset
    #(15, "elementwise_comparison", "IoU", True, "first"), # 10% of the original dataset
    #(15, "elementwise_comparison", "IoU", True, "first"), # 25% of the original dataset
    #(15, "elementwise_comparison", "IoU", True, "first"), # 50% of the original dataset
    #(15, "elementwise_comparison", "IoU", True, "first"), # 75% of the original dataset
    #(15, "elementwise_comparison", "IoU", True, "first") # 100% of the original dataset
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

    #########################################################################################################################
    # EXPLAIN SEVERAL PAIRS TO EVALUATE THE METHOD'S PERFORMANCE (USING THE CONFIGURATIONS CHOSEN ABOVE)
    #########################################################################################################################
    for idx, i in enumerate(CONFIGURATIONS):
        print("CONFIG " + str(idx + 1) + "/" + str(len(CONFIGURATIONS)))

        for jdx, j in enumerate(directory):

            #if(idx >= 0 and idx <= 8 and j != "139_I_R"): continue
            #if(idx >= 9 and j != "140_I_R"): continue
            
            '''if(idx >= 0): 
                dataset_proportions = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
                r_value = os.system("python3 ablation_study_dataset_length.py " + str(dataset_proportions[idx]))
                check_for_errors(r_value)'''
            
            config_dir_name = SAVE_DIR + timestamp + "/" + str(i[0]) + "_" + i[1] + "_" + i[2] + "_" + str(i[3]) + "_" + i[4]
            #if(idx >= 0): config_dir_name += "_" + str(dataset_proportions[idx])

            if(not os.path.exists(config_dir_name)): os.makedirs(config_dir_name)

            # -------------------------------------------------------------------------------
            # move the pair's images to the "images" folder
            # -------------------------------------------------------------------------------
            rmtree("../images")
            os.makedirs("../images")
            copyfile("performance_evaluation_pairs/" + j + "/1.jpg", "../images/1.jpg")
            copyfile("performance_evaluation_pairs/" + j + "/2.jpg", "../images/2.jpg")
            
            with open("pair_name.txt", "w") as file:
                file.write(j)
            
            # -----------------------------------------------------------------------------------------
            # call the "explain_pair.py" script to explain this pair
            # -----------------------------------------------------------------------------------------
            os.chdir("..")
            
            r_value = os.system("python3 explain_pair.py " + " ".join(list(map(lambda x : str(x), i))))
            check_for_errors(r_value)

            os.chdir("code")

            # -----------------------------------------------------------------
            # move the explanation to the performance evaluation directory
            # -----------------------------------------------------------------
            os.rename("../explanation.png", config_dir_name + "/" + j + ".png")