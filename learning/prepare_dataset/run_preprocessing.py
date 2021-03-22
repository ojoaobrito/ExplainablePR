import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

########################################################
# CONTROL VARIABLES
########################################################
GENUINE_OR_IMPOSTOR = "G" # either "G", "I" or "both"
CREATE_CSV_FILE = True
TRAIN_GAN_SINGLE_IMAGES = False
TRAIN_CNN_SIDE = False
TRAIN_GAN_PAIRS = False
TRAIN_CNN_DISTINGUISH_PAIRS = False
TRAIN_CNN_LIME_SHAP = False
TRAIN_CNN_ID_IDENTIFICATION = False
IMAGE_SIZE = 128

def check_for_errors(r_value): # auxiliary function, checks for errors when running the main scripts

    if(r_value != 0):
        print("ERROR FOUND, STOPPING NOW...")
        sys.exit()

if(__name__ == "__main__"):

    #######################################################################################################
    # RUN EVERY STEP OF THE DATA PREPROCESSING PHASE
    #######################################################################################################
    # move every image to a common folder
    print("\nMOVING THE IMAGES TO A COMMON FOLDER...")
    r_value = os.system("python3 dataset_one_folder.py")
    check_for_errors(r_value)
    print("DONE MOVING THE IMAGES TO A COMMON FOLDER!")

    # resize every image to a common size
    print("\nRESIZING THE IMAGES...")
    r_value = os.system("python3 dataset_resize.py " + str(IMAGE_SIZE))
    check_for_errors(r_value)
    print("DONE RESIZING THE IMAGES!")

    # make sure every ID has the same number of images
    print("\nENSURING FAIR PLAY...")
    r_value = os.system("python3 dataset_fair_play.py")
    check_for_errors(r_value)
    print("DONE ENSURING FAIR PLAY!")

    # resize every image to a common size
    print("\nRESIZING THE IMAGES...")
    r_value = os.system("python3 dataset_resize.py " + str(IMAGE_SIZE))
    check_for_errors(r_value)
    print("DONE RESIZING THE IMAGES!")

    # group the images by ID
    print("\nGROUPING THE IMAGES BY ID...")
    r_value = os.system("python3 dataset_group_by_id.py")
    check_for_errors(r_value)
    print("DONE GROUPING THE IMAGES BY ID!")
    
    # reserve some of the IDs just for testing
    print("\nRESERVING SOME IDS JUST FOR TESTING...")
    r_value = os.system("python3 dataset_reserve_test_IDs.py")
    check_for_errors(r_value)
    print("DONE RESERVING SOME IDS JUST FOR TESTING!")

    # move every image to a common folder
    print("\nMOVING THE IMAGES TO A COMMON FOLDER...")
    r_value = os.system("python3 dataset_one_folder.py ../../dataset/dataset_images_per_id")
    check_for_errors(r_value)
    print("DONE MOVING THE IMAGES TO A COMMON FOLDER!")

    if(CREATE_CSV_FILE):

        # if required, create a .csv file with the annotations
        print("\nCREATING A CSV FILE...")
        r_value = os.system("python3 dataset_to_csv_file.py")
        check_for_errors(r_value)
        print("DONE CREATING A CSV FILE!")

    if(TRAIN_GAN_SINGLE_IMAGES):

        # move the images to a structured folder
        print("\nMOVING TO A STRUCTURED FOLDER...")
        r_value = os.system("python3 dataset_move_to_folders.py gan 0.8 0.1 0.1")
        check_for_errors(r_value)
        print("DONE MOVING TO A STRUCTURED FOLDER!")

    if(TRAIN_CNN_SIDE):

        # move the images to a structured folder
        print("\nMOVING TO A STRUCTURED FOLDER...")
        r_value = os.system("python3 dataset_move_to_folders.py cnn_side")
        check_for_errors(r_value)
        print("DONE MOVING TO A STRUCTURED FOLDER!")

    if(TRAIN_GAN_PAIRS):

        # make image pairs and move them to a structured folder
        print("\nMAKING PAIRS AND MOVING TO A STRUCTURED FOLDER...")
        r_value = os.system("python3 dataset_make_pairs_and_move_to_folders.py gan " + GENUINE_OR_IMPOSTOR)
        check_for_errors(r_value)
        print("DONE MAKING PAIRS AND MOVING TO A STRUCTURED FOLDER!")

    if(TRAIN_CNN_DISTINGUISH_PAIRS):

        # make image pairs and move them to a structured folder
        print("\nMAKING PAIRS AND MOVING TO A STRUCTURED FOLDER...")
        r_value = os.system("python3 dataset_make_pairs_and_move_to_folders.py cnn_distinguish_pairs both")
        check_for_errors(r_value)
        print("DONE MAKING PAIRS AND MOVING TO A STRUCTURED FOLDER!")

    if(TRAIN_CNN_LIME_SHAP):

        # make image pairs and move them to a structured folder
        print("\nMAKING PAIRS AND MOVING TO A STRUCTURED FOLDER...")
        r_value = os.system("python3 dataset_make_pairs_and_move_to_folders.py cnn_lime_shap both")
        check_for_errors(r_value)
        print("DONE MAKING PAIRS AND MOVING TO A STRUCTURED FOLDER!")

    if(TRAIN_CNN_ID_IDENTIFICATION):

        # move the images to a structured folder
        print("\nMOVING THE IMAGES TO A STRUCTURED FOLDER...")
        r_value = os.system("python3 dataset_move_single_images_to_folders.py")
        check_for_errors(r_value)
        print("DONE MOVING THE IMAGES TO A STRUCTURED FOLDER!")

    print("")