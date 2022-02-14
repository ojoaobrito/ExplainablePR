import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

from PIL import Image, ImageOps

##############################################################################################################################
# CONTROL VARIABLES
##############################################################################################################################
SOURCE_DIR = "../../dataset/annotations_merged/"
REPEATED_IDS_DIR = "old_problems/repeated_ids"
FEATURES = ["iris/color", "eyebrow/distribution", "eyebrow/shape", "skin/color", "skin/texture", "skin/spots", "eyelid/shape"]

def main_function_restored(current): # main function, removes the given image if it has a space in its name

    if(os.path.isdir(current)): # it's a directory, let's explore it
        for i in list(filter(lambda x : x[0] != ".", os.listdir(current))):
            main_function_restored(current + "/" + i)

    else: # it's a file, let's do something with it

        # check if the image is not a restored version
        if(" " in current): 
            os.remove(current)
            print("Image removed! (" + current + ")")

def main_function_check(current): # main function, applies the "check_annotations()" function to every image

    if(os.path.isdir(current)): # it's a directory, let's explore it
        for i in list(filter(lambda x : x[0] != ".", os.listdir(current))):
            main_function_check(current + "/" + i)

    else: # it's a file, let's do something with it

        # check the annotations
        is_ok = check_annotations(current.split("/")[-1], feature_index = 0)

        if(not is_ok): 
            os.remove(current)
            print("Image removed! (" + current + ")")

def main_function_pairs(current): # main function, applies the "add_pair()" function to every image

    if(os.path.isdir(current)): # it's a directory, let's explore it
        for i in list(filter(lambda x : x[0] != ".", os.listdir(current))):
            main_function_pairs(current + "/" + i)

    else: # it's a file, let's do something with it

        # check the presence of a pair
        add_pair(current)

def main_function_IDs(current): # main function, applies the "fix_ID()" function to every image

    if(os.path.isdir(current)): # it's a directory, let's explore it
        for i in list(filter(lambda x : x[0] != ".", os.listdir(current))):
            main_function_IDs(current + "/" + i)

    else: # it's a file, let's do something with it

        # check (and potentially fix) the ID
        fix_ID(current)

def check_annotations(image, feature_index): # auxiliary function, checks if the given image is completely annotated

    try: # check if the images has been annotated with regards to the given feature
        if(feature_index == (len(FEATURES))): return(True)

        for i in list(filter(lambda x : x[0] != ".", os.listdir(SOURCE_DIR + FEATURES[feature_index]))):
            if(image in os.listdir(SOURCE_DIR + FEATURES[feature_index] + "/" + i)): raise Exception
        
        # the image is not present in any folder of this feature (i.e. not annotated)
        return(False)

    except Exception: return(check_annotations(image, feature_index + 1)) # move to the next feature

def add_pair(image): # auxiliary function, adds a pair (either "L" or "R") to the given image, if needed

    # check the annotations
    try:
        side = "L" if("L" in image.split("/")[-1]) else "R"
        other_side = "L" if(side == "R") else "R"
        pair = image.replace(side, other_side)
        is_ok = check_annotations(pair.split("/")[-1], feature_index = 0)

        if(is_ok): # the pair is okay, our work here is done
            return

        # add a pair to this image
        for i in FEATURES:
            for j in list(filter(lambda x : x[0] != ".", os.listdir(SOURCE_DIR + i))):
                if((image.split("/")[-1]) in os.listdir(SOURCE_DIR + i + "/" + j)):
                    img = Image.open(image)
                    img = ImageOps.mirror(img)
                    img.save(SOURCE_DIR + i + "/" + j + "/" + image.split("/")[-1].replace(side, other_side))

    except: return

def fix_ID(image): # auxiliary function, fixes the poorly attributed IDs (different IDs that are, in fact, the same person)

    id_maps = list(map(lambda x : x.split(".")[0], list(filter(lambda x : x[0] != ".", os.listdir(REPEATED_IDS_DIR)))))
    
    for i in id_maps:
        
        # fix the ID
        if(i.split("_")[1] == image.split("/")[-1].split("_")[0]): 
            
            os.rename(image, image.replace(image.split("/")[-1].split("_")[0], i.split("_")[0]).replace(".jpg", "+IDf.jpg"))
            break

def aux(current):

    if(os.path.isdir(current)): # it's a directory, let's explore it
        for i in list(filter(lambda x : x[0] != ".", os.listdir(current))):
            aux(current + "/" + i)

    else: # it's a file, let's do something with it

        if("+HF" in current): os.remove(current)

if(__name__ == "__main__"):

    main_function_restored(SOURCE_DIR)
    main_function_check(SOURCE_DIR)
    main_function_pairs(SOURCE_DIR)
    main_function_IDs(SOURCE_DIR)

    aux(SOURCE_DIR)