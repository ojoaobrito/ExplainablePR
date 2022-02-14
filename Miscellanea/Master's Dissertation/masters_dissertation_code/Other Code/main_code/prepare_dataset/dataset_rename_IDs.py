import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

################################################
# CONTROL VARIABLES
################################################
SOURCE_DIR = "../../dataset/annotations_merged/"

ID_MAPS = []

def get_ID_maps(current): # auxiliary function, prepares a list with the new IDs

    global ID_MAPS

    if(os.path.isdir(current)): # it's a directory, let's explore it
        for i in list(filter(lambda x : x[0] != ".", os.listdir(current))):
            get_ID_maps(current + "/" + i)

    else: # it's a file, let's do something with it
        if(not ((current.split("/")[-1].split("_")[0]) in ID_MAPS)): 
            ID_MAPS.append((current.split("/")[-1].split("_")[0]))

def rename_IDs(current): # auxiliary function, replaces the ID present in "current" with the correct version in "ID_MAPS"

    if(os.path.isdir(current)): # it's a directory, let's explore it
        for i in list(filter(lambda x : x[0] != ".", os.listdir(current))):
            rename_IDs(current + "/" + i)

    else: # it's a file, let's do something with it
        os.rename(current, current.replace(current.split("/")[-1].split("_")[0], "C" + str(ID_MAPS.index((current.split("/")[-1]).split("_")[0]) + 1)).replace(".jpg", "+IDr.jpg"))

def final_step(current): # auxiliary function, removes the "+IDr" tag on every image

    if(os.path.isdir(current)): # it's a directory, let's explore it
        for i in list(filter(lambda x : x[0] != ".", os.listdir(current))):
            final_step(current + "/" + i)

    else: # it's a file, let's do something with it
        if("+I" in current and not os.path.isdir(current)):
            new_name = current.split("+I")[0]

            if("+IDr" in current): new_name = new_name.replace("+IDr", "")

            if("+IDf" in current): new_name += "_2.jpg"
            else: new_name += ".jpg"
        
            os.rename(current, new_name)

if(__name__ == "__main__"):

    get_ID_maps(SOURCE_DIR)
    ID_MAPS.sort(key = lambda x : int(x.split("C")[1]))
    rename_IDs(SOURCE_DIR)

    final_step(SOURCE_DIR)