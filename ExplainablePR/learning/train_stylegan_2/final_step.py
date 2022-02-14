import os, sys
from natsort import natsorted
from shutil import move

total_images_aux = [list(filter(lambda x : x[0] != ".", os.listdir("/media/socialab/HARD disc/synthetic_dataset_G/temp_images_" + str(i + 1)))) for i in range(4)]

for i in range(5):
    if(not os.path.exists("/media/socialab/HARD disc/synthetic_dataset_G/images_" + str(i + 1))): 
        os.makedirs("/media/socialab/HARD disc/synthetic_dataset_G/images_" + str(i + 1))

total_images = []
for idx, i in enumerate(total_images_aux):
    print(str(idx + 1) + "/4")
    for j in i: 
        total_images.append("/media/socialab/HARD disc/synthetic_dataset_G/old_images_" + str(idx + 1) + "/" + j)

count = 1
index = 0

while(index < len(total_images)):
    #print(index)
    if((index % 80000) == 0): count += 1

    new_name = str(count) + "_" + "".join(total_images[index].split("/")[-1][2:])

    #os.rename(total_images[index], "/media/socialab/HARD disc/synthetic_dataset_G/images_" + str(count) + "/" + new_name)

    corresponding_seg_maps = "segmentation_maps_" + total_images[index].split("/")[-1][0] + "/" + total_images[index].split("/")[-1].replace(".npy", "")
    new_corresponding_seg_maps = "segmentation_maps_" + str(count) + "/" + str(count) + "_" + "".join(total_images[index].split("/")[-1][2:]).replace(".npy", "")
    
    print(new_name)
    print("")
    print(corresponding_seg_maps)
    print(new_corresponding_seg_maps)
    #move(corresponding_seg_maps, new_corresponding_seg_maps)

    index += 1
    sys.exit()