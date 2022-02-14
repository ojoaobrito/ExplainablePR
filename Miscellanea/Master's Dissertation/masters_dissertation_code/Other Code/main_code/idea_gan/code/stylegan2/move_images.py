import os, sys
from shutil import rmtree, move

total_images_aux = [list(filter(lambda x : x[0] != ".", os.listdir("/media/socialab/HARD disc/synthetic_dataset_G/images_revised_" + str(i + 1)))) for i in range(6)]

'''for i in range(6):
    #if(os.path.exists("/media/socialab/HARD disc/synthetic_dataset_G/images_" + str(i + 1))): 
        #rmtree("/media/socialab/HARD disc/synthetic_dataset_G/images_" + str(i + 1))
    os.makedirs("/media/socialab/HARD disc/synthetic_dataset_G/images_" + str(i + 1))

for i in range(6):
    #if(os.path.exists("synthetic_dataset_G/segmentation_maps_" + str(i + 1))): 
        #rmtree("synthetic_dataset_G/segmentation_maps_" + str(i + 1))
    os.makedirs("synthetic_dataset_G/segmentation_maps_" + str(i + 1))'''

total_images = []
for idx, i in enumerate(total_images_aux):
    print(str(idx + 1) + "/" + str(len(total_images_aux)))
    for j in i: 
        total_images.append("/media/socialab/HARD disc/synthetic_dataset_G/images_revised_" + str(idx + 1) + "/" + j)

amount = len(total_images) // 5
count = 1
index = 0

print(count)

while(index < len(total_images)):
    
    if((index % (amount) == 0) and (index != 0)): 
        count += 1
        print(count)

    image_name = total_images[index].split("/")[-1]

    # move the image
    new_name = str(count) + "_" + image_name[2:]
    os.rename(total_images[index], "/media/socialab/HARD disc/synthetic_dataset_G/images_" + str(count) + "/" + new_name)

    # move the corresponding segmentation maps
    move("synthetic_dataset_G/segmentation_maps_" + image_name[0] + "_revised/" + image_name.replace(".npy", ""), "synthetic_dataset_G/segmentation_maps_" + str(count) + "/" + new_name.replace(".npy", ""))

    index += 1