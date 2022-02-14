import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

from natsort import natsorted

HEADER = ["iris_color_0-blue", "iris_color_1-hazel", "iris_color_2-brown", "iris_color_3-dark-brown", "eyebrow_distribution_0-sparse", "eyebrow_distribution_1-average", "eyebrow_distribution_2-dense",
            "eyebrow_shape_0-angled", "eyebrow_shape_1-curved", "eyebrow_shape_2-straight", "skin_color_0-light", "skin_color_1-average", "skin_color_2-dark", "skin_texture_0-average", "skin_texture_1-middle-aged", "skin_texture_2-aged", 
            "skin_spots_0-0", "skin_spots_1-1", "skin_spots_2-2+", "eyelid_shape_0-exposed", "eyelid_shape_1-covered"]

HEADER_ENCODING = {"iris_color_0-blue": "1,0,0,0", "iris_color_1-hazel": "0,1,0,0", "iris_color_2-brown": "0,0,1,0", "iris_color_3-dark-brown": "0,0,0,1", "eyebrow_distribution_0-sparse": "1,0,0", "eyebrow_distribution_1-average": "0,1,0",
                    "eyebrow_distribution_2-dense": "0,0,1", "eyebrow_shape_0-angled": "1,0,0", "eyebrow_shape_1-curved": "0,1,0", "eyebrow_shape_2-straight": "0,0,1", "skin_color_0-light": "1,0,0", "skin_color_1-average": "0,1,0",
                    "skin_color_2-dark": "0,0,1", "skin_texture_0-average": "1,0,0", "skin_texture_1-middle-aged": "0,1,0", "skin_texture_2-aged": "0,0,1", "skin_spots_0-0": "1,0,0", "skin_spots_1-1": "0,1,0", "skin_spots_2-2+": "0,0,1", 
                    "eyelid_shape_0-exposed": "1,0", "eyelid_shape_1-covered": "0,1"}

def get_attribute_name(current, image): # auxiliary function, retrieves the image's attribute with respect to the feature given

    for i in list(filter(lambda x : x[0] != ".", os.listdir("../../dataset/annotations_merged/" + current))):
        directory = list(filter(lambda x : x[0] != ".", os.listdir("../../dataset/annotations_merged/" + current + "/" + i)))
        if(image in directory): return((current + "/" + i).replace("/","_"))

def make_csv_file(file_name): # main function, creates a ".csv" file with the annotations

    images = natsorted(list(filter(lambda x : x[0] != ".", os.listdir("../../dataset/dataset_one_folder"))),reverse = False)

    with open(file_name, "w") as file:

        # write the header
        file.write("image_name," + ",".join(HEADER) + "\n")

        # write the annotations
        for i in images:
            
            # in case of being an augmented image, we'll get the name of the original image (to get its annotations)
            if("+aug" in i): og_image = i.split("+aug")[0] + ".jpg"
            else: og_image = i
            
            line = i + ","

            # iris color
            line += HEADER_ENCODING[get_attribute_name("iris/color", og_image)] + ","
            
            # eyebrow distribution
            line += HEADER_ENCODING[get_attribute_name("eyebrow/distribution", og_image)] + ","

            # eyebrow shape
            line += HEADER_ENCODING[get_attribute_name("eyebrow/shape", og_image)] + ","

            # skin color
            line += HEADER_ENCODING[get_attribute_name("skin/color", og_image)] + ","

            # skin texture
            line += HEADER_ENCODING[get_attribute_name("skin/texture", og_image)] + ","

            # skin spots
            line += HEADER_ENCODING[get_attribute_name("skin/spots", og_image)] + ","
            
            # eyelid shape
            line += HEADER_ENCODING[get_attribute_name("eyelid/shape", og_image)]

            # update the file
            file.write(line + "\n")

if(__name__ == "__main__"):

    make_csv_file("../../dataset/annotations.csv")