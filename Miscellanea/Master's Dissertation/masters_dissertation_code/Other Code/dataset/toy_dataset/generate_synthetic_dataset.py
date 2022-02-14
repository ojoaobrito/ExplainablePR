import os, sys
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from random import randint, uniform, choice
import numpy as np
from math import ceil
from shutil import rmtree
from PIL import Image

#####################################################################################################
# CONTROL VARIABLES
#####################################################################################################
NUM_IMAGES = 15000 # total amount of images to be generated
IMAGE_SIZE = 128 # size of the final images
SHAPE_SIZE = (IMAGE_SIZE // 4, IMAGE_SIZE // 3) # min and max possible sizes for each generated shape
BORDER_WIDTH = (5, 15) # min and max possible widths for each generated shape
PNG_OR_JPEG = ".png" # either ".png" or ".jpg"

if __name__ == "__main__":

    if(os.path.exists("imgs")): rmtree("imgs")
    os.makedirs("imgs")

    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

    face_colors = ["black", "grey", "lightgrey", "whitesmoke", "white"]
    edge_colors = ["lime", "red", "blue"]

    count = 0

    ##################################################################################################################################################################################################################################
    # GENERATE THE SYNTHETIC DATASET
    ##################################################################################################################################################################################################################################
    while(count < NUM_IMAGES):

        print(str(count + 1) + "/" + str(NUM_IMAGES))

        plt.clf()
        ax = plt.gca()
        shape = None

        # randomly select a shape
        shape_number = 2
        #shape_number = choice([1, 2, 3, 4])

        # randomly select a background color
        #background_color = "black"
        background_color = colors[face_colors[randint(0, len(face_colors) - 1)]]
        
        try:
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # generate a square
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------
            if(shape_number == 1):
                
                line_width = float(randint(BORDER_WIDTH[0], BORDER_WIDTH[1]))

                width = randint(SHAPE_SIZE[0], SHAPE_SIZE[1])
                
                diagonal = ceil(np.sqrt((width ** 2) + (width ** 2)))
                start_position = (randint((diagonal // 2) + line_width, IMAGE_SIZE - width - line_width), randint(line_width, IMAGE_SIZE - diagonal -  line_width))
                
                face_color = colors[choice(list(filter(lambda x : x != background_color, face_colors)))]

                edge_color = colors[edge_colors[randint(0, len(edge_colors) - 1)]]

                angle = uniform(0, 60.0)

                #shape = plt.Rectangle(xy = start_position, width = ((SHAPE_SIZE[0] + SHAPE_SIZE[1]) // 2), height = ((SHAPE_SIZE[0] + SHAPE_SIZE[1]) // 2), angle = angle, linewidth = 0, facecolor = face_color, edgecolor = None)
                shape = plt.Rectangle(xy = start_position, width = width, height = width, angle = angle, linewidth = line_width, facecolor = face_color, edgecolor = edge_color)

            # ---------------------------------------------------------------------------------------------------------------------------------------------------
            # generate a circle
            # ---------------------------------------------------------------------------------------------------------------------------------------------------
            elif(shape_number == 2):

                line_width = float(randint(BORDER_WIDTH[0], BORDER_WIDTH[1]))

                radius = randint(SHAPE_SIZE[0] // 2, SHAPE_SIZE[1] // 2)
                
                start_position = (randint(radius + line_width, IMAGE_SIZE - radius - line_width), randint(radius + line_width, IMAGE_SIZE - radius - line_width))

                face_color = colors[choice(list(filter(lambda x : x != background_color, face_colors)))]

                edge_color = colors[edge_colors[randint(0, len(edge_colors) - 1)]]

                #shape = plt.Circle(start_position, radius = ((SHAPE_SIZE[0] + SHAPE_SIZE[1]) // 3), linewidth = 0, facecolor = "white", edgecolor = None)
                shape = plt.Circle(start_position, radius = radius, linewidth = line_width, facecolor = face_color, edgecolor = edge_color)

            # -------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # generate a rectangle
            # -------------------------------------------------------------------------------------------------------------------------------------------------------------------
            elif(shape_number == 3):

                line_width = float(randint(BORDER_WIDTH[0], BORDER_WIDTH[1]))
                
                while(True): # make sure the width is bigger than the height by, at least, 25% (thus ensuring a rectangle with minimum proportions)
                    
                    height = randint(SHAPE_SIZE[0], SHAPE_SIZE[1])
                    width = randint(SHAPE_SIZE[0], SHAPE_SIZE[1])

                    if(height == width): continue
                    if(width < (1.25 * height)): continue
                    break
                
                diagonal = ceil(np.sqrt((width ** 2) + (height ** 2)))
                start_position = (randint(height + line_width, IMAGE_SIZE - width - line_width), randint(line_width, IMAGE_SIZE - diagonal - line_width))

                face_color = colors[choice(list(filter(lambda x : x != background_color, face_colors)))]

                edge_color = colors[edge_colors[randint(0, len(edge_colors) - 1)]]

                angle = uniform(0, 90.0)

                shape = plt.Rectangle(xy = start_position, width = width, height = height, angle = angle, linewidth = line_width, facecolor = face_color, edgecolor = edge_color)

            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # generate a triangle
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            else:

                line_width = float(randint(BORDER_WIDTH[0], BORDER_WIDTH[1]))
                
                while(True): # make sure the area of the generated triangle is acceptable (at least, 20% of the total area)

                    random_points = np.asarray([[randint(line_width, IMAGE_SIZE - line_width), randint(line_width, IMAGE_SIZE - BORDER_WIDTH[0] - line_width)] for i in range(3)])

                    area = (random_points[0][0] * (random_points[1][1] - random_points[2][1]) + random_points[1][0] * (random_points[2][1] - random_points[0][1]) + random_points[2][0] * (random_points[0][1] - random_points[1][1]))
                    
                    if(abs(area / (IMAGE_SIZE ** 2)) < 0.2): continue
                    break

                face_color = colors[choice(list(filter(lambda x : x != background_color, face_colors)))]

                edge_color = colors[edge_colors[randint(0, len(edge_colors) - 1)]]
                
                shape = plt.Polygon(random_points, linewidth = line_width, facecolor = face_color, edgecolor = edge_color)
            
        except Exception as e: print(e)

        # --------------------------------------------------------------------------------------------------------------------------------------------
        # save the final image
        # --------------------------------------------------------------------------------------------------------------------------------------------
        ax.add_patch(shape)
        ax.set_aspect("equal", adjustable = "box")
        ax.set_axis_off()
        plt.axis([0, IMAGE_SIZE, 0, IMAGE_SIZE])

        # save the final image
        image_name = "imgs/" + f"{str(count + 1).zfill(len(str(NUM_IMAGES)))}" + PNG_OR_JPEG
        plt.savefig(image_name, dpi = 35, bbox_inches = "tight", facecolor = background_color)

        # resize the final image
        if(PNG_OR_JPEG == ".png"): Image.open(image_name).resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS).save(image_name)
        else: Image.open(image_name).resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS).save(image_name, format = "JPEG", subsampling = 0, quality = 100)

        count += 1