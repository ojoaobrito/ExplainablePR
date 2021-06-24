import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import numpy as np
from PIL import Image

####################################################################################################
# CONTROL VARIABLES
####################################################################################################
# the values are the default ones (NOTE: if the values are to be changed, do so in this "if" branch)
if(len(sys.argv) == 1):
    IMAGE_SIZE = 128
    EXPLANATION_PATH = "../explanation.png"
    GREEN_NUMERATOR = 1
    DENOMINATOR = 3
    GREEN_MIN = 0.5
    GREEN_MAX = 0.0
    RED_MIN = 0.15
    RED_MAX = 1.0
    RED_INTENSITY = 300
    PASTE_EXPLANATION_ON_WHITE_BACKGROUND = True

# the values are coming from the "explain_pair.py" master script
else:
    IMAGE_SIZE = int(sys.argv[1])
    EXPLANATION_PATH = sys.argv[2]
    GREEN_NUMERATOR = int(sys.argv[3])
    DENOMINATOR = int(sys.argv[4])
    GREEN_MIN = float(sys.argv[5])
    GREEN_MAX = float(sys.argv[6])
    RED_MIN = float(sys.argv[7])
    RED_MAX = float(sys.argv[8])
    RED_INTENSITY = int(sys.argv[9])
    PASTE_EXPLANATION_ON_WHITE_BACKGROUND = True if(sys.argv[10] == "True") else False

def add_interpretable_colour(explanation, real_image): # auxiliary function, adds colour to an existing greyscale explanation

    if(explanation is None): return(None)

    real_image = Image.fromarray(np.dot(real_image[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)).convert("RGBA")
    real_image_np = np.asarray(real_image).copy()
    real_image_np[:, :, 3] = 150
    real_image = Image.fromarray(real_image_np.astype(np.uint8))

    coloured_explanation = explanation.copy()

    # ---------------------------------------------------------------------------------------------------
    # gather all the unique values in the greyscale explanation
    # ---------------------------------------------------------------------------------------------------
    values = []

    for i in range(coloured_explanation.shape[0]):
        for j in range(coloured_explanation.shape[1]):
            if(coloured_explanation[i][j][0] not in values): values.append(coloured_explanation[i][j][0])

    values.sort()

    # -------------------------------------------------------------------------------------------------
    # determine which values should be green and which should be red, according to the parameters above
    # -------------------------------------------------------------------------------------------------
    amount_of_greens = int(GREEN_NUMERATOR * int(round(len(values) / DENOMINATOR, 0)) + 1)
    green_values = values[:amount_of_greens]
    red_values = values[amount_of_greens:]

    green_tones = np.linspace(GREEN_MIN, GREEN_MAX, len(green_values))
    greens = {j: green_tones[green_values.index(j)] for j in green_values}

    red_tones = np.linspace(RED_MIN, RED_MAX, len(red_values))
    reds = {j: red_tones[red_values.index(j)] for j in red_values}
    
    # ---------------------------------------------------------------------------------------------------------------------------------------
    # actually change the greyscale values to either green or red tones
    # ---------------------------------------------------------------------------------------------------------------------------------------
    for i in range(coloured_explanation.shape[0]):
        for j in range(coloured_explanation.shape[1]):
            if(coloured_explanation[i][j][0] in reds.keys()): 
                coloured_explanation[i][j] = np.asarray([245, 39, 87, min(RED_INTENSITY * max(0, reds[coloured_explanation[i][j][0]]), 255)])
            else: 
                coloured_explanation[i][j] = np.asarray([24, 196, 93, 255 * greens[coloured_explanation[i][j][0]]])
    
    coloured_explanation_img = Image.fromarray(coloured_explanation.astype(np.uint8))
    real_image.paste(coloured_explanation_img, (0, 0), coloured_explanation_img)

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # if required, paste the RGBA explanation on top of a white background, so that the tones become more discernible
    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    if(PASTE_EXPLANATION_ON_WHITE_BACKGROUND):
        white_template = Image.fromarray((np.ones((coloured_explanation_img.shape[0], coloured_explanation_img.shape[1], 3), int) * 255).astype(np.uint8))
        real_image = real_image.convert("RGBA")
        white_template.paste(real_image, (0, 0), real_image)
        real_image = white_template

    return(real_image)

if(__name__ == "__main__"):

    explanation = np.asarray(Image.open(EXPLANATION_PATH).convert("RGBA")).copy()

    # -----------------------------------------------------------------------------------
    # load the greyscale explanation(s)
    # -----------------------------------------------------------------------------------
    if(explanation.shape[1] < 500): # the explanation is only applied to image B
        image_A = None
        image_B = explanation[38:37 + IMAGE_SIZE, 139:(139 + (IMAGE_SIZE - 1)), :].copy()
        
        explanation_A = None
        explanation_B = explanation[38:37 + IMAGE_SIZE, 285:(285 + (IMAGE_SIZE - 1)), :]
    
    else:
        image_A = explanation[38:37 + IMAGE_SIZE, 1:(1 + IMAGE_SIZE - 1), :].copy()
        image_B = explanation[38:37 + IMAGE_SIZE, 139:(139 + (IMAGE_SIZE - 1)), :].copy()
        
        explanation_A = explanation[38:37 + IMAGE_SIZE, 284:(284 + (IMAGE_SIZE - 1)), :]
        explanation_B = explanation[38:37 + IMAGE_SIZE, 422:(422 + (IMAGE_SIZE - 1)), :]
    
    # ----------------------------------------------------------------------------------------------------------------------
    # add colour to the existing explanation(s)
    # ----------------------------------------------------------------------------------------------------------------------
    coloured_explanation_A = add_interpretable_colour(explanation_A, image_A)
    coloured_explanation_B = add_interpretable_colour(explanation_B, image_B)

    if(explanation_A is not None): explanation[38:37 + IMAGE_SIZE, 284:(284 + (IMAGE_SIZE - 1)), :] = coloured_explanation_A
    explanation[38:37 + IMAGE_SIZE, 422:(422 + (IMAGE_SIZE - 1)), :] = coloured_explanation_B

    Image.fromarray(explanation.astype(np.uint8)).save(EXPLANATION_PATH)