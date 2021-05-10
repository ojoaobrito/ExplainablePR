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

# the values are coming from the "explain_pair.py" master script
else:
    IMAGE_SIZE = int(sys.argv[1])
    EXPLANATION_PATH = sys.argv[2]

def add_interpretable_colour(explanation, real_image): # auxiliary function, adds colour to an existing greyscale explanation

    if(explanation is None): return(None)

    real_image = Image.fromarray(np.dot(real_image[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)).convert("RGBA")
    real_image_np = np.asarray(real_image).copy()
    real_image_np[:, :, 3] = 150
    real_image = Image.fromarray(real_image_np.astype(np.uint8))

    coloured_explanation = explanation.copy()

    values = []

    for i in range(coloured_explanation.shape[0]):
        for j in range(coloured_explanation.shape[1]):
            if(coloured_explanation[i][j][0] not in values): values.append(coloured_explanation[i][j][0])

    values.sort()

    green_tones = np.linspace(0.5, 0, int(1.0 * int(round(len(values) / 3, 0)) + 1))
    greens = {i: green_tones[values.index(i)] for i in values[:int((len(values) // 3) * 1.0)]}

    red_tones = np.linspace(0.15, 1.0, 2 * int(round(len(values) / 3, 0)) + 2)
    reds = {i: red_tones[values.index(i) - ((len(values) // 3) * 1)] for i in values[int((len(values) // 3) * 1.0):]}
    
    for i in range(coloured_explanation.shape[0]):
        for j in range(coloured_explanation.shape[1]):
            if(coloured_explanation[i][j][0] in reds.keys()): coloured_explanation[i][j] = np.asarray([245, 39, 87, min(300 * max(0, reds[coloured_explanation[i][j][0]]), 255)])
            else: coloured_explanation[i][j] = np.asarray([24, 196, 93, 255 * greens[coloured_explanation[i][j][0]]])
    
    coloured_explanation_img = Image.fromarray(coloured_explanation.astype(np.uint8))
    real_image.paste(coloured_explanation_img, (0, 0), coloured_explanation_img)

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