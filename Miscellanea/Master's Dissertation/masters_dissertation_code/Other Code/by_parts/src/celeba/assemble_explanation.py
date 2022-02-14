import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import numpy as np
from PIL import Image
from natsort import natsorted
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

def assemble_explanation(image_A, image_B, difference_mask_1, difference_mask_2, score, genuine_or_impostor, save_path): # auxiliary function, creates the final explanation and saves it

    IMAGE_SIZE = 128
    TRANSPARENT_BACKGROUND = True

    # ---------------------------------------------------------------------------------------------------------------
    # place image A and B on the explanation template (the pair was classified as being genuine)
    # ---------------------------------------------------------------------------------------------------------------
    if(difference_mask_1 is None and difference_mask_2 is None):
        final_img = np.asarray(Image.open("explanation_resources/explanation_template_G.png").convert("RGBA")).copy()
        if(not TRANSPARENT_BACKGROUND): final_img[:, :, 3] = 255

        # add image A to the explanation template
        final_img[38:37 + IMAGE_SIZE, 2:IMAGE_SIZE + 1, :] = image_A

        # add image B to the explanation template
        final_img[38:37 + IMAGE_SIZE, IMAGE_SIZE + 12:(IMAGE_SIZE * 2)  + 11, :] = image_B

        final_img = Image.fromarray(final_img.astype(np.uint8))

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # place image A, B and the mask(s) on the explanation template (the pair was classified as being impostor)
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    elif(difference_mask_1 is not None and difference_mask_2 is None):
        final_img = np.asarray(Image.open("explanation_resources/explanation_template_I_single_explanation.png").convert("RGBA")).copy()
        if(not TRANSPARENT_BACKGROUND): final_img[:, :, 3] = 255

        # add image A to the explanation template
        final_img[38:37 + IMAGE_SIZE, 2:(2 + IMAGE_SIZE - 1), :] = image_A

        # add image B to the explanation template
        final_img[38:37 + IMAGE_SIZE, 140:(140 + (IMAGE_SIZE - 1)), :] = image_B

        # add the first difference mask to the explanation template
        final_img[38:37 + IMAGE_SIZE, 285:(285 + (IMAGE_SIZE - 1)), :] = np.asarray(Image.fromarray(difference_mask_1.astype(np.uint8)).resize((IMAGE_SIZE - 1, IMAGE_SIZE - 1)).convert("RGBA"))
    
    else:

        final_img = np.asarray(Image.open("explanation_resources/explanation_template_I_double_explanation.png").convert("RGBA")).copy()
        if(not TRANSPARENT_BACKGROUND): final_img[:, :, 3] = 255

        # add image A to the explanation template
        final_img[38:37 + IMAGE_SIZE, 1:(1 + IMAGE_SIZE - 1), :] = image_A

        # add image B to the explanation template
        final_img[38:37 + IMAGE_SIZE, 139:(139 + (IMAGE_SIZE - 1)), :] = image_B

        # add the first difference mask to the explanation template
        final_img[38:37 + IMAGE_SIZE, 284:(284 + (IMAGE_SIZE - 1)), :] = np.asarray(Image.fromarray(difference_mask_2.astype(np.uint8)).resize((IMAGE_SIZE - 1, IMAGE_SIZE - 1)).convert("RGBA"))

        # add the second difference mask to the explanation template
        final_img[38:37 + IMAGE_SIZE, 422:(422 + (IMAGE_SIZE - 1)), :] = np.asarray(Image.fromarray(difference_mask_1.astype(np.uint8)).resize((IMAGE_SIZE - 1, IMAGE_SIZE - 1)).convert("RGBA"))

    final_img = Image.fromarray(final_img.astype(np.uint8))

    # ---------------------------------------------------------------------------------------------------------------------------
    # add text to the final explanation (with the final answer and CNN score)
    # ---------------------------------------------------------------------------------------------------------------------------
    font_fname = "explanation_resources/helvetica_bold.ttf"
    font_size = 21
    font = ImageFont.truetype(font_fname, font_size)

    draw = ImageDraw.Draw(final_img)

    # construct the final text answer
    final_answer = "Same subject" if(genuine_or_impostor == "G") else "Different subjects"
    final_answer += " (" + str(score) + ")"

    w, h = font.getsize(final_answer)

    draw.text(((final_img.size[0] - w) / 2, (final_img.size[1] - 3) - h), final_answer, font = font, fill = "rgb(160, 160, 160)")

    final_img.save(save_path)

if(__name__ == "__main__"):

    directory = natsorted(list(filter(lambda x : x[0] != ".", os.listdir("../images"))))
    genuine_or_impostor = "G" if("G" in directory[0]) else "I"
    score = float(str(directory[0].split("_")[2]).replace(",", "."))

    # load both images
    image_A = np.asarray(Image.open("../images/" + directory[0]).resize((IMAGE_SIZE - 1, IMAGE_SIZE - 1), Image.LANCZOS).convert("RGBA"))
    image_B = np.asarray(Image.open("../images/" + directory[1]).resize((IMAGE_SIZE - 1, IMAGE_SIZE - 1), Image.LANCZOS).convert("RGBA"))

    # if they exist, load the difference masks
    if(os.path.exists("difference_mask_1.npy")): difference_mask_1 = np.load("difference_mask_1.npy")
    else: difference_mask_1 = None

    if(os.path.exists("difference_mask_2.npy")): difference_mask_2 = np.load("difference_mask_2.npy")
    else: difference_mask_2 = None

    # create the final explanation and save it
    assemble_explanation(image_A, image_B, difference_mask_1, difference_mask_2, score, genuine_or_impostor, "../explanation.png")