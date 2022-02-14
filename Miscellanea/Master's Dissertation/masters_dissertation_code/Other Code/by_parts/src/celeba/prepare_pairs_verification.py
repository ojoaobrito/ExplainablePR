import os
from PIL import Image
import numpy as np

pairs_to_explain = list(filter(lambda x : x[0] != ".", os.listdir("pairs_to_explain_verification")))

for i in pairs_to_explain:
    black = np.zeros((256, 256, 3))
    
    image_A_np = np.asarray(Image.open("pairs_to_explain_verification/" + i + "/1.jpg").resize((128, 128), Image.LANCZOS))
    image_B_np = np.asarray(Image.open("pairs_to_explain_verification/" + i + "/2.jpg").resize((128, 128), Image.LANCZOS))
    
    black[64:64+128, :128, :] = image_A_np
    black[64:64+128, 128:, :] = image_B_np

    Image.fromarray(black.astype(np.uint8)).save("pairs_to_explain_verification/" + i + ".jpg")
