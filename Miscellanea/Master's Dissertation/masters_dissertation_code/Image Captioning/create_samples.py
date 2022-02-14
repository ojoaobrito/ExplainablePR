import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from natsort import natsorted
import time
import datetime

TRANSPARENT_BACKGROUND = True
IMAGE_SIZE = 128

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_text_caption(image):
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderCNN(args.embed_size, args.model_type, args.mode)  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.model_path + "_" + args.model_type + "/encoder.pt"))
    encoder.eval()
    decoder.load_state_dict(torch.load(args.model_path + "_" + args.model_type + "/decoder.pt"))
    decoder.eval()

    # Prepare an image
    image_tensor = image.to(device)
    
    # Generate an caption from the image
    feature = encoder(image_tensor)

    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    print(sampled_ids)
    
    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    
    return(sentence.split("<start> ")[1].split(" <end>")[0][:-2].capitalize().replace(" , ", ", "))

def highlight_words(draw, font, explanation, words_to_highlight, lines):

    for i in words_to_highlight:
        if(i in lines[0]): # there is a word that needs colouring in the first (and possibly only) line

            caption_w, caption_h = font.getsize(lines[0])
            left_side = (explanation.size[0] - caption_w) / 2

            caption_before_w, caption_before_h = font.getsize(lines[0].split(i)[0])
            left_side += caption_before_w

            offset = 18 if(len(lines) == 1) else 24
            draw.text((left_side, (explanation.size[1] - offset) - caption_h), i, font = font, fill = "rgb(179, 0, 0)")

        elif(len(lines) == 2 and i in lines[1]):
            
            caption_w, caption_h = font.getsize(lines[1])
            left_side = (explanation.size[0] - caption_w) / 2

            caption_before_w, caption_before_h = font.getsize(lines[1].split(i)[0])
            left_side += caption_before_w

            draw.text((left_side, (explanation.size[1] - 8) - caption_h), i, font = font, fill = "rgb(179, 0, 0)")

def assemble_explanation(img_A_np, img_B_np, caption):

    words_to_highlight = ["iris", "irises", "skin", "skins", "eyebrow", "eyebrows", "eyelid", "eyelids", "distribution", "distributions", 
                            "spot", "spots", "texture", "textures", "color", "colors", "shape", "shapes"]

    image_A = Image.fromarray(img_A_np.astype(np.uint8)).resize((IMAGE_SIZE - 1, IMAGE_SIZE - 1), Image.LANCZOS).convert("RGBA")
    image_B = Image.fromarray(img_B_np.astype(np.uint8)).resize((IMAGE_SIZE - 1, IMAGE_SIZE - 1), Image.LANCZOS).convert("RGBA")

    explanation = np.asarray(Image.open("explanation_resources/explanation_template_G.png").convert("RGBA")).copy()
    if(not TRANSPARENT_BACKGROUND): explanation[:, :, 3] = 255

    # add image A to the explanation template
    explanation[38:37 + IMAGE_SIZE, 2:IMAGE_SIZE + 1, :] = image_A

    # add image B to the explanation template
    explanation[38:37 + IMAGE_SIZE, IMAGE_SIZE + 12:(IMAGE_SIZE * 2)  + 11, :] = image_B

    explanation = Image.fromarray(explanation.astype(np.uint8))

    font_fname = "explanation_resources/helvetica_bold.ttf"
    font_size = 15
    font = ImageFont.truetype(font_fname, font_size)

    draw = ImageDraw.Draw(explanation)

    if(len(caption) > 33):
        best_space_index = 0
        for idx, i in enumerate(caption):
            if((i == ' ') and (idx <= 33)): best_space_index = idx
        
        caption = caption[:best_space_index] + "\n" + caption[best_space_index:]

    if(len(caption) <= 33):
        w, h = font.getsize(caption)
        draw.text(((explanation.size[0] - w) / 2, (explanation.size[1] - 18) - h), caption, font = font, fill = "rgb(160, 160, 160)")
        highlight_words(draw, font, explanation, words_to_highlight, [caption])

    else:
        first_line = caption.split("\n")[0]
        second_line = caption.split("\n")[1]

        w, h = font.getsize(first_line)
        draw.text(((explanation.size[0] - w) / 2, (explanation.size[1] - 24) - h), first_line, font = font, fill = "rgb(160, 160, 160)")
        
        w, h = font.getsize(second_line)
        draw.text(((explanation.size[0] - w) / 2, (explanation.size[1] - 8) - h), second_line, font = font, fill = "rgb(160, 160, 160)")

        highlight_words(draw, font, explanation, words_to_highlight, [first_line, second_line])

    return(explanation)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='depthwise' , choices=["depthwise", "side_by_side"] , help='image mode')
    parser.add_argument('--model_path', type=str, default='models' , help='path for saving trained models')
    parser.add_argument('--vocab_path', type=str, default='our_data/vocab.pkl', help='path for vocabulary wrapper')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--model_type', type=str , default="vgg_19", help='cnn architecture')
    parser.add_argument('--embed_size', type=int , default=512, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    
    pairs_to_explain = natsorted(list(filter(lambda x : x[0] != ".", os.listdir("pairs_to_explain"))))

    os.makedirs("explanations", exist_ok = True)
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H_%M_%S")
    os.makedirs("explanations/" + timestamp)

    # Image preprocessing
    transform = transforms.Compose([
        #transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    for idx, i in enumerate(pairs_to_explain):

        print(str(idx + 1) + "/" + str(len(pairs_to_explain)))

        img_A_np = np.asarray(Image.open("pairs_to_explain/" + i + "/1.jpg"))
        img_B_np = np.asarray(Image.open("pairs_to_explain/" + i + "/2.jpg"))

        # -------------------------------------------------------------------
        # prepare the image for the captioning step
        # -------------------------------------------------------------------
        if(args.mode == "side_by_side"):
            black_template = np.zeros((512, 512, 3))

            black_template[128:384, :256, :] = img_A_np
            black_template[128:384, 256:, :] = img_B_np

            img = Image.fromarray(black_template.astype(np.uint8)).resize((224, 224), Image.LANCZOS)
            image = transform(img)

        elif(args.mode == "depthwise"):
            black_template = np.zeros((224, 224, 6))

            img_A = Image.fromarray(img_A_np.astype(np.uint8)).resize((224, 224), Image.LANCZOS)
            img_B = Image.fromarray(img_B_np.astype(np.uint8)).resize((224, 224), Image.LANCZOS)

            img_A = transform(img_A)
            img_B = transform(img_B)
            
            image = torch.cat((img_A, img_B), 0)

        # -------------------------------------------
        # generate the corresponding caption
        # -------------------------------------------
        sample_caption = get_text_caption(image.unsqueeze(0))
        print(sample_caption)

        # -------------------------------------------------------------------------------------
        # assemble the final explanation
        # -------------------------------------------------------------------------------------
        explanation = assemble_explanation(img_A_np, img_B_np, sample_caption)
        explanation.save("explanations/" + timestamp + "/" + i + "_explanation.png")