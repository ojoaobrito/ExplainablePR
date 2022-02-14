# pytorch, vis and image libs
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
from PIL import Image
import colorsys
import torch
import torch.nn as nn
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import cv2
import json
from collections import OrderedDict

# sys libs
import os
import argparse
import random

# dataset, utils and model
import sys
import os
import time
sys.path.append(os.path.abspath('../common'))
from utils import *
from celeba import *
from model import ResNet101, ResNet50
from natsort import natsorted
np.set_printoptions(threshold = sys.maxsize)

# fix all the randomness for reproducibility
torch.backends.cudnn.enabled = False
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

# number of attributes
num_classes = 2

# arguments
parser = argparse.ArgumentParser(description='Result Visualization')
parser.add_argument('--load', default='', type=str, help='name of model to visualize')
args = parser.parse_args()
'''
UBIPR_CLASSES = ['0', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '18', 
                '19', '20', '21', '22', '23', '24', '25', '26', '28', '29', '30', '31', '32', '33', '34', '35', 
                '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', 
                '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', 
                '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', 
                '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '98', '99', '100', 
                '101', '102', '103', '104', '105', '107', '108', '109', '110', '111', '112', '113', '114', '115', 
                '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', 
                '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', 
                '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', 
                '159', '160', '161', '162', '163', '164', '165', '166', '168', '169', '170', '171', '172', '173', 
                '174', '175', '176', '177', '178', '179', '180', '181', '182', '183', '184', '185', '186', '187', 
                '188', '189', '190', '191', '192', '193', '195', '196', '197', '198', '199', '200', '202', '203', 
                '204', '206', '207', '208', '209', '210', '211', '212', '213', '214', '215', '216', '217', '218', 
                '219', '220', '221', '222', '223', '224', '225', '226', '227', '228', '229', '230', '231', '232', 
                '233', '234', '235', '236', '237', '238', '239', '240', '241', '242', '243', '244', '245', '246', 
                '247', '248', '249', '250', '251', '253', '254', '255', '256', '257', '258']
'''

UBIPR_CLASSES = ["I", "G"]
import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import numpy as np
from PIL import Image
from random import shuffle
from torch.utils.data import Dataset
from pickle import load
from assemble_explanation import assemble_explanation
from shutil import rmtree

JUST_CARE_ABOUT_THE_SCORES = True

class UBIPr_Identification(Dataset): # custom Dataset class

    def __init__(self, img_path, split, transform): # class constructor

        self.img_path = img_path
        self.transform = transform
        self.split = split

        images_aux = natsorted(list(filter(lambda x : x[0] != "." and ".jpg" in x, os.listdir(self.img_path))))
        self.images = []

        for idx in range(0, len(images_aux), 2):
            self.images.append((images_aux[idx], images_aux[idx + 1]))

        with open("../../data/ubipr_identification/ids_one_hot_encoding.pkl", "rb") as file:
            self.ids_one_hot = load(file)

        #shuffle(self.images)

    def __len__(self): # auxiliary method, retrieves the length of the dataset
        return(len(self.images))

    def __getitem__(self, idx): # auxiliary method, retrieves a sample and its label
        
        image_A_label = str(int(self.images[idx][0].split("_")[2].replace(".jpg", "")) - 1)
        image_B_label = str(int(self.images[idx][1].split("_")[2].replace(".jpg", "")) - 1)

        image_A = Image.open(self.img_path + "/" + self.images[idx][0])
        image_B = Image.open(self.img_path + "/" + self.images[idx][1])

        if(self.transform is not None):
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)

        return((image_A, image_B), (self.ids_one_hot[image_A_label], self.ids_one_hot[image_B_label]), (image_A_label, image_B_label), np.zeros((1,)))

class UBIPr_Verification(Dataset): # custom Dataset class

    def __init__(self, img_path, split, transform): # class constructor
        
        self.img_path = img_path
        self.transform = transform
        self.split = split

        self.images = natsorted(list(filter(lambda x : x[0] != "." and ".jpg" in x, os.listdir(self.img_path))))

        shuffle(self.images)

    def __len__(self): # auxiliary method, retrieves the length of the dataset
        return(len(self.images))

    def __getitem__(self, idx): # auxiliary method, retrieves a sample and its label
        
        label = [1, 0]

        image = Image.open(self.img_path + "/" + self.images[idx])

        if(self.transform is not None):
            image = self.transform(image)

        return(image, np.asarray(label), np.zeros((1,)))

def generate_colors(num_colors):
    """
    Generate distinct value by sampling on hls domain.

    Parameters
    ----------
    num_colors: int
        Number of colors to generate.

    Returns
    ----------
    colors_np: np.array, [num_colors, 3]
        Numpy array with rows representing the colors.

    """
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = 0.5
        saturation = 0.9
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    colors_np = np.array(colors)*255.

    return colors_np

def show_att_on_image(img, mask, output):
    """
    Convert the grayscale attention into heatmap on the image, and save the visualization.

    Parameters
    ----------
    img: np.array, [H, W, 3]
        Original colored image.
    mask: np.array, [H, W]
        Attention map normalized by subtracting min and dividing by max.
    output: str
        Destination image (path) to save.

    Returns
    ----------
    Save the result to output.

    """
    # generate heatmap and normalize into [0, 1]
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    # add heatmap onto the image
    merged = heatmap + np.float32(img)

    # re-scale the image
    merged = merged / np.max(merged)
    cv2.imwrite(output, np.uint8(255 * merged))

def plot_assignment(root, assign_hard, num_parts, A_or_B):
    """
    Blend the original image and the colored assignment maps.

    Parameters
    ----------
    root: str
        Root path for saving visualization results.
    assign_hard: np.array, [H, W]
        Hard assignment map (int) denoting the deterministic assignment of each pixel. Generated via argmax.
    num_parts: int, number of object parts.

    Returns
    ----------
    Save the result to root/assignment.png.

    """
    if(A_or_B is None): A_or_B = ""
    else: A_or_B = "_" + A_or_B
    # generate the numpy array for colors
    colors = generate_colors(num_parts)
    
    # coefficient for blending
    coeff = 0.4

    # load the input as RGB image, convert into numpy array
    input = Image.open(os.path.join(root, 'input' + A_or_B + '.png')).convert('RGB')
    input_np = np.array(input).astype(float)

    # blending by each pixel
    for i in range(assign_hard.shape[0]):
        for j in range(assign_hard.shape[1]):
            assign_ij = assign_hard[i][j]
            input_np[i, j] = (1-coeff) * input_np[i, j] + coeff * colors[assign_ij]

    # save the resulting image
    im = Image.fromarray(np.uint8(input_np))
    im.save(os.path.join(root, 'assignment' + A_or_B + '.png'))

def main():

    # load the config file
    config_file = '../../log/'+ args.load +'/train_config.json'
    with open(config_file) as fi:
        config = json.load(fi)
        print(" ".join("\033[96m{}\033[0m: {},".format(k, v) for k, v in config.items()))

    # define data transformation (no crop)
    test_transforms = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                std=(0.229, 0.224, 0.225))
        ])

    # wrap to dataset
    #test_data = UBIPr_Identification("pairs_to_explain_identification", split='train', transform=test_transforms)
    test_data = UBIPr_Verification("pairs_to_explain_verification", split='train', transform=test_transforms)

    # wrap to dataloader
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=False, drop_last=False)
    
    test_loader_iter = iter(test_loader)

    # define the figure layout
    fig_rows = 5
    fig_cols = 5
    f_assign, axarr_assign = plt.subplots(fig_rows, fig_cols, figsize=(fig_cols*2,fig_rows*2))
    f_assign.subplots_adjust(wspace=0, hspace=0)

    # load the model in eval mode
    # with batch size = 1, we only support single GPU visaulization
    if config['arch'] == 'resnet101':
        model = ResNet101(num_classes, num_parts=config['nparts']).cuda()
    elif config['arch'] == 'resnet50':
        model = ResNet50(num_classes, num_parts=config['nparts']).cuda()
    else:
        raise(RuntimeError("Only support resnet50 or resnet101 for architecture!"))

    # load model
    resume = '../../checkpoints/'+args.load+'_best.pth.tar'
    print("=> loading checkpoint '{}'".format(resume))
    checkpoint = torch.load(resume)
    # remove the module prefix
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    with torch.no_grad():
        # the visualization code
        current_id = 0
        for i in range(100):

            t0 = time.time()

            # inference the model
            img_batch, ground_truth, _ = next(test_loader_iter)
            
            input = img_batch.cuda()
            target = ground_truth.cuda()

            #image_A = img_batch[0][0].cuda()
            #image_B = img_batch[1][0].cuda()

            '''image_A_labels = img_labels[0][0].cuda()
            image_B_labels = img_labels[1][0].cuda()'''

            current_id += 1
            with torch.no_grad():
                print("Visualizing %dth image..." % current_id)
                #output_list_A, att_list_A, assign_A = model(torch.reshape(image_A, [1, 3, 256, 256]))
                #output_list_B, att_list_B, assign_B = model(torch.reshape(image_B, [1, 3, 256, 256]))
                output_list, att_list, assign = model(input)

            # define root for saving results and make directories correspondingly
            root = os.path.join('../../visualization', args.load, str(current_id))
            os.makedirs(root, exist_ok=True)

            '''os.makedirs(os.path.join(root, 'attentions_A'), exist_ok=True)
            os.makedirs(os.path.join(root, 'attentions_B'), exist_ok=True)'''
            
            os.makedirs(os.path.join(root, 'attentions'), exist_ok=True)
            
            if(not JUST_CARE_ABOUT_THE_SCORES):
                '''os.makedirs(os.path.join(root, 'assignments_A'), exist_ok=True)
                os.makedirs(os.path.join(root, 'assignments_B'), exist_ok=True)'''
                os.makedirs(os.path.join(root, 'assignments'), exist_ok=True)

            # denormalize the image and save the input
            '''save_input = transforms.Normalize(mean=(0, 0, 0),std=(1/0.229, 1/0.224, 1/0.225))(torch.reshape(image_A, [1, 3, 256, 256]).data[0].cpu())
            save_input = transforms.Normalize(mean=(-0.485, -0.456, -0.406),std=(1, 1, 1))(save_input)
            save_input = torch.nn.functional.interpolate(save_input.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
            img = torchvision.transforms.ToPILImage()(save_input)
            
            img.save(os.path.join(root, 'input_A.png'))

            save_input = transforms.Normalize(mean=(0, 0, 0),std=(1/0.229, 1/0.224, 1/0.225))(torch.reshape(image_B, [1, 3, 256, 256]).data[0].cpu())
            save_input = transforms.Normalize(mean=(-0.485, -0.456, -0.406),std=(1, 1, 1))(save_input)
            save_input = torch.nn.functional.interpolate(save_input.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
            img = torchvision.transforms.ToPILImage()(save_input)
            
            img.save(os.path.join(root, 'input_B.png'))'''

            # denormalize the image and save the input
            save_input = transforms.Normalize(mean=(0, 0, 0),std=(1/0.229, 1/0.224, 1/0.225))(input.data[0].cpu())
            save_input = transforms.Normalize(mean=(-0.485, -0.456, -0.406),std=(1, 1, 1))(save_input)
            save_input = torch.nn.functional.interpolate(save_input.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
            img = torchvision.transforms.ToPILImage()(save_input)
            img.save(os.path.join(root, 'input.png'))
            
            # save the labels and pred as list
            '''label_A = list(torch.reshape(image_A_labels, [1, image_A_labels.shape[0]]).data[0].cpu().numpy())
            assert (len(label_A) == num_classes)
            prediction_A = []
            highest_predicted_class_A = (0.0, 0, 0)
            for k in range(num_classes):
                current_pred = torch.sigmoid(output_list_A[k]).squeeze().data.item()
                if(current_pred > highest_predicted_class_A[0]): highest_predicted_class_A = (current_pred, UBIPR_CLASSES[k], k)
                
                prediction_A.append(current_pred)
            
            label_B = list(torch.reshape(image_B_labels, [1, image_B_labels.shape[0]]).data[0].cpu().numpy())
            prediction_B = []
            highest_predicted_class_B = (0.0, 0, 0)
            for k in range(num_classes):
                current_pred = torch.sigmoid(output_list_B[k]).squeeze().data.item()
                if(current_pred > highest_predicted_class_B[0]): highest_predicted_class_B = (current_pred, UBIPR_CLASSES[k], k)
                
                prediction_B.append(current_pred)'''

            # save the labels and pred as list
            label = list(target.data[0].cpu().numpy())
            prediction = []
            assert (len(label) == num_classes)
            highest_predicted_class = (0.0, 0, 0)
            for k in range(num_classes):
                current_pred = torch.sigmoid(output_list[k]).squeeze().data.item()
                #current_pred = int(current_score > 0.5)
                if(current_pred > highest_predicted_class[0]): highest_predicted_class = (current_pred, UBIPR_CLASSES[k], k)
                
                prediction.append(current_pred)

            # write the labels and pred
            '''if(not JUST_CARE_ABOUT_THE_SCORES):
                with open(os.path.join(root, 'prediction_A.txt'), 'w') as pred_log:
                    for k in range(num_classes):
                        pred_log.write('%s pred: %f, label: %d\n' % (UBIPR_CLASSES[k], prediction_A[k], label_A[k]))

                with open(os.path.join(root, 'prediction_B.txt'), 'w') as pred_log:
                    for k in range(num_classes):
                        pred_log.write('%s pred: %f, label: %d\n' % (UBIPR_CLASSES[k], prediction_B[k], label_B[k]))

            # upsample the assignment and transform the attention correspondingly
            assign_A_reshaped = torch.nn.functional.interpolate(assign_A.data.cpu(), size=(256, 256), mode='bilinear', align_corners=False)
            assign_B_reshaped = torch.nn.functional.interpolate(assign_B.data.cpu(), size=(256, 256), mode='bilinear', align_corners=False)'''

            # write the labels and pred
            with open(os.path.join(root, 'prediction.txt'), 'w') as pred_log:
                for k in range(num_classes):
                    pred_log.write('%s pred: %f, label: %d\n' % (UBIPR_CLASSES[k], prediction[k], label[k]))

            # upsample the assignment and transform the attention correspondingly
            assign_reshaped = torch.nn.functional.interpolate(assign.data.cpu(), size=(256, 256), mode='bilinear', align_corners=False)

            # visualize the attention
            '''for k in range(num_classes):

                #if(k != highest_predicted_class[2]): continue

                # attention vector for kth attribute
                att = att_list_A[k].view(1, config['nparts'], 1, 1).data.cpu()

                # multiply the assignment with the attention vector
                assign_att = assign_A_reshaped * att

                # sum along the part dimension to calculate the spatial attention map
                attmap_hw = torch.sum(assign_att, dim=1).squeeze(0).numpy()

                # normalize the attention map and merge it onto the input
                img = cv2.imread(os.path.join(root, 'input_A.png'))
                mask_A = attmap_hw / attmap_hw.max()

                # save the attention map for image A
                np.save(os.path.join(root, 'attention_map_A.npy'), mask_A)

                img_float = img.astype(float) / 255.

                show_att_on_image(img_float, mask_A, os.path.join(root, 'attentions_A', UBIPR_CLASSES[k]+'.png'))

            # generate the one-channel hard assignment via argmax
            _, assign = torch.max(assign_A_reshaped, 1)

            # colorize and save the assignment
            if(not JUST_CARE_ABOUT_THE_SCORES):
                plot_assignment(root, assign.squeeze(0).numpy(), config['nparts'], "A")

                # collect the assignment for the final image array
                color_assignment_name = os.path.join(root, 'assignment_A.png')
                color_assignment = mpimg.imread(color_assignment_name)
                #axarr_assign[j, col_id].imshow(color_assignment)
                #axarr_assign[j, col_id].axis('off')

            # plot the assignment for each dictionary vector
            if(not JUST_CARE_ABOUT_THE_SCORES):
                for i in range(config['nparts']):
                    img = torch.nn.functional.interpolate(assign_A_reshaped.data[:, i].cpu().unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)
                    img = torchvision.transforms.ToPILImage()(img.squeeze(0))
                    img.save(os.path.join(root, 'assignments_A', 'part_'+str(i)+'.png'))
            
            # visualize the attention
            for k in range(num_classes):

                #if(k != highest_predicted_class[2]): continue

                # attention vector for kth attribute
                att = att_list_B[k].view(1, config['nparts'], 1, 1).data.cpu()

                # multiply the assignment with the attention vector
                assign_att = assign_B_reshaped * att

                # sum along the part dimension to calculate the spatial attention map
                attmap_hw = torch.sum(assign_att, dim=1).squeeze(0).numpy()

                # normalize the attention map and merge it onto the input
                img = cv2.imread(os.path.join(root, 'input_B.png'))
                mask_B = attmap_hw / attmap_hw.max()

                # save the attention map for image B
                np.save(os.path.join(root, 'attention_map_B.npy'), mask_B)
                
                img_float = img.astype(float) / 255.

                show_att_on_image(img_float, mask_B, os.path.join(root, 'attentions_B', UBIPR_CLASSES[k]+'.png'))

            # generate the one-channel hard assignment via argmax
            _, assign = torch.max(assign_B_reshaped, 1)

            # colorize and save the assignment
            if(not JUST_CARE_ABOUT_THE_SCORES):
                plot_assignment(root, assign.squeeze(0).numpy(), config['nparts'], "B")

                # collect the assignment for the final image array
                color_assignment_name = os.path.join(root, 'assignment_B.png')
                color_assignment = mpimg.imread(color_assignment_name)
                #axarr_assign[j, col_id].imshow(color_assignment)
                #axarr_assign[j, col_id].axis('off')

            # plot the assignment for each dictionary vector
            if(not JUST_CARE_ABOUT_THE_SCORES):
                for i in range(config['nparts']):
                    img = torch.nn.functional.interpolate(assign_B_reshaped.data[:, i].cpu().unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)
                    img = torchvision.transforms.ToPILImage()(img.squeeze(0))
                    img.save(os.path.join(root, 'assignments_B', 'part_'+str(i)+'.png'))
            '''
            # visualize the attention
            for k in range(num_classes):

                if(k != 0): continue

                # attention vector for kth attribute
                att = att_list[k].view(1, config['nparts'], 1, 1).data.cpu()

                # multiply the assignment with the attention vector
                assign_att = assign_reshaped * att

                # sum along the part dimension to calculate the spatial attention map
                attmap_hw = torch.sum(assign_att, dim=1).squeeze(0).numpy()

                # normalize the attention map and merge it onto the input
                img = cv2.imread(os.path.join(root, 'input.png'))
                mask = attmap_hw / attmap_hw.max()
                
                # save the attention map
                np.save(os.path.join(root, 'attention_map.npy'), mask)

                img_float = img.astype(float) / 255.
                show_att_on_image(img_float, mask, os.path.join(root, 'attentions', UBIPR_CLASSES[k]+'.png'))

            # generate the one-channel hard assignment via argmax
            _, assign = torch.max(assign_reshaped, 1)

            # colorize and save the assignment
            if(not JUST_CARE_ABOUT_THE_SCORES):
                plot_assignment(root, assign.squeeze(0).numpy(), config['nparts'], None)

                # collect the assignment for the final image array
                color_assignment_name = os.path.join(root, 'assignment.png')
                color_assignment = mpimg.imread(color_assignment_name)
                #axarr_assign[j, col_id].imshow(color_assignment)
                #axarr_assign[j, col_id].axis('off')

            # plot the assignment for each dictionary vector
            if(not JUST_CARE_ABOUT_THE_SCORES):
                for i in range(config['nparts']):
                    img = torch.nn.functional.interpolate(assign_reshaped.data[:, i].cpu().unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)
                    img = torchvision.transforms.ToPILImage()(img.squeeze(0))
                    img.save(os.path.join(root, 'assignments', 'part_'+str(i)+'.png'))

            # --------------------------------------------------------------------------------------------------------------------------------
            # build the final explanation
            # --------------------------------------------------------------------------------------------------------------------------------
            '''difference_mask_1 = np.asarray(Image.open(os.path.join(root, 'attentions_A') + "/" + ground_truth[0][0] + ".png").convert("RGBA"))
            difference_mask_2 = np.asarray(Image.open(os.path.join(root, 'attentions_B') + "/" + ground_truth[1][0] + ".png").convert("RGBA"))

            image_A = np.asarray(Image.open(os.path.join(root, 'input_A.png')).convert("RGBA").resize((127, 127), Image.LANCZOS))
            image_B = np.asarray(Image.open(os.path.join(root, 'input_B.png')).convert("RGBA").resize((127, 127), Image.LANCZOS))

            assemble_explanation(image_A, image_B, difference_mask_2, difference_mask_1, 0.0, "I", os.path.join(root, 'explanation.png'))

            if(JUST_CARE_ABOUT_THE_SCORES):
                rmtree(os.path.join(root, 'attentions_A'))
                rmtree(os.path.join(root, 'attentions_B'))

            elapsed_time = time.time() - t0
            print("[INFO] ELAPSED TIME: %.2fs\n" % (elapsed_time))

            with open("times_by_parts.txt", "a") as file:
                file.write(str(elapsed_time) + "\n")'''

            difference_mask_aux = np.asarray(Image.open(os.path.join(root, 'attentions') + "/I.png").convert("RGBA"))
            difference_mask_1 = difference_mask_aux[64:64+128, :128, :]
            difference_mask_2 = difference_mask_aux[64:64+128, 128:, :]

            input_aux = np.asarray(Image.open(os.path.join(root, 'input.png')).convert("RGBA"))
            image_A = np.asarray(Image.fromarray(input_aux[64:64+128, :128, :].astype(np.uint8)).resize((127, 127), Image.LANCZOS).convert("RGBA"))
            image_B = np.asarray(Image.fromarray(input_aux[64:64+128, 128:, :].astype(np.uint8)).resize((127, 127), Image.LANCZOS).convert("RGBA"))

            assemble_explanation(image_A, image_B, difference_mask_2, difference_mask_1, 0.0, "I", os.path.join(root, 'explanation.png'))

            if(JUST_CARE_ABOUT_THE_SCORES):
                rmtree(os.path.join(root, 'attentions'))
            
            elapsed_time = time.time() - t0
            print("[INFO] ELAPSED TIME: %.2fs\n" % (elapsed_time))

            with open("times_by_parts.txt", "a") as file:
                file.write(str(elapsed_time) + "\n")

        # save the array version
        os.makedirs('../../visualization/collected', exist_ok=True)
        f_assign.savefig(os.path.join('../../visualization/collected', args.load+'.png'))

        print('Visualization finished!')

# main method
if __name__ == '__main__':
    main()
