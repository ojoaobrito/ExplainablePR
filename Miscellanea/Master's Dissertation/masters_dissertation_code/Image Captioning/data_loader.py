import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO
from natsort import natsorted


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, images_path, captions, vocab, transform=None, mode="depthwise"):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            images_path: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.images_path = images_path
        self.images = natsorted(list(filter(lambda x : x[0] != ".", os.listdir(images_path))))
        self.mode = mode
        
        with open(captions, "rb") as file:
            self.captions = pickle.load(file)

        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        vocab = self.vocab
        caption = self.captions[index]

        # --------------------------------------------------------------------------
        # NOTE: 3 channels stuff
        # --------------------------------------------------------------------------
        if(self.mode == "side_by_side"):
            image = Image.open(self.images_path + self.images[index]).convert('RGB')

            if self.transform is not None:
                image = self.transform(image)
        # --------------------------------------------------------------------------

        # ----------------------------------------------------------------
        # NOTE: 6 channels stuff
        # ----------------------------------------------------------------
        elif(self.mode == "depthwise"):
            image_np = np.load(self.images_path + self.images[index])

            image_A = Image.fromarray(image_np[:, :, :3].astype(np.uint8))
            image_B = Image.fromarray(image_np[:, :, 3:].astype(np.uint8))

            if self.transform is not None:
                image_A = self.transform(image_A)
                image_B = self.transform(image_B)
            
            image = torch.cat((image_A, image_B), 0)
        # ----------------------------------------------------------------
        
        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.images)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

def get_loader(images_path, captions, vocab, transform, batch_size, shuffle, num_workers, mode):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(images_path=images_path,
                       captions=captions,
                       vocab=vocab,
                       transform=transform,
                       mode=mode)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn,
                                              drop_last=True)
    return data_loader