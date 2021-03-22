import os, sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import argparse
from io import BytesIO
import multiprocessing
from functools import partial

from PIL import Image
import lmdb
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import functional as trans_fn
import numpy as np
from shutil import rmtree
from random import shuffle, choice

IMAGE_SIZE = 256

class DepthWiseDataset(Dataset): # custom Dataset class

    def __init__(self, root): # class constructor
        
        self.root = root
        self.images = list(filter(lambda x : x[0] != ".", os.listdir(self.root)))
        shuffle(self.images)

    def __len__(self): # retrieves the length of the dataset
        return(len(self.images))

    def __getitem__(self, idx): # retrieves a sample
        
        image = np.load(self.root + self.images[idx])
        
        return(image)

def resize_multiple(img, resample = Image.LANCZOS, quality = 100):
    
    buffer = BytesIO()
    img.save(buffer, format = "jpeg", quality = 100)
    val = buffer.getvalue()

    return val

def resize_worker(img_file, root, size, resample):

    i, file = img_file
    
    random_order = choice([0, 1])

    if(random_order == 0):
        file1 = np.load(root + file)[:,:,:3]
        file2 = np.load(root + file)[:,:,3:]
    
    else:
        file2 = np.load(root + file)[:,:,:3]
        file1 = np.load(root + file)[:,:,3:]
    
    img = Image.fromarray(file1.astype(np.uint8))
    img = img.resize((size[0], size[0]), Image.LANCZOS)
    out1 = resize_multiple(img, resample = resample)
    
    img = Image.fromarray(file2.astype(np.uint8))
    img = img.resize((size[0], size[0]), Image.LANCZOS)
    out2 = resize_multiple(img, resample = resample)
    
    out = [out1]
    out[0] += out2 

    return i, out

def prepare(env, root, dataset, n_worker, size = [IMAGE_SIZE], resample = Image.LANCZOS):

    resize_fn = partial(resize_worker, root = root, size = size, resample = resample)

    files = sorted(dataset.images, key = lambda x: x[0])
    files = [(idx, file) for idx, file in enumerate(files)]
    total = 0
    
    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs in tqdm(pool.imap_unordered(resize_fn, files)):
            for size_aux, img in zip(size, imgs):
                key = f"{size_aux}-{str(i).zfill(5)}".encode("utf-8")

                with env.begin(write = True) as txn:
                    txn.put(key, img)

            total += 1

        with env.begin(write = True) as txn:
            txn.put("length".encode("utf-8"), str(total).encode("utf-8"))

if(__name__ == "__main__"):

    parser = argparse.ArgumentParser(description = "Preprocess images for model training")
    parser.add_argument("--out", type = str, help = "filename of the result lmdb dataset")
    parser.add_argument(
        "--size",
        type = str,
        default = IMAGE_SIZE,
        help = "resolutions of images for the dataset",
    )
    parser.add_argument(
        "--n_worker",
        type = int,
        default = 4,
        help = "number of workers for preparing dataset",
    )
    parser.add_argument(
        "--resample",
        type = str,
        default = "lanczos",
        help = "resampling methods for resizing images",
    )
    parser.add_argument("path", type = str, help = "path to the image dataset")

    args = parser.parse_args()

    resample_map = {"lanczos": Image.LANCZOS, "bilinear": Image.BILINEAR}
    resample = resample_map[args.resample]

    size = [int(s.strip()) for s in args.size.split(",")]

    print(f"Make dataset of image size:", ", ".join(str(s) for s in size))

    imgset = DepthWiseDataset(root = args.path)

    if(os.path.exists("outputs/dataset")): rmtree("outputs/dataset")
    os.makedirs("outputs/dataset")

    with lmdb.open(args.out, map_size = 1024 ** 4, readahead=False) as env:
        prepare(env, args.path, imgset, args.n_worker, size = size, resample = resample)