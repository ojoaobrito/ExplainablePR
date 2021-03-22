import sys

# prevent the creation of "__pycache__"
sys.dont_write_bytecode = True

import lmdb
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

JPEG_HEADER = b'\xff\xd8\xff\xe0\x00\x10JFIF'

class MultiResolutionDataset(Dataset):

    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        bytes_buffer = buffer.getvalue()

        aux = [JPEG_HEADER + i for i in bytes_buffer.split(JPEG_HEADER)][1:]

        first_image = np.asarray(Image.open(BytesIO(aux[0])))
        second_image = np.asarray(Image.open(BytesIO(aux[1])))
        concatenated = np.dstack((first_image, second_image))
        
        concatenated = self.transform(concatenated)

        return concatenated