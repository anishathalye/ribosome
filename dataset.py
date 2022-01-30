from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import numpy as np
import torch

import base64
import csv
import os


class HashToImage(Dataset):
    def __init__(self, hashes_csv, image_dir):
        self.image_dir = image_dir
        self.names_and_hashes = []
        with open(hashes_csv) as f:
            r = csv.reader(f)
            for line in r:
                path = line[0]
                h = np.array(list(base64.b64decode(line[1])), dtype=np.uint8)
                self.names_and_hashes.append((path, h))

    def __len__(self):
        return len(self.names_and_hashes)

    def __getitem__(self, idx):
        name, h = self.names_and_hashes[idx]
        img_path = os.path.join(self.image_dir, name)
        img = read_image(img_path, mode=ImageReadMode.RGB)
        return torch.tensor(h), img
