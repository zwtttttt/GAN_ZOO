# -*- encoding: utf-8 -*-
'''
@File    :   dataset.py
@Time    :   2023/03/06 11:03:27
@Author  :   zwt 
@Version :   1.0
@Contact :   1030456532@qq.com
'''

# here put the import lib

import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

FEATURES = {
    "aqua": "0",
    "gray": "1",
    "green": "2",
    "orange": "3",
    "red": "4",
    "white": "5",
    "black": "6",
    "blonde": "7",
    "blue": "8",
    "brown": "9",
    "pink": "10",
    "purple": "11",
    "yellow": "12"
}


class Datasets(Dataset):
    def __init__(self, root) -> None:
        super().__init__()
        fts = pd.read_csv(os.path.join(root, 'tags.csv'), header=None)
        fts = fts.values
        imgs = [os.path.join(root, "images", f"{index}.jpg")
                for index in list(fts[:, 0])]
        colors = [[color.split(' ')[0], color.split(' ')[2]]
                  for color in list(fts[:, 1])]

        # build the label.
        labels = []
        for color in colors:
            feature = torch.zeros(28, dtype=torch.int8)
            feature[13], feature[-1] = 1, 1
            feature[int(FEATURES[color[0]])] = 1
            feature[int(FEATURES[color[1]]) + 14] = 1
            labels.append(feature)

        self.imgs = imgs
        self.labels = labels
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, index: int):
        img = Image.open(self.imgs[index])
        img = self.transforms(img)
        label = self.labels[index]
        return img, label


# if __name__ == "__main__":
#     dataset = Datasets("../data/data/extra_data/")

#     img, label = dataset.__getitem__(1)
#     print(img.shape, torch.mean(img), label)
