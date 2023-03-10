# -*- encoding: utf-8 -*-
'''
@File    :   dataset.py
@Time    :   2023/03/08 13:48:36
@Author  :   zwt 
@Version :   1.0
@Contact :   1030456532@qq.com
'''

# here put the import lib

import os
import random
import glob

from torch.utils.data.dataset import Dataset
import torchvision.transforms as transform
from PIL import Image
import cv2


def split_data(root):
    random.seed(0) # Get the same results form random.
    imgs = glob.glob(os.path.join(root, '*.png'))
    train_imgs, val_imgs = [], random.sample(imgs, int(0.2 * len(imgs)))
    for img in imgs:
        if img not in val_imgs: train_imgs.append(img)
    return train_imgs, val_imgs


class Datasets(Dataset):
    def __init__(self, imgs, size) -> None:
        super().__init__()
        self.imgs = imgs
        """
            transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

            RGB Mean: 0.5 , 0.5 , 0.5 均值: 数据的平均水平
            RGB Std: 0.5 , 0.5 , 0.5  方差: 数据的离散程度
        """
        self.transform = transform.Compose([
            transform.ToTensor(), # C, H, W [0, 1]
            transform.Resize((size, size)),
            transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self) -> int:
        return len(self.imgs)
    
    def __getitem__(self, index: int):
        img = cv2.imread(self.imgs[index])
        img = img[:, :, ::-1]
        real_img = Image.open(self.imgs[index].replace('.png', '.jpg'))
        return self.transform(img.copy()), self.transform(real_img)


if __name__ == '__main__':
    train, val = split_data("./data/base/")

    print(len(train), len(val))