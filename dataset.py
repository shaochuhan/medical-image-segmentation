import os
from glob import glob

import PIL.Image as Image
# from skimage.io import imread
import cv2
import numpy as np
import torch.utils.data as data
from sklearn.model_selection import train_test_split


# import imageio


class ProstateDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.train_root = r"ProstateData/train"
        self.val_root = r"ProstateData/train"
        self.test_root = self.val_root
        self.pics, self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            root = self.train_root
        if self.state == 'val':
            root = self.val_root
        if self.state == 'test':
            root = self.test_root

        train_path = os.path.join(root, "imagesTr")
        label_path = os.path.join(root, "labelsTr")

        pics = []
        masks = []
        n = len(os.listdir(train_path))
        for i in range(n):
            img = os.path.join(train_path, "%03d.png" % i)
            mask = os.path.join(label_path, "%03d.png" % i)
            pics.append(img)
            masks.append(mask)
            # imgs.append((img, mask))
        return pics, masks

    def __getitem__(self, index):
        cv2.ocl.setUseOpenCL(False)  # 设置opencv不使用多进程运行，但这句命令只在本作用域有效。
        cv2.setNumThreads(0)  # 设置opencv不使用多进程运行，但这句命令只在本作用域有效。
        # x_path, y_path = self.imgs[index]
        x_path = self.pics[index]
        y_path = self.masks[index]

        origin_x = Image.open(x_path).convert("RGB")
        # origin_y = Image.open(y_path)

        # origin_x = cv2.cvtColor(np.array(origin_x), cv2.COLOR_RGB2BGR)
        # origin_y = cv2.cvtColor(np.array(origin_y), cv2.COLOR_BGR2GRAY)
        # origin_x = cv2.imread(x_path)
        origin_y = cv2.imread(y_path, cv2.COLOR_BGR2GRAY)

        if self.transform is not None:
            img_x = self.transform(origin_x)
        if self.target_transform is not None:
            img_y = self.target_transform(origin_y)
        return img_x, img_y, x_path, y_path

    def __len__(self):
        return len(self.pics)
