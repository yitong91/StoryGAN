import os
import functools
import re
import pdb
import random
import pickle

import tqdm
import numpy as np
import torch.utils.data
from torchvision.datasets import ImageFolder
from torchvision import transforms
import PIL


class StoryDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, desc_path, transform, video_len=4, is_train=True):
        self.img_dir = img_dir
        self.descriptions = pickle.load(open(desc_path, "rb"))
        self.transforms = transform
        self.video_len = video_len

    def __getitem__(self, item):
        image = []
        des = []
        story_data = self.descriptions[item]
        for i in range(self.video_len):
            image_id, desc = story_data[i]
            img_path = valid_img_path('%s/%d.jpg' % (self.img_dir, image_id))
            im = PIL.Image.open(img_path)
            image.append(valid_np_img(im))
            des.append(np.expand_dims(desc.astype(np.float32), axis = 0))

        # image is T x H x W x C
        # After transform, image is C x T x H x W    
        image_numpy = image
        image = self.transforms(image_numpy)

        des = np.concatenate(des, axis = 0)        
        des = torch.tensor(des)
        super_label = np.array([[0, 0] * self.video_len]) # TODO
        return {'images': image, 'description': des, 'label':super_label}

    def __len__(self):
        return len(self.descriptions)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, desc_path, transform, video_len=4, is_train=True):
        self.img_dir = img_dir
        self.transforms = transform
        self.descriptions = pickle.load(open(desc_path, "rb"))
        self.transforms = transform
        self.video_len = video_len

    def __getitem__(self, item):
        story_data = self.descriptions[item]
        image_id, desc = random.choice(story_data)
        path = valid_img_path('%s/%d.jpg' % (self.img_dir, image_id))
        im = PIL.Image.open(path)
        image = valid_np_img(im)
        image = self.transforms(image)
        des = desc.astype(np.float32)

        content = []
        for i in range(self.video_len):
            image_id, desc = story_data[i]
            v = valid_img_path('%s/%d.jpg' % (self.img_dir, image_id))
            content.append(np.expand_dims(desc.astype(np.float32), axis = 0))                       
        content = np.concatenate(content, 0)
        content = torch.tensor(content)

        super_label = np.array([0, 0]) # TODO

        return {'images': image, 'description': des, 'label':super_label, 'content': content}

    def __len__(self):
        return len(self.descriptions)


def valid_img_path(img_path):
    if not os.path.exists(img_path):
        if os.path.exists(img_path.replace('.jpg', '.png')):
            img_path = img_path.replace('.jpg', '.png')
        elif os.path.exists(img_path.replace('.jpg', '.gif')):
            img_path = img_path.replace('.jpg', '.gif')
    return img_path

def valid_np_img(img):
    np_img = np.array(img)
    if len(np_img.shape) >= 3 and np_img.shape[2] != 3:
        return np.stack((np_img[:, :, 0],) * 3, axis=2)
    elif len(np_img.shape) == 2:
        return np.stack((np_img,) * 3, axis=-1)
    else:
        return np_img