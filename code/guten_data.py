import os
import tqdm
import numpy as np
import torch.utils.data
from torchvision.datasets import ImageFolder
from torchvision import transforms
import functools
import PIL
import re
import pdb


class StoryDataset(torch.utils.data.Dataset):
    def __init__(self, image_path,transform, videl_len = 4, is_train = True):
        self.dir_path = image_path
        self.descriptions = np.load(image_path +'/guten_dict.npy', allow_pickle=True, encoding="latin1").item()
        self.transforms = transform
        if is_train:
            self.srt = 0
            self.edn = 599
        else:
            self.srt = 599
            self.edn = 699
        self.video_len = videl_len

    def __getitem__(self, item):
        image = []
        des = []
        item = item + self.srt
        for i in range(self.video_len):
            v = valid_img_path('%s/images_grouped/%d_%d.jpg' % (self.dir_path, item, i))
            id = v.split('/')[-1]
            im = PIL.Image.open(v)
            image.append(valid_np_img(im))
            des.append(np.expand_dims(self.descriptions[id].astype(np.float32), axis = 0))

        # image is T x H x W x C
        # After transform, image is C x T x H x W    
        image_numpy = image
        image = self.transforms(image_numpy)

        des = np.concatenate(des, axis = 0)        
        des = torch.tensor(des)
        super_label = np.array([[0, 0], [0, 0], [0, 0], [0, 0]]) # TODO

        return {'images': image, 'description': des, 'label':super_label}

    def __len__(self):
        return self.edn - self.srt + 1


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, transform,video_len = 4, is_train = True):
        self.dir_path = image_path
        self.transforms = transform
        self.descriptions = np.load(image_path +'/guten_dict.npy', allow_pickle=True, encoding="latin1").item()
        self.transforms = transform
        if is_train:
            self.srt = 0
            self.edn = 599
        else:
            self.srt = 599
            self.edn = 699
        self.video_len = video_len

    def __getitem__(self, item):
        item = item + self.srt
        se = np.random.randint(0, self.video_len, 1)
        path = valid_img_path('%s/images_grouped/%d_%d.jpg' % (self.dir_path, item, se))        
        id = path.split('/')[-1]
        im = PIL.Image.open(path)
        image = valid_np_img(im)
        image = self.transforms(image)
        des = self.descriptions[id].astype(np.float32)

        content = []
        for i in range(self.video_len):
            v = valid_img_path('%s/images_grouped/%d_%d.jpg' % (self.dir_path, item, i))
            id = v.split('/')[-1]
            content.append(np.expand_dims(self.descriptions[id].astype(np.float32), axis = 0))                       
        content = np.concatenate(content, 0)
        content = torch.tensor(content)

        super_label = np.array([0, 0]) # TODO          

        return {'images': image, 'description': des, 'label':super_label, 'content': content}

    def __len__(self):
        return self.edn - self.srt + 1


def video_transform(video, image_transform):
    vid = []
    for im in video:
        vid.append(image_transform(im))

    vid = torch.stack(vid).permute(1, 0, 2, 3)

    return vid

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