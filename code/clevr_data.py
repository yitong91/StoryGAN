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
        self.descriptions = np.load(image_path +'CLEVR_dict.npy', allow_pickle=True, encoding="latin1").item()
        self.transforms = transform
        if is_train:
            self.srt = 100
            self.edn = 172
        else:
            self.srt = 172
            self.edn = 196
        self.video_len = videl_len

    def __getitem__(self, item):
        label = []
        super_label = []
        image = []
        lists = []
        des = []
        item = item + self.srt
        for i in range(self.video_len):
            v = '%simages/CLEVR_new_%06d_%d.png' % (self.dir_path, item, i+1)
            id = v.split('/')[-1]
            im = PIL.Image.open(v)
            image.append( np.expand_dims(np.array(im), axis = 0) )   
            des.append(np.expand_dims(self.descriptions[id].astype(np.float32), axis = 0))
            l = des[-1].reshape(-1)
            label.append(l[i*18 + 3: i*18 + 11])
            super_label.append(l[i*18:i*18+15])

        label[0] = np.expand_dims(label[0], axis = 0)
        super_label[0] = np.expand_dims(super_label[0], axis = 0)
        for i in range(1,4):
            label[i] = label[i] + label[i-1]
            super_label[i] = super_label[i] + super_label[i-1]
            temp = label[i].reshape(-1)
            super_temp = super_label[i].reshape(-1)
            temp[temp>1] = 1
            super_temp[super_temp>1] = 1
            label[i] = np.expand_dims(temp, axis = 0)
            super_label[i] = np.expand_dims(super_temp, axis = 0)
        des = np.concatenate(des, axis = 0)
        image_numpy = np.concatenate(image, axis = 0)
        label = np.concatenate(label, axis = 0)
        super_label = np.concatenate(super_label, axis = 0)
        # image is T x H x W x C
        image = self.transforms(image_numpy)  
        # After transform, image is C x T x H x W
        des = torch.tensor(des)
        ## des is attribute, subs is encoded text description
        return {'images': image, 'description': des, 'label':super_label}

    def __len__(self):
        return self.edn - self.srt + 1


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, transform,video_len = 4, is_train = True):
        self.dir_path = image_path
        self.transforms = transform
        self.descriptions = np.load(image_path +'CLEVR_dict.npy', allow_pickle=True, encoding="latin1").item()
        self.transforms = transform
        if is_train:
            self.srt = 100
            self.edn = 172
        else:
            self.srt = 172
            self.edn = 196
        self.video_len = video_len

    def __getitem__(self, item):
        item = item + self.srt
        se = np.random.randint(1,self.video_len+1, 1)
        path = '%simages/CLEVR_new_%06d_%d.png' % (self.dir_path, item, se)
        id = 'CLEVR_new_%06d_%d.png' % (item, se)
        im = PIL.Image.open(path)
        image = np.array(im)
        image = self.transforms(image)
        des = self.descriptions[id].astype(np.float32)
        label = des[3:11]
        super_label = des[:15]
        content = []
        for i in range(self.video_len):
            v = '%simages/CLEVR_new_%06d_%d.png' % (self.dir_path, item, i+1)
            id = v.split('/')[-1]
            content.append(np.expand_dims(self.descriptions[id].astype(np.float32), axis = 0))

        for i in range(1,4):
            label = label + des[i*18 + 3: i*18 + 11]
            super_label = super_label + des[i*18:i*18+15]
        label = label.reshape(-1)
        super_label = super_label.reshape(-1)
        label[label>1] = 1
        super_label[super_label>1] = 1
        content = np.concatenate(content, 0)
        content = torch.tensor(content)
        ## des is attribute, subs is encoded text description
        return {'images': image, 'description': des, 'label':super_label, 'content': content}

    def __len__(self):
        return self.edn - self.srt + 1



def video_transform(video, image_transform):
    vid = []
    for im in video:
        vid.append(image_transform(im))

    vid = torch.stack(vid).permute(1, 0, 2, 3)

    return vid


