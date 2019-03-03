from __future__ import print_function
import torch.backends.cudnn as cudnn
import torch
import torchvision.transforms as transforms
import PIL
import argparse
import os
import random
import sys
import pprint
import dateutil
import dateutil.tz
import numpy as np
import functools
import clevr_data as data
import pdb
dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

from miscc.config import cfg, cfg_from_file
from miscc.utils import mkdir_p
from trainer import GANTrainer




def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='./cfg/clevr.yml', type=str)
    parser.add_argument('--gpu',  dest='gpu_id', type=str, default='0')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    dir_path = '../clevr_dataset/'
    #args.cfg_file = './cfg/clevr.yml'
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)
    output_dir = './output/%s_%s/' % \
                 (cfg.DATASET_NAME, cfg.CONFIG_NAME)

    test_sample_save_dir = output_dir + 'test/'
    if not os.path.exists(test_sample_save_dir):
        os.makedirs(test_sample_save_dir)
    num_gpu = len(cfg.GPU_ID.split(','))
    n_channels = 3
    
    image_transforms = transforms.Compose([
        PIL.Image.fromarray,
        transforms.Resize( (cfg.IMSIZE, cfg.IMSIZE) ),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        lambda x: x[:n_channels, ::],
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def video_transform(video, image_transform):
        vid = []
        for im in video:
            vid.append(image_transform(im))

        vid = torch.stack(vid).permute(1, 0, 2, 3)

        return vid
    
    video_transforms = functools.partial(video_transform, image_transform=image_transforms)

    storydataset = data.StoryDataset(dir_path, video_transforms, cfg.VIDEO_LEN, True)
    imagedataset = data.ImageDataset(dir_path, image_transforms, cfg.VIDEO_LEN, True)
    testdataset = data.StoryDataset(dir_path, video_transforms, cfg.VIDEO_LEN, False)

    imageloader = torch.utils.data.DataLoader(
        imagedataset, batch_size=cfg.TRAIN.IM_BATCH_SIZE * num_gpu,
        drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))

    storyloader = torch.utils.data.DataLoader(
        storydataset, batch_size=cfg.TRAIN.ST_BATCH_SIZE * num_gpu,
        drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))


    testloader = torch.utils.data.DataLoader(
        testdataset, batch_size=24 * num_gpu,
        drop_last=True, shuffle=False, num_workers=int(cfg.WORKERS))

    algo = GANTrainer(output_dir, cfg.ST_WEIGHT, test_sample_save_dir)
    algo.train(imageloader, storyloader, testloader)
