import os
import errno
import numpy as np
import PIL
from copy import deepcopy
from miscc.config import cfg
import pdb
from torch.nn import init
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
#############################
def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD


def compute_discriminator_loss(netD, real_imgs, fake_imgs,
                               real_labels, fake_labels, real_catelabels,
                               conditions, gpus):
    ratio = 1.0
    cate_criterion =nn.MultiLabelSoftMarginLoss()
    criterion = nn.BCELoss()
    batch_size = real_imgs.size(0)
    cond = conditions.detach()
    fake = fake_imgs.detach()
    real_features = nn.parallel.data_parallel(netD, (real_imgs), gpus)
    fake_features = nn.parallel.data_parallel(netD, (fake), gpus)
    # real pairs
    inputs = (real_features, cond)
    real_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    errD_real = criterion(real_logits, real_labels)
    # wrong pairs
    inputs = (real_features[:(batch_size-1)], cond[1:])
    wrong_logits = \
        nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    errD_wrong = criterion(wrong_logits, fake_labels[1:])
    # fake pairs
    inputs = (fake_features, cond)
    fake_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    errD_fake = criterion(fake_logits, fake_labels)

    if netD.get_uncond_logits is not None:
        real_logits = \
            nn.parallel.data_parallel(netD.get_uncond_logits,
                                      (real_features), gpus)
        fake_logits = \
            nn.parallel.data_parallel(netD.get_uncond_logits,
                                      (fake_features), gpus)
        uncond_errD_real = criterion(real_logits, real_labels)
        uncond_errD_fake = criterion(fake_logits, fake_labels)
        #
        errD = ((errD_real + uncond_errD_real) / 2. +
                (errD_fake + errD_wrong + uncond_errD_fake) / 3.)
        errD_real = (errD_real + uncond_errD_real) / 2.
        errD_fake = (errD_fake + uncond_errD_fake) / 2.
    else:
        errD = errD_real + (errD_fake + errD_wrong) * 0.5
    acc = 0
    if netD.cate_classify is not None:
        cate_logits = nn.parallel.data_parallel(netD.cate_classify, real_features, gpus)
        cate_logits = cate_logits.squeeze()
        errD = errD + ratio * cate_criterion(cate_logits, real_catelabels)
        acc = accuracy_score(real_catelabels.cpu().data.numpy().astype('int32'), 
            (cate_logits.cpu().data.numpy() > 0.5).astype('int32'))
    return errD, errD_real.data, errD_wrong.data, errD_fake.data, acc


def compute_generator_loss(netD, fake_imgs, real_labels, fake_catelabels, conditions, gpus):
    ratio = 0.4
    criterion = nn.BCELoss()
    cate_criterion =nn.MultiLabelSoftMarginLoss()
    cond = conditions.detach()
    fake_features = nn.parallel.data_parallel(netD, (fake_imgs), gpus)
    # fake pairs
    inputs = (fake_features, cond)
    fake_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    errD_fake = criterion(fake_logits, real_labels)
    if netD.get_uncond_logits is not None:
        fake_logits = \
            nn.parallel.data_parallel(netD.get_uncond_logits,
                                      (fake_features), gpus)
        uncond_errD_fake = criterion(fake_logits, real_labels)
        errD_fake += uncond_errD_fake
    acc = 0
    if netD.cate_classify is not None:
        cate_logits = nn.parallel.data_parallel(netD.cate_classify, fake_features, gpus)
        cate_logits = cate_logits.squeeze()
        errD_fake = errD_fake + ratio * cate_criterion(cate_logits, fake_catelabels)
        acc = accuracy_score(fake_catelabels.cpu().data.numpy().astype('int32'), 
            (cate_logits.cpu().data.numpy() > 0.5).astype('int32'))
    return errD_fake, acc


#############################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


#############################
def save_img_results(data_img, fake,epoch, image_dir):
    num = cfg.VIS_COUNT
    fake = fake[0:num]
    # data_img is changed to [0,1]
    if data_img is not None:
        data_img = data_img[0:num]
        vutils.save_image(
            data_img, '%s/real_samples_epoch_%03d.png' % 
            (image_dir, epoch), normalize=True)
        # fake.data is still [-1, 1]
        vutils.save_image(
            fake.data, '%s/fake_samples_epoch_%03d.png' %
            (image_dir, epoch), normalize=True)
    else:
        vutils.save_image(
            fake.data, '%s/lr_fake_samples_epoch_%03d.png' %
            (image_dir, epoch), normalize=True)

##########################\
def images_to_numpy(tensor):
    generated = tensor.data.cpu().numpy().transpose(1,2,0)
    generated[generated < -1] = -1
    generated[generated > 1] = 1
    generated = (generated + 1) / 2 * 255
    return generated.astype('uint8')

def save_story_results(ground_truth, images, epoch, image_dir, video_len = 5, test = False, ):
    all_images = []
    for i in range(images.shape[0]):
        all_images.append(vutils.make_grid(torch.transpose(images[i], 0,1), video_len))
    all_images= vutils.make_grid(all_images, 1)
    all_images = images_to_numpy(all_images)

    if ground_truth is not None:
        gts = []
        for i in range(ground_truth.shape[0]):
            gts.append(vutils.make_grid(torch.transpose(ground_truth[i], 0,1), video_len))
        gts = vutils.make_grid(gts, 1)
        gts = images_to_numpy(gts)
        all_images = np.concatenate([all_images, gts], axis = 1)

    output = PIL.Image.fromarray(all_images)
    if not test:
        output.save('%s/fake_samples_epoch_%03d.png' % (image_dir, epoch) )
    else:
        output.save('%s/test_samples_%03d.png' % (image_dir, epoch) )
    return 

def get_multi_acc(predict, real):
    predict = 1/(1+np.exp(-predict))
    correct = 0
    for i in range(predict.shape[0]):
        for j in range(predict.shape[1]):
            if real[i][j] == 1 and predict[i][j]>=0.5 :
                correct += 1
    acc = correct / float(np.sum(real))
    return acc


def save_model(netG, netD_im, netD_st, epoch, model_dir):
    torch.save(
        netG.state_dict(),
        '%s/netG_epoch_%d.pth' % (model_dir, epoch))
    torch.save(
        netD_im.state_dict(),
        '%s/netD_im_epoch_last.pth' % (model_dir))
    torch.save(
        netD_st.state_dict(),
        '%s/netD_st_epoch_last.pth' % (model_dir))
    print('Save G/D models')


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def save_test_samples(netG, dataloader, save_path):
    print('Generating Test Samples...')
    labels = []
    gen_images = []
    real_images = []
    for i, batch in enumerate(dataloader, 0):
        real_cpu = batch['images']
        motion_input = batch['description']
        content_input = batch['description']
        catelabel = batch['label']
        real_imgs = Variable(real_cpu)
        motion_input = Variable(motion_input)
        content_input = Variable(content_input)
        if cfg.CUDA:
            real_imgs = real_imgs.cuda()            
            motion_input = motion_input.cuda()
            content_input = content_input.cuda()
            catelabel = catelabel.cuda()

        _, fake, _,_,_,_ = netG.sample_videos(motion_input, content_input)
        save_story_results(real_cpu, fake, i, save_path)

    for i, batch in enumerate(dataloader, 0):
        if i>10:
            break
        real_cpu = batch['images']
        motion_input = batch['description']
        content_input = batch['description']
        catelabel = batch['label']
        real_imgs = Variable(real_cpu)
        motion_input = Variable(motion_input)
        content_input = Variable(content_input)
        if cfg.CUDA:
            real_imgs = real_imgs.cuda()            
            motion_input = motion_input.cuda()
            content_input = content_input.cuda()
            catelabel = catelabel.cuda()

        _, fake, _,_,_,_ = netG.sample_videos(motion_input, content_input)
        save_story_results(real_cpu, fake, i, save_path, 5, True)
   

            
