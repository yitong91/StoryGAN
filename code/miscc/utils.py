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

def compute_discriminator_loss(type, netD, real_imgs, fake_imgs,
                               real_labels, fake_labels, real_catelabels,
                               conditions, gpus):
    if type == 'bce':
        return _compute_bce_discriminator_loss(netD, real_imgs, fake_imgs,
                               real_labels, fake_labels, real_catelabels,
                               conditions, gpus)
    elif type == 'wgan-gp':
        return _compute_wasserstein_discriminator_loss(netD, real_imgs, fake_imgs,
                               real_labels, fake_labels, real_catelabels,
                               conditions, gpus)

def _compute_bce_discriminator_loss(netD, real_imgs, fake_imgs,
                               real_labels, fake_labels, real_catelabels,
                               conditions, gpus):
    cate_criterion =nn.MultiLabelSoftMarginLoss()
    criterion = nn.BCELoss()

    ratio = 1.0    
    # gp_lambda = 0.00005
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

    # gradient penalty
    # epsilon = torch.rand([len(real_imgs)] + [1 for _ in range(len(real_imgs.shape)-1)], \
    #                         device='cuda', requires_grad=True)
    # grad = _get_gradient(netD, real_imgs, fake_imgs, epsilon).contiguous()
    # grad -= grad.min(1, keepdim=True)[0]
    # grad /= grad.max(1, keepdim=True)[0]
    # gp = _gradient_penalty(grad)

    if netD.get_uncond_logits is not None:
        real_logits = \
            nn.parallel.data_parallel(netD.get_uncond_logits,
                                      (real_features), gpus)
        fake_logits = \
            nn.parallel.data_parallel(netD.get_uncond_logits,
                                      (fake_features), gpus)
        uncond_errD_real = criterion(real_logits, real_labels)
        uncond_errD_fake = criterion(fake_logits, fake_labels)
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

def _compute_wasserstein_discriminator_loss(netD, real_imgs, fake_imgs,
                               real_labels, fake_labels, real_catelabels,
                               conditions, gpus):     
    ratio = 1.0
    gp_lambda = 10
        
    criterion = nn.BCELoss()
    bce_criterion = nn.BCELoss()
    cate_criterion =nn.MultiLabelSoftMarginLoss()

    batch_size = real_imgs.size(0)
    cond = conditions.detach()
    fake = fake_imgs.detach()
    real_features = nn.parallel.data_parallel(netD, (real_imgs), gpus)
    fake_features = nn.parallel.data_parallel(netD, (fake), gpus)

    # real pairs
    inputs = (real_features, cond)
    D_real = nn.parallel.data_parallel(netD.get_critic, inputs, gpus)
    real_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    errD_real = criterion(real_logits, real_labels)    

    # wrong pairs
    inputs = (real_features[:(batch_size-1)], cond[1:])
    D_wrong = nn.parallel.data_parallel(netD.get_critic, inputs, gpus)
    wrong_logits = \
        nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    errD_wrong = bce_criterion(wrong_logits, fake_labels[1:])    
    
    # fake pairs
    inputs = (fake_features, cond)
    D_fake = nn.parallel.data_parallel(netD.get_critic, inputs, gpus)
    fake_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    errD_fake = criterion(fake_logits, fake_labels)

    # gradient penalty
    epsilon = torch.rand([len(real_imgs)] + [1 for _ in range(len(real_imgs.shape)-1)], \
                            device='cuda', requires_grad=True)
    grad = _get_gradient(netD, real_imgs, fake_imgs, epsilon).contiguous()
    gp = _gradient_penalty(grad)

    # Wasserstein loss
    errD = (torch.mean(D_fake) + torch.mean(D_wrong)) * 0.5 \
            - torch.mean(D_real) \
            + gp_lambda * gp

    if netD.cate_classify is not None:
        cate_logits = nn.parallel.data_parallel(netD.cate_classify, real_features, gpus)
        cate_logits = cate_logits.squeeze()
        errD = errD + ratio * cate_criterion(cate_logits, real_catelabels)
        acc = accuracy_score(real_catelabels.cpu().data.numpy().astype('int32'), 
            (cate_logits.cpu().data.numpy() > 0.5).astype('int32'))

    return errD, errD_real.data, errD_wrong.data, errD_fake.data, 0

def _gradient_penalty(gradient):
    '''
    Original code from the Coursera GAN course.
    Return the gradient penalty, given a gradient.
    Given a batch of image gradients, you calculate the magnitude of each image's gradient
    and penalize the mean quadratic distance of each magnitude to 1.
    Parameters:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    Returns:
        penalty: the gradient penalty
    '''
    # Flatten the gradients so that each row captures one image
    gradient = gradient.view(len(gradient), -1)

    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)
    
    # Penalize the mean squared distance of the gradient norms from 1
    penalty = torch.mean(torch.pow(gradient_norm - 1, 2))
    return penalty

def _get_gradient(crit, real, fake, epsilon):
    '''
    Original code from the Coursera GAN course.
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        crit: the critic model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    '''
    # Mix the images together
    mixed_images = real * epsilon + fake * (1 - epsilon)
    mixed_scores = crit(mixed_images)
    
    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient    

def _triplet_loss(D_fake, D_real, D_wrong):    
    pos_dist = torch.sqrt(torch.square(D_real[1:] - D_fake[1:]))
    neg_dist = torch.sqrt(torch.square(D_real[1:] - D_wrong))

    dists = torch.cat((pos_dist.unsqueeze(1), neg_dist.unsqueeze(1)), axis=1)
    m = torch.max(dists)
    dists = dists - m
    log_sum_exp = m + torch.log(torch.sum(torch.exp(dists)))
    loss = -torch.mean(neg_dist) + torch.mean(log_sum_exp)
    return loss

def compute_generator_loss(type, netD, fake_imgs, real_labels, fake_catelabels, conditions, gpus):
    if type == 'bce':
        return _compute_bce_generator_loss(netD, fake_imgs, real_labels,
                                            fake_catelabels, conditions, gpus)
    elif type == 'wgan-gp':
        return _compute_wasserstein_generator_loss(netD, fake_imgs, real_labels,
                                                    fake_catelabels, conditions, gpus)
    elif type =='non-saturating':
        return _compute_non_saturating_generator_loss(netD, fake_imgs, real_labels,
                                                    fake_catelabels, conditions, gpus)

def _compute_bce_generator_loss(netD, fake_imgs, real_labels, fake_catelabels, conditions, gpus):
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

def _compute_wasserstein_generator_loss(netD, fake_imgs, real_labels, fake_catelabels, conditions, gpus):
    ratio = 0.4

    cate_criterion =nn.MultiLabelSoftMarginLoss()
    cond = conditions.detach()
    fake_features = nn.parallel.data_parallel(netD, (fake_imgs), gpus)

    # Wasserstein loss
    inputs = (fake_features, cond)
    D_fake = nn.parallel.data_parallel(netD.get_critic, inputs, gpus)
    errD_fake = -torch.mean(D_fake)

    if netD.cate_classify is not None:
        cate_logits = nn.parallel.data_parallel(netD.cate_classify, fake_features, gpus)
        cate_logits = cate_logits.squeeze()
        errD_fake = errD_fake + ratio * cate_criterion(cate_logits, fake_catelabels)
        acc = accuracy_score(fake_catelabels.cpu().data.numpy().astype('int32'), 
            (cate_logits.cpu().data.numpy() > 0.5).astype('int32'))
    return errD_fake, 0

def _compute_non_saturating_generator_loss(netD, fake_imgs, real_labels, fake_catelabels, conditions, gpus):
    ratio = 0.4
    criterion = nn.BCELoss()
    cate_criterion =nn.MultiLabelSoftMarginLoss()
    cond = conditions.detach()
    fake_features = nn.parallel.data_parallel(netD, (fake_imgs), gpus)

    # fake pairs
    inputs = (fake_features, cond)
    fake_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    errD_fake = torch.mean(-torch.log(fake_logits))

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
    return errD_fake, 0


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


def save_model(netG, netD_im, netD_st, epoch, iteration, model_dir):
    netG_snapshot = {
        'state_dict': netG.state_dict(),
        'epoch': epoch,
        'iteration': iteration
    }
    torch.save(
        netG_snapshot,
        '%s/netG_epoch_%d.pth' % (model_dir, epoch))

    netD_im_snapshot = {
        'state_dict': netD_im.state_dict(),
        'epoch': epoch,
        'iteration': iteration
    }    
    torch.save(
        netD_im_snapshot,
        '%s/netD_im_epoch_%d.pth' % (model_dir, epoch))

    netD_st_snapshot = {
        'state_dict': netD_st.state_dict(),
        'epoch': epoch,
        'iteration': iteration
    }            
    torch.save(
        netD_st_snapshot,
        '%s/netD_st_epoch_last_%d.pth' % (model_dir, epoch))
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
        break

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
        break


#############################
def valid_img_path(img_path):
    if not os.path.exists(img_path):
        if os.path.exists(img_path.replace('.jpg', '.png')):
            img_path = img_path.replace('.jpg', '.png')
        elif os.path.exists(img_path.replace('.jpg', '.gif')):
            img_path = img_path.replace('.jpg', '.gif')
    return img_path

def valid_np_img(img, image_id):
    np_img = np.array(img)
    if len(np_img.shape) >= 3 and np_img.shape[2] != 3:
        return np.stack((np_img[:, :, 0],) * 3, axis=2)
    elif len(np_img.shape) == 2:
        return np.stack((np_img,) * 3, axis=-1)
    else:
        return np_img

def video_transform(video, image_transform):
    vid = []
    for im in video:
        try:
            vid.append(image_transform(im))
        except Exception as err:
            print(err, "/", im.shape, "/", im)
            raise
    vid = torch.stack(vid).permute(1, 0, 2, 3)
    return vid        