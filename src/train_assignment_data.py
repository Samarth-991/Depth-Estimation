import argparse
import datetime
import os
import os.path as osp
import time

import pandas as pd
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from glob import glob
from data_loader.data_loader_assignment import CreateAssignemntDataset
from depth_model.loss import ssim
from depth_model.model import PTModel as Model
from utils.common import AverageMeter, DepthNorm, colorize
from utils.common import create_logger
from utils.data_transforms import get_training_augmentation as augmentation
from utils.data_transforms import pre_process as preprocessing


# learning rate decay
def update_lr(opt, lr):
    for param_group in opt.param_groups:
        param_group['lr'] = lr


def fine_tune(pretrined_model):
    # save layer names
    layer_names = []
    for idx, (name, param) in enumerate(pretrined_model.named_parameters()):
        layer_names.append(name)
    # Getting the depth layers for decoder
    # reverse layers
    layer_names.reverse()

    # learning rate
    lr = 1e-2
    lr_mult = 0.9
    # placeholder
    parameters = []

    # store params & learning rates
    for idx, name in enumerate(layer_names):
        # append layer parameters
        parameters += [{'params': [p for n, p in pretrined_model.named_parameters() if n == name and p.requires_grad],
                        'lr': lr}]
        # update learning rate
        lr *= lr_mult
    return parameters


def train(args):
    logger = create_logger()
    data_path = args.path
    model_path = args.outdir

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    child_rgb_files = glob(data_path + "/*/rgb/*")

    train_sample = int(len(child_rgb_files) * .80)
    train_rgb_images = child_rgb_files[: train_sample]
    train_depth_images = [fname.replace('rgb', 'depth') for fname in train_rgb_images]

    valid_rgb_images = child_rgb_files[train_sample:]
    valid_depth_images = [fname.replace('rgb', 'depth') for fname in valid_rgb_images]

    logger.info("Data Size {}".format(len(child_rgb_files)))
    logger.info(
        "Images in Training  dataset {}\t Images in Validation Dataset {}".format(len(train_rgb_images),
                                                                                  len(valid_rgb_images)))

    train_dataset = CreateAssignemntDataset(rgb_files=train_rgb_images,
                                            depth_files=train_depth_images,
                                            transform=None,
                                            process_image=preprocessing(),
                                            task='Training'
                                            )
    valid_dataset = CreateAssignemntDataset(rgb_files=valid_rgb_images,
                                            depth_files=valid_depth_images,
                                            transform=None,
                                            task='validation',
                                            process_image=preprocessing())

    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=1)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

    # Create model
    model = Model().float().to(device)

    if args.pretrained:
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint)
        parameters = fine_tune(pretrined_model=model)
    else:
        parameters = model.parameters()

    # Define Optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    if args.pretrained:
        logger.info("Loading model with pre-trained weights from {}".format(args.pretrained))
        torch.optim.Adam(parameters)
        # Loss
    l1_criterion = nn.L1Loss()

    # model Name
    model_name = args.model_name + '.ckpt'

    # start Training
    epoch_loss = 0.0
    for epoch in range(args.epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()
        N = len(train_loader)
        # switch to train
        model.train()
        end = time.time()

        for i, sample_image in enumerate(train_loader):
            image = sample_image['image'].to(device)
            depth = sample_image['depth'].to(device)
            depth_n = depth
            # Predict
            output = model(image)
            l_depth = l1_criterion(output, depth_n)
            l_ssim = torch.clamp((1 - ssim(output, depth_n, val_range=1000.0 / 10.0)) * 0.5, 0, 1)
            loss = (1.0 * l_ssim) + (0.1 * l_depth)
            # Update step
            optimizer.zero_grad()
            losses.update(loss.data.item(), image.size(0))
            loss.backward()
            optimizer.step()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val * (N - i))))
            # Log progress
            niter = epoch * N + i
            if i % 50 == 0:
                # Print to console
                logger.info('Epoch: [{0}][{1}/{2}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                            'ETA {eta}\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})'
                            .format(epoch + 1, i + 1, N, batch_time=batch_time, loss=losses, eta=eta))

        if epoch == 0:
            epoch_loss = losses.avg
            logger.info("Loss decreased from -inf. to {}".format(losses.avg))

        elif epoch_loss > losses.avg:
            logger.info("Loss decreased from {} to {}".format(epoch_loss, losses.avg))
            epoch_loss = losses.avg
            # Save the model checkpoint
            torch.save(model.state_dict(), osp.join(model_path, model_name))
            logger.info("saving model checkpoint {}".format(osp.join(model_path, model_name)))

    torch.save(model, os.path.join(model_path, args.model_name + '.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--path', type=str, default=None, help='data_path')
    parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--batchsize', default=4, type=int, help='batch size')
    parser.add_argument('--pretrained', default=None, type=str, help='pretrained network checkpoints ')
    parser.add_argument('--outdir', default='model', type=str, help='output directory for saving model')
    parser.add_argument('--model_name', default='depth_model2', type=str, help='Model name')
    args = parser.parse_args()
    train(args)
