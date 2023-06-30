import argparse
import datetime
import os
import os.path as osp
import time
from tqdm import tqdm
import numpy
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.utils as utils
import torchvision.utils as vutils
from data_loader.data_creation import CreateDataset
from depth_model.loss import ssim
from depth_model.model import PTModel as Model
from utils.common import AverageMeter, DepthNorm ,colorize
from utils.data_transforms import get_training_augmentation as augmentation
from utils.data_transforms import pre_process as preprocessing
from utils.common import create_logger


def train(args):
    logger = create_logger()

    data_path = "/mnt/e/Personal/Samarth/repository/RGB_to_3D_conversion/Data"
    model_path = "/mnt/e/Personal/Samarth/repository/RGB_to_3D_conversion/model"

    train_df = pd.read_csv(os.path.join(data_path, 'data/nyu2_train.csv'))
    train_df.columns = ['RGB_images', 'Depth_images']

    val_df = pd.read_csv(os.path.join(data_path, 'data/nyu2_test.csv'))
    val_df.columns = ['RGB_images', 'Depth_images']

    ## Sample Data to avoid training for long time
    train_df = train_df.sample(1000)
    val_df = val_df.sample(100)

    batch_size = args.batchsize
    epochs = args.epochs

    rgb_image_files = [osp.join(data_path, filename) for filename in train_df['RGB_images'].to_list()]
    depth_image_files = [osp.join(data_path, filename) for filename in train_df['Depth_images'].to_list()]

    val_rgb_files = [osp.join(data_path, filename) for filename in val_df['RGB_images'].to_list()]
    val_depth_files = [osp.join(data_path, filename) for filename in val_df['Depth_images'].to_list()]

    train_dataset = CreateDataset(rgb_files=rgb_image_files,
                                  depth_files=depth_image_files,
                                  transform=augmentation(),
                                  process_image=preprocessing(),
                                  task='Training'
                                  )

    valid_dataset = CreateDataset(rgb_files=val_rgb_files,
                                  depth_files=val_depth_files,
                                  transform=None,
                                  task='validation',
                                  process_image=preprocessing())

    logger.info("Loaded {} images for training ".format(len(train_dataset)))
    logger.info("Loaded {} images for validation".format(len(valid_dataset)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

    # Create model
    if torch.cuda.is_available():
        model = Model().cuda()
    else:
        model = Model()

    # Define Optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # Loss
    l1_criterion = nn.L1Loss()

    # start Training
    epoch_loss = 0.0
    for epoch in range(epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()
        N = len(train_loader)
        # switch to train
        model.train()
        end = time.time()

        for i, sample_image in tqdm(enumerate(train_loader),total=N):
            optimizer.zero_grad()
            image = torch.autograd.Variable(sample_image['image'].cuda())
            depth = torch.autograd.Variable(sample_image['depth'].cuda())
            depth_n = DepthNorm(depth)
            # Predict
            output = model(image)
            l_depth = l1_criterion(output, depth_n)
            l_ssim = torch.clamp((1 - ssim(output, depth_n, val_range=1000.0 / 10.0)) * 0.5, 0, 1)
            loss = (1.0 * l_ssim) + (0.1 * l_depth)
            # Update step
            losses.update(loss.data.item(), image.size(0))
            loss.backward()
            optimizer.step()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val * (N - i))))
            # Log progress
            niter = epoch * N + i
            if i % 30 == 0:
                # Print to console
                logger.info('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                      'ETA {eta}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'
                      .format(epoch + 1, i + 1, N, batch_time=batch_time, loss=losses, eta=eta))
            if i + 1 % 10 == 0:
                logger.info("logging progress ")
                LogProgress(model, valid_loader, niter)
        if epoch == 0:
            epoch_loss = losses.avg
            torch.save(model, os.path.join(model_path,'depth_model.pt'))
        elif epoch_loss < losses.avg:
            epoch_loss = losses.avg
            logger.info("Loss decreased from {} to {}".format(epoch_loss, losses.avg))
            torch.save(model, os.path.join(model_path,'depth_model.pt'))

    model_name = 'final_model.pt'
    torch.save(osp.join(data_path, model_name))


def LogProgress(model, val_loader, niter,out_dir='model/runs'):
    model.eval()
    sequential = val_loader
    sample_batch = next(iter(sequential))
    image = torch.autograd.Variable(sample_batch['image'].cuda())
    depth = torch.autograd.Variable(sample_batch['depth'].cuda(non_blocking=True))
    output = DepthNorm(model(image))
    predicted_image  = colorize(vutils.make_grid(output.data, nrow=6, normalize=False))
    print(predicted_image,type(predicted_image))
    if not out_dir:
        os.makedirs(out_dir)
    del image
    del depth
    del output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--path', type=str, default=None, help='data_path')
    parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--batchsize', default=4, type=int, help='batch size')
    parser.add_argument('--outdir',default='model',type=str,help='output directory')
    args = parser.parse_args()
    train(args)
