# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2023/03/08 14:21:06
@Author  :   zwt 
@Version :   1.0
@Contact :   1030456532@qq.com
'''

# here put the import lib

import argparse
import os

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from model.pixel2pixel import Discriminator_256, Generator_256
from torch.utils.tensorboard import SummaryWriter

from utils.dataset import Datasets, split_data
from utils.utils import train_one_epoch, val


def train(opt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not os.path.exists(opt.save_path): os.mkdir(opt.save_path)

    # load the dataset.
    train_dataset, val_dataset = split_data(opt.data_path)
    train_dataset = Datasets(train_dataset, opt.img_size)
    val_dataset = Datasets(val_dataset, opt.img_size)

    train_loader = DataLoader(train_dataset, opt.batch, shuffle=True, num_workers=opt.num_worker, drop_last=True)
    val_loader = DataLoader(val_dataset, opt.batch, shuffle=True, num_workers=opt.num_worker, drop_last=True)


    start = 0
    # init the models.
    G = Generator_256().to(device)
    D = Discriminator_256().to(device)

    if opt.weight != '':
        ckpt = torch.load(opt.weight)
        G.load_state_dict(ckpt['G_model'], strict=False)
        D.load_state_dict(ckpt['D_model'], strict=False)
        start = ckpt['epoch'] + 1
    
    # init optim and loss.
    optim_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optim_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    writer = SummaryWriter('train_logs')
    for epoch in range(start, opt.epoch):
        loss_G, loss_D = train_one_epoch(G, D, train_loader, optim_G, optim_D, writer, loss, device, opt.every, opt.epoch, l1_loss)

        writer.add_scalars(main_tag='train_loss', tag_scalar_dict={
            'loss_G': loss_G,
            'loss_D': loss_D
        }, global_step=epoch)

        val(G, D, val_loader, loss, l1_loss, device, epoch)

        torch.save({
            'G_model': G.state_dict(),
            'D_model': D.state_dict(),
            'epoch': epoch
        }, './weights/pix2pix_256.pth')

def cfg():
    parse = argparse.ArgumentParser()
    parse.add_argument('--batch', type=int, default=16)
    parse.add_argument('--epoch', type=int, default=200)
    parse.add_argument('--img_size', type=int, default=256)
    parse.add_argument('--data_path', type=str, default='../base', help='data root path')
    parse.add_argument('--weight', type=str, default='', help='load pre train weight')
    parse.add_argument('--save_path', type=str, default='./weights', help='weight save path')
    parse.add_argument('--num_worker', type=int, default=4)
    parse.add_argument('--every', type=int, default=2, help='plot train result every * iters')
    opt = parse.parse_args()
    return opt

if __name__ == '__main__':
    opt = cfg()
    print(opt)
    train(opt)