# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2023/03/06 10:42:02
@Author  :   zwt 
@Version :   1.0
@Contact :   1030456532@qq.com
'''

# here put the import lib



import argparse
import os

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from dataset.dataset import Datasets
from model.cGAN import Discriminator, Generator
from torch.utils.tensorboard import SummaryWriter

from utils.train import eval_G, train_one_epoch

def main(opt):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if not os.path.exists('./weights'): os.mkdir('./weights')

    train_dataset = Datasets(opt.dir_path)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch, shuffle=True)

    D = Discriminator(opt.c_dim).to(device)
    G = Generator(opt.x_dim, opt.c_dim).to(device)

    D_optim = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    G_optim = optim.Adam(G.parameters(), lr=0.002, betas=(0.5, 0.999))
    loss = nn.MSELoss()

    start = 0

    if opt.weights:
        ckpt = torch.load(opt.weights)
        D.load_state_dict(ckpt['D_model'])
        G.load_state_dict(ckpt['G_model'])
        try:
            start = ckpt['epoch'] + 1
        except: pass
    
    writer = SummaryWriter(log_dir='train_logs')

    for epoch in range(start, opt.epoches):
        D_mean_loss, G_mean_loss = train_one_epoch(
            opt.epoches,
            D,
            G,
            D_optim,
            G_optim,
            train_dataloader,
            loss,
            opt.x_dim,
            opt.every,
            writer
        )
    
        #绘制损失曲线
        writer.add_scalars('mean_loss',{
            'G_loss': G_mean_loss,
            'D_loss': D_mean_loss
        },epoch)
 
        #保存模型
        save_dict = {
            'D_model' : D.state_dict(),
            'G_model' : G.state_dict(),
            'epoch' : epoch
        }
        torch.save(save_dict,'./weights/CGAN_best1.pth')

        if (epoch+1) % 1 == 0:
            eval_G(G=G, batch=opt.batch, x_dim=opt.x_dim, epoch=epoch)

def parse_opt():
    arg=argparse.ArgumentParser()
    arg.add_argument('--batch',default=64,type=int)
    arg.add_argument('--epoches', default=100, type=int)
    arg.add_argument('--weights',default='',help='load weights')
    arg.add_argument('--x_dim',default=100,type=int,help='input noise length')
    arg.add_argument('--c_dim', default=28, type=int, help='feature length')
    arg.add_argument('--every', default=10, type=int, help='visible train result every 100 batch')
    arg.add_argument('--dir_path',default='',type=str,help='train data dir path')
    opt=arg.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    print(opt)
    main(opt)