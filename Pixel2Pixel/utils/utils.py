# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2023/03/08 14:33:56
@Author  :   zwt 
@Version :   1.0
@Contact :   1030456532@qq.com
'''

# here put the import lib

import torchvision
from tqdm import tqdm
import torch
import os


def train_one_epoch(G, D, train_loader, optim_G, optim_D, writer, loss, device, plot_every, epoch, l1_loss):
    tq = tqdm(train_loader)
    D_loss, G_loss = 0, 0
    step = 0
    G.train()
    D.train()
    for index, data in enumerate(tq):
        input_img = data[0].to(device) # seg img.
        real_img = data[1].to(device)

        # train D.
        fake_img = G(input_img) # fake real img.
        D_fake_out = D(fake_img, input_img).squeeze() # fake img with real seg img.
        D_real_out = D(real_img, input_img).squeeze() # real img with real seg img.
        loss_D1 = loss(D_fake_out, torch.zeros(D_fake_out.size()).cuda()) # fake img to fake label.
        loss_D2 = loss(D_real_out, torch.ones(D_real_out.size()).cuda()) # true img to true label.
        loss_D = (loss_D1 + loss_D2) * 0.5

        optim_D.zero_grad()
        loss_D.backward()
        optim_D.step()

        # train G.
        fake_img = G(input_img) # fake img with real seg img.
        D_fake_out = D(fake_img, input_img).squeeze()
        loss_G1 = loss(D_fake_out, torch.ones(D_fake_out.size()).cuda()) # fake img to true label.
        loss_G2 = l1_loss(fake_img, real_img)
        loss_G = loss_G1 + loss_G2 * 100

        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        G_loss += loss_G
        D_loss += loss_D

        tq.desc = 'train_{} G_loss: {} D_loss: {}'.format(epoch, loss_G.item(), loss_D.item())

        if index % plot_every == 0:
            writer.add_images(tag='train_epoch_{}'.format(epoch), img_tensor=0.5 * (fake_img + 1), global_step=step)
            step += 1
        
    mean_G_loss = G_loss / len(train_loader)
    mean_D_loss = D_loss / len(train_loader)

    return mean_G_loss, mean_D_loss



@torch.no_grad()
def val(G, D, val_loader, loss, l1_loss, device, epoch):
    pd = tqdm(val_loader)
    loss_D, loss_G = 0, 0
    G.eval()
    D.eval()
    all_loss = 10000
    for idx, item in enumerate(pd):
        in_img = item[0].to(device)
        real_img = item[1].to(device)
        fake_img = G(in_img)
        D_fake_out = D(fake_img, in_img).squeeze()
        ls_D1 = loss(D_fake_out, torch.zeros(D_fake_out.size()).cuda())
        ls_D = ls_D1 * 0.5
        ls_G1 = loss(D_fake_out, torch.ones(D_fake_out.size()).cuda())
        ls_G2 = l1_loss(fake_img, real_img)
        ls_G = ls_G1 + ls_G2 * 100
        loss_G += ls_G
        loss_D += ls_D
        pd.desc = 'val_{}: G_loss:{} D_Loss:{}'.format(epoch, ls_G.item(), ls_D.item())
 
        # 保存最好的结果
        all_ls = ls_G + ls_D
        if all_ls < all_loss:
            all_loss = all_ls
            best_image = fake_img
    result_img = (best_image + 1) * 0.5
    if not os.path.exists('./results'):
        os.mkdir('./results')
 
    torchvision.utils.save_image(result_img, './results/val_epoch{}.jpg'.format(epoch))