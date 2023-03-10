# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2023/03/06 17:27:50
@Author  :   zwt 
@Version :   1.0
@Contact :   1030456532@qq.com
'''

# here put the import lib
import os
import torch
import numpy as np
import torchvision
from torch.autograd import Variable
from tqdm import tqdm
 
def train_one_epoch(epoch, D, G, D_optim, G_optim, train_loader, loss, input_dim, visable_every, writer):
    tq = tqdm(train_loader)

    all_loss, D_loss, G_loss = 0, 0, 0
    step = 0
    for index, data in enumerate(tq):
        img = data[0].to('cuda')
        label = data[1].to('cuda')

        D.train()
        G.train()

        f_label, r_label = Variable(torch.zeros((img.size(0), 1))).cuda(), Variable(torch.ones((img.size(0), 1))).cuda()
        noise = np.random.normal(0, 1, (2, img.size(0), input_dim)).astype(np.float32)
        noise = torch.from_numpy(noise).to('cuda')

        # train D 3 times.
        noise1 = Variable(noise[0], requires_grad=True)
        D_r = D(img, label)
        G_f1 = G(noise1, label).detach()
        D_f1 = D(G_f1, label)
        
        loss_D1 = loss(D_f1, f_label) # the loss to fake input (data from Generator with noise) true label
        loss_D2 = loss(D_r, r_label) # the loss to true input (data from img) fake label

        D_optim.zero_grad()
        loss_D = 0.5 * (loss_D1 + loss_D2) # sum of the D loss.
        loss_D.backward()
        D_optim.step()
        D_loss += loss_D

        # train G.
        if (index + 1) % 3 == 0:
            noise2 = Variable(noise[1], requires_grad=True)
            G_f2 = G(noise2, label)
            D_f2 = D(G_f2, label)
            loss_G = loss(D_f2, r_label) # the loss to fake input (data from Generator with noise) true label

            G_optim.zero_grad()
            loss_G.backward()
            G_optim.step()

            G_loss += loss_G

        try:
            tq.desc = 'G_loss : {}  D_loss : {}'.format(loss_G.item(), loss_D.item())
        except:
            tq.desc = 'D_loss : {}'.format(loss_D.item())
        
         #可视化训练效果 == 每(3*every)个batch显示一次结果
        if (index + 1) % visable_every == 0 and (index + 1) % 3 == 0:
            writer.add_images(tag='train_epoch{}'.format(epoch), img_tensor=(G_f2 +1)/ 2.0, global_step=step)
            step +=1
    
    mean_D_loss = D_loss / len(train_loader)
    mean_G_loss = G_loss / len(train_loader)
    return mean_D_loss , mean_G_loss



@torch.no_grad()
def eval_G(G, batch, x_dim, epoch):
    #生成特征c
    c = gene_C(batch).to('cuda')    #[batch,]
    # 随机生成噪声
    z = Variable(torch.normal(0,1,(batch,x_dim),dtype=torch.float32).cuda())
    G.eval()
    f_img = G(z,c)
 
    #将图像还原回原样
    result_img = (f_img + 1) / 2.0
    #保存图片
    if not os.path.exists('./results'):
        os.mkdir('./results')
    torchvision.utils.save_image(result_img,'./results/epoch{}_result.jpg'.format(epoch))

def gene_C(batch):
    #生成目标特征c
    c1 = [torch.eye(13, 14)] * (batch // 13)
    c2 = [torch.eye(13, 14)] * (batch // 13)
    c1 = torch.cat(c1, dim=0)
    c2 = torch.cat(c2, dim=0)
    c = torch.cat([c1, c2], dim=1)
    c3 = torch.zeros((batch % 13,28), dtype=torch.int8)
    c3[:, 1], c3[:, -1] = 1, 1
    c = torch.cat([c, c3], dim=0)
    c[:, 14], c[:, -1] = 1, 1
    return c