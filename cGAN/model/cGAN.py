# -*- encoding: utf-8 -*-
'''
@File    :   cGAN.py
@Time    :   2023/03/06 15:35:04
@Author  :   zwt 
@Version :   1.0
@Contact :   1030456532@qq.com
'''

# here put the import lib
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Generator(nn.Module):
    """c-GAN Generator class.

    Attributes:
        x_dim: The noise dim that we input here, in this code is 100.
        c_dim: The label dim that we input here, in this code is 28.
    
    """
    def __init__(self, x_dim, c_dim) -> None:
        super().__init__()

        def base_conv(input_channel, output_channel, stride):
            padding = stride // 2
            return nn.Sequential(
                nn.ConvTranspose2d(input_channel, output_channel, 4, stride, padding),
                nn.BatchNorm2d(output_channel),
                nn.ReLU(inplace=True)
            )
        
        dim = x_dim + c_dim
        self.G = nn.Sequential(
            # input: [batch, 128, 1, 1]
            base_conv(dim, dim * 2, 1), # batch, 256, 4, 4
            base_conv(dim * 2, dim * 4, 2), # batch, 512, 8, 8
            base_conv(dim * 4, dim * 2, 2), # batch, 256, 16, 16
            base_conv(dim * 2, dim, 2), # batch, 128, 32, 32
            nn.ConvTranspose2d(dim, 3, 4, 2, 1) # batch, 3, 64, 64
        )

    def forward(self, nosie, label):
        # noise: [batch, x_dim]
        # label: [batch, c_dim]
        input = torch.cat([nosie, label], 1) # [batch, c_dim + x_dim]
        input = input.view(input.size(0), input.size(1), 1, 1) # [batch, x_dim + c_dim, 1, 1]
        output = self.G(input) # [batch, 3, 64, 64]
        return output


class Discriminator(nn.Module):
    
    def __init__(self, c_dim) -> None:
        super().__init__()

        def base_conv(input_channel, output_channel, kernel_size, stride):
            padding = stride // 2
            return nn.Sequential(
                weight_norm(nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding)),
                nn.BatchNorm2d(output_channel),
                nn.LeakyReLU(0.2, inplace=True)
            )
        
        self.D = nn.Sequential(
            # input: [batch, 3 + c_dim, 64, 64]
            base_conv(3 + c_dim, 64, 3, 2), # bacth, 64, 32, 32
            base_conv(64, 128, 3, 2), # batch, 128, 16,16
            base_conv(128, 256, 3, 2), # batch, 256, 8, 8
            base_conv(256, 256 ,3, 2), # bacth, 256, 4, 4
            nn.AvgPool2d(kernel_size=4), # batch, 256, 1, 1
        )

        self.linear = weight_norm(nn.Linear(256, 1))

    def forward(self, input, label):
        label = label.view(label.size(0), label.size(1), 1, 1) * torch.ones(label.size(0), label.size(1), input.size(2), input.size(3), dtype=torch.int8, device='cuda')
        input = torch.cat([input, label], 1)
        output = self.D(input)
        output = output.flatten(1)
        output = self.linear(output)
        return output


# if __name__ == "__main__":
#     a = torch.ones((4, 6), dtype=torch.int8)
#     b = torch.ones((4,2), dtype=torch.int8)
#     c = torch.cat([a, b], 1)
#     d = c.view(c.size(0), c.size(1), 1, 1)
#     print(a, b, c, d, sep='\n')