# -*- encoding: utf-8 -*-
'''
@File    :   cycleGAN.py
@Time    :   2023/03/10 13:14:04
@Author  :   zwt 
@Version :   1.0
@Contact :   1030456532@qq.com
'''

# here put the import lib

from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F

class RedisualBlock(nn.Module):
    """
    Don't change the input shape.
    """
    def __init__(self, input_channel) -> None:
        super().__init__()

        conv_block = [
            # Use the boundary as the axis of symmetry reflection to padding it.
            # 
            #                           5 4 5 6 5
            # 1 2 3      2 1 2 3 2      2 1 2 3 2
            # 4 5 6  =>  5 4 5 6 5  =>  5 4 5 6 5
            # 7 8 9      8 7 8 9 8      8 7 8 9 8
            #                           5 4 5 6 5
            # 
            nn.ReflectionPad2d(1), 
            nn.Conv2d(input_channel, input_channel, 3),
            # ** Is more suit for style transform missions (cycle gan style transform). **
            nn.InstanceNorm2d(input_channel),
            nn.ReLU(inplace=True), 
            nn.ReflectionPad2d(1), 
            nn.Conv2d(input_channel, input_channel, 3), 
            nn.InstanceNorm2d(input_channel)
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_channel, output_channel, redisual_block_nums=9) -> None:
        super().__init__()

        # ReflectionPad2d 3 with kernel_size 7. Then the output shape will not changed.
        # input.
        model = [
            nn.ReflectionPad2d(3), 
            nn.Conv2d(input_channel, 64, 7), 
            nn.InstanceNorm2d(64), 
            nn.ReLU(inplace=True)
        ]

        # down sample.
        input_channel = 64
        for _ in range(2):
            model += [
                nn.Conv2d(input_channel, input_channel * 2, 3, stride=2, padding=1), 
                nn.InstanceNorm2d(input_channel * 2), 
                nn.ReLU(inplace=True)
            ]
            input_channel = input_channel * 2
        
        # redisual block.
        for _ in range(redisual_block_nums):
            model += [RedisualBlock(input_channel)]

        # up sample.
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(input_channel, input_channel // 2, 3, stride=2, padding=1, output_padding=1), 
                nn.InstanceNorm2d(input_channel // 2),
                nn.ReLU(inplace=True)
            ]
            input_channel = input_channel // 2
        
        # output.
        """
        saturated neurons: sigmoid , tanh. etc

        one-sided saturations: relu , leaky relu. etc

            1. solve the vanishing gradients problems.
        """
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_channel, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

if __name__ == '__main__':
    G = Generator(3, 32)
    R = RedisualBlock(3)
    print(G)
    print(R)

    import torch
    input = torch.zeros((4, 3, 256, 256))
    output = G(input)
    r_output = R(input)
    print(f"input.shape: {input.shape} output.shape: {output.shape} r: {r_output.shape}")