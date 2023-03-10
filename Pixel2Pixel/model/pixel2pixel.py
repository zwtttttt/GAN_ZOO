# -*- encoding: utf-8 -*-
'''
@File    :   pixel2pixel.py
@Time    :   2023/03/07 16:49:35
@Author  :   zwt 
@Version :   1.0
@Contact :   1030456532@qq.com
'''

# here put the import lib


import torch 
import torch.nn as nn
from collections import OrderedDict


class DownSample(nn.Module):
    def __init__(self, input_channel, output_channel) -> None:
        super().__init__()

        self.down = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(input_channel, output_channel, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(output_channel)
        )

    def forward(self, x):
        return self.down(x)

class UpSample(nn.Module):
    def __init__(self, input_channel, output_channel, drop_out=False) -> None:
        super().__init__()

        self.up = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(input_channel, output_channel, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(output_channel),
            # Identity just like a placeholder.
            nn.Dropout(0.5) if drop_out else nn.Identity()
        )

    def forward(self, x):
        return self.up(x)

class Generator_256(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        # down sample
        self.down_1 = nn.Conv2d(3, 64, 4, 2, 1) # b, 3, 256, 256 => b, 64, 128, 128
        for i in range(7):
            if i == 0:
                self.down_2 = DownSample(64, 128)  # [batch,64,128,128]=>[batch,128,64,64]
                self.down_3 = DownSample(128, 256)  # [batch,128,64,64]=>[batch,256,32,32]
                self.down_4 = DownSample(256, 512)  # [batch,256,32,32]=>[batch,512,16,16]
                self.down_5 = DownSample(512, 512)  # [batch,512,16,16]=>[batch,512,8,8]
                self.down_6 = DownSample(512, 512)  # [batch,512,8,8]=>[batch,512,4,4]
                self.down_7 = DownSample(512, 512)  # [batch,512,4,4]=>[batch,512,2,2]
                self.down_8 = DownSample(512, 512)  # [batch,512,2,2]=>[batch,512,1,1]
        
        # up sample
        for i in range(7):
            if i == 0:
                self.up_1 = UpSample(512, 512)  # [batch,512,1,1]=>[batch,512,2,2]
                self.up_2 = UpSample(1024, 512, drop_out=True)  # [batch,1024,2,2]=>[batch,512,4,4]
                self.up_3 = UpSample(1024, 512, drop_out=True)  # [batch,1024,4,4]=>[batch,512,8,8]
                self.up_4 = UpSample(1024, 512)  # [batch,1024,8,8]=>[batch,512,16,16]
                self.up_5 = UpSample(1024, 256)  # [batch,1024,16,16]=>[batch,256,32,32]
                self.up_6 = UpSample(512, 128)  # [batch,512,32,32]=>[batch,128,64,64]
                self.up_7 = UpSample(256, 64)  # [batch,256,64,64]=>[batch,64,128,128]

        self.last_Conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
 
        self.init_weight()

    def init_weight(self):
        for weight in self.modules():
            if isinstance(weight, nn.Conv2d):
                nn.init.kaiming_normal_(weight.weight, mode='fan_out')
                if weight.bias is not None: nn.init.zeros_(weight.bias)
            elif isinstance(weight, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(weight.weight, mode='fan_in')
            elif isinstance(weight, nn.BatchNorm2d):
                nn.init.ones_(weight.weight)
                nn.init.zeros_(weight.bias)
    
    def forward(self, x):
        # down
        down_1 = self.down_1(x)
        down_2 = self.down_2(down_1)
        down_3 = self.down_3(down_2)
        down_4 = self.down_4(down_3)
        down_5 = self.down_5(down_4)
        down_6 = self.down_6(down_5)
        down_7 = self.down_7(down_6)
        down_8 = self.down_8(down_7)

        # up
        up_1 = self.up_1(down_8)
        up_2 = self.up_2(torch.cat([up_1, down_7], dim=1))
        up_3 = self.up_3(torch.cat([up_2, down_6], dim=1))
        up_4 = self.up_4(torch.cat([up_3, down_5], dim=1))
        up_5 = self.up_5(torch.cat([up_4, down_4], dim=1))
        up_6 = self.up_6(torch.cat([up_5, down_3], dim=1))
        up_7 = self.up_7(torch.cat([up_6, down_2], dim=1))
        out = self.last_Conv(torch.cat([up_7, down_1], dim=1))
        return out

class Discriminator_256(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        def base_conv(input_channel, output_channel, stride):
            return nn.Sequential(
                nn.Conv2d(input_channel, output_channel, 4, stride, 1),
                nn.BatchNorm2d(output_channel),
                nn.LeakyReLU(0.2)
            )
        
        order_dict = OrderedDict()
        input_channel = 6
        output_channel = 64
        for i in range(4):
            order_dict.update({f"layer_{i + 1}": base_conv(input_channel, output_channel, 2 if i < 3 else 1)})
            input_channel = output_channel
            output_channel *= 2
        
        order_dict.update({"last_layer": nn.Conv2d(512, 1, 4, 1, 1)})
        self.D = nn.Sequential(order_dict)
    
    def forward(self, x1, x2):
        input = torch.cat([x1, x2], dim=1)
        return self.D(input)

if __name__ == '__main__':
    D = Discriminator_256()
    x1 = torch.normal(0, 1, (1, 3, 256, 256), dtype=torch.float32)
    x2 = torch.normal(0, 1, (1, 3, 256, 256), dtype=torch.float32)
    output = D(x1, x2) # patch gan.
    print(output.shape, sep='\n')

