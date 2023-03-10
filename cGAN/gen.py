# -*- encoding: utf-8 -*-
'''
@File    :   gen.py
@Time    :   2023/03/07 14:41:51
@Author  :   zwt 
@Version :   1.0
@Contact :   1030456532@qq.com
'''

# here put the import lib



import sys
import torch
import torchvision
from torch.autograd import Variable
from model.cGAN import Generator

FEATURES = {
    "aqua": "0",
    "gray": "1",
    "green": "2",
    "orange": "3",
    "red": "4",
    "white": "5",
    "black": "6",
    "blonde": "7",
    "blue": "8",
    "brown": "9",
    "pink": "10",
    "purple": "11",
    "yellow": "12"
}

def gen_label(feature_hair, feature_eyes):
    label = torch.zeros(28, dtype=torch.int8)
    idx1 = int(FEATURES[feature_hair])
    idx2 = int(FEATURES[feature_eyes]) + 14
    label[13], label[-1] = 1, 1
    label[idx1], label[idx2] = 1, 1
    return label.unsqueeze(0)


if __name__ == '__main__':
    weights = sys.argv[1]
    hair = sys.argv[2]
    eyes = sys.argv[3]

    device='cuda' if torch.cuda.is_available() else 'cpu'
    G = Generator(100, 28).to(device)
    ckpt = torch.load(weights)
    G.load_state_dict(ckpt['G_model'])

    label = gen_label(hair, eyes).cuda()
    print(f"label: {label}, {label.shape}")
    # 随机生成噪声
    z = Variable(torch.normal(0, 1, (label.size(0), 100), dtype=torch.float32).cuda())

    G.eval()
    output = G(z,label)
    output = (output + 1) / 2.0
    
    torchvision.utils.save_image(output, f"{hair}_{eyes}.jpg")