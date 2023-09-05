'''
Author: MiltonZheng zheng12238@gmail.com
Date: 2023-08-23 16:05:06
LastEditors: MiltonZheng zheng12238@gmail.com
LastEditTime: 2023-08-25 11:43:02
FilePath: \CSSPaddleVer\dsth.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from paddle import nn
import paddle
from paddle.vision.transforms import Resize, Transpose, Normalize
import numpy as np
import utils
# from visualdl import LogWriter
from tqdm import tqdm

class DSTH(nn.Layer):
    def __init__(self, height = 32, width = 32, channel = 3):
        super().__init__()
        self.input_height = height
        self.input_width = width
        self.input_channel = channel
        # * we calculate the size of the feature maps after the pooling and convolution layers
        # * the calculation varies as the kernel size and stride length change
        self.f_height = int((int((self.input_height-1)/2)-1)/2)
        self.f_width = int((int((self.input_width-1)/2)-1)/2)
        
        self.features = nn.Sequential(
            nn.Conv2D(self.input_channel, 32, 5, stride=1, padding=2),
            nn.MaxPool2D(kernel_size=3, stride=2),
            nn.Conv2D(32, 32, 5, stride=1, padding=2),
            nn.AvgPool2D(kernel_size=3, stride=2),
            nn.Conv2D(32, 64, 5, stride=1, padding=2),
            nn.BatchNorm(64, momentum=0.9, epsilon=1e-5),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*self.f_height*self.f_width, 4096),
            # nn.Linear(4096, 512),
            # nn.Linear(512, 48)
        )
        self.slice = nn.Sequential(
            nn.Linear(256, 3),
            nn.BatchNorm(3, momentum=0.9, epsilon=1e-5)
        )
        
    @paddle.jit.to_static    
    def forward(self, x):
        features = self.features(x)
        # * sliceNet
        # * the output from previous layer, or input of this subnet, is divided equally into 16 parts
        # * each part is responsible for 3 dimensions of the final output
        # * so we get 16*3=48 dimensions
        output = self.slice(paddle.slice(features, [1], [0], [256]))
        for i in np.arange(1,16):
            feature_x = paddle.slice(features, [1], [256*i], [256*(i+1)])
            feature_x = self.slice(feature_x)
            output = paddle.concat([output, feature_x], axis=-1)
        return output

