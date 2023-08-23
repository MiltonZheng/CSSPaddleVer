from paddle import nn
import paddle
from paddle.vision.transforms import Resize
import numpy as np
from paddle.vision.transforms import Normalize

class DSTH(nn.Layer):
    def __init__(self, height = 256, width = 256, channel = 3):
        super().__init__()
        self.input_height = height
        self.input_width = width
        self.input_channel = channel
        # * 此处计算卷积层和池化层的输出大小
        # * 具体计算随着池化层和卷积层的变化而变化
        self.f_height = int((int((self.input_height-1)/2)-1)/2)
        self.f_width = int((int((self.input_width-1)/2)-1)/2)
        
        self.features = nn.Sequential(
            nn.Conv2D(self.input_channel, 32, 5, stride=1, padding=2),
            nn.MaxPool2D(kernel_size=3, stride=2),
            nn.Conv2D(32, 32, 5, stride=1, padding=2),
            nn.AvgPool2D(kernel_size=3, stride=2),
            nn.Conv2D(32, 64, 5, stride=1, padding=2),
            # nn.BatchNorm(64, momentum=0.9, epsilon=1e-5),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2304, 512),
            nn.Linear(512, 10),
            # nn.BatchNorm(10, momentum=0.9, epsilon=1e-5),
        )
        
    def forward(self, x):
        features = self.features(x)
        return features

class loss(nn.Layer):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, label):
        loss = paddle.nn.functional.cross_entropy(x, label)
        # loss = paddle.mean(loss)
        return loss



h, w, c = [28, 28, 1]
batch_size = 1
dsthModel = DSTH(h, w, c)
params_info = paddle.summary(dsthModel, (batch_size, c, 28, 28))
print(params_info)

path = "../datasets/NWPU-RESISC45/train"

transform = Normalize(mean=[127.5], std=[127.5], data_format='CHW')
# 下载数据集并初始化 DataSet
train_set = paddle.vision.datasets.MNIST(mode='train', transform=transform)

model = paddle.Model(dsthModel)
# 为模型训练做准备，设置优化器及其学习率，并将网络的参数传入优化器，设置损失函数和精度计算方式
model.prepare(optimizer=paddle.optimizer.Adam(parameters=model.parameters()), 
              loss=loss(), 
              metrics=paddle.metric.Accuracy())

# train_set = utils.build_trainset(path)

# 启动模型训练，指定训练数据集，设置训练轮次，设置每次数据集计算的批次大小，设置日志格式
model.fit(train_set, 
          epochs=5, 
          batch_size=64,
          verbose=1)