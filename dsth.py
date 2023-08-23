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
        # # * 此处计算卷积层和池化层的输出大小
        # # * 具体计算随着池化层和卷积层的变化而变化
        # self.f_height = int((int((self.input_height-1)/2)-1)/2)
        # self.f_width = int((int((self.input_width-1)/2)-1)/2)
        
        self.features = nn.Sequential(
            nn.Conv2D(self.input_channel, 6, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2),
            nn.Conv2D(6, 16, 5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(400, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10),
        )
        
    def forward(self, x):
        features = self.features(x)
        return features

class loss(nn.Layer):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, label):
        # loss = paddle.nn.functional.pairwise_distance(x, paddle.cast(label, dtype='float32'))
        loss = paddle.nn.functional.cross_entropy(x, label)
        # loss = paddle.mean(loss)
        return loss



h, w, c = [28, 28, 1]
batch_size = 1
dsthModel = DSTH(h, w, c)
params_info = paddle.summary(dsthModel, (batch_size, c, 28, 28))
print(params_info)

transform = Normalize(mean=[127.5], std=[127.5], data_format='CHW')
# 下载数据集并初始化 DataSet
train_set = paddle.vision.datasets.MNIST(mode='train', transform=transform)

# 模型组网并初始化网络
model = paddle.Model(dsthModel)
# 为模型训练做准备，设置优化器及其学习率，并将网络的参数传入优化器，设置损失函数和精度计算方式
model.prepare(paddle.optimizer.Adam(parameters=model.parameters()), 
              loss(), 
              paddle.metric.Accuracy())

# train_set = utils.build_trainset(path)

# 启动模型训练，指定训练数据集，设置训练轮次，设置每次数据集计算的批次大小，设置日志格式
model.fit(train_set, epochs=5, batch_size=64, verbose=1)