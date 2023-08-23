from dsth import DSTH
from dsth import loss
import h5py
import os
import paddle
import utils
from paddle.vision.transforms import Normalize

path = "../datasets/NWPU-RESISC45/train"

batch_size = 16
dsth = DSTH(28, 28, 1, batch_size=16)
model = paddle.Model(dsth)
# 为模型训练做准备，设置优化器及其学习率，并将网络的参数传入优化器，设置损失函数和精度计算方式
model.prepare(optimizer=paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()), 
              loss=loss(), 
              metrics=paddle.metric.Accuracy())

# train_set = utils.build_trainset(path)

transform = Normalize(mean=[127.5], std=[127.5], data_format='CHW')
# 下载数据集并初始化 DataSet
train_set = paddle.vision.datasets.MNIST(mode='train', transform=transform)
# 启动模型训练，指定训练数据集，设置训练轮次，设置每次数据集计算的批次大小，设置日志格式
model.fit(train_set, 
          epochs=5, 
          batch_size=16,
          verbose=1)

