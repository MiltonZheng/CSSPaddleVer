from dsth import DSTH
from dsth import loss
import h5py
import os
import paddle
import utils

path = "../datasets/NWPU-RESISC45/train"

batch_size = 16
dsth = DSTH(32, 32, 3, batch_size=16)
model = paddle.Model(dsth)
# 为模型训练做准备，设置优化器及其学习率，并将网络的参数传入优化器，设置损失函数和精度计算方式
model.prepare(optimizer=paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()), 
              loss=loss(), metrics=None)

train_set = utils.build_trainset(path)
# 启动模型训练，指定训练数据集，设置训练轮次，设置每次数据集计算的批次大小，设置日志格式
model.fit(train_set, 
          epochs=5, 
          batch_size=16,
          verbose=1)

