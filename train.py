from dsth import DSTH
import h5py
import os
import paddle

path = "../datasets/NWPU-RESISC45/train"

hashtagsH5 = h5py.File(os.path.join(path, "hashtags.hy"), 'r')
hashtags = hashtagsH5["hashtags"][()]
print(hashtags.shape)
dsth = DSTH(256, 256, 3, hashtags)
# 为模型训练做准备，设置优化器及其学习率，并将网络的参数传入优化器，设置损失函数和精度计算方式
dsth.prepare(optimizer=paddle.optimizer.Adam(learning_rate=0.001, parameters=dsth.parameters()), 
              loss=paddle.nn.CrossEntropyLoss(), 
              metrics=paddle.metric.Accuracy())
