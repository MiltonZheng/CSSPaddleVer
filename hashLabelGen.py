import utils
import os
import h5py
import paddle
from dataset import MyDataset
from tqdm import tqdm
import numpy as np


__PATH__ = "../datasets/NWPU-RESISC45/train"
file = os.path.join(__PATH__, "data.hy")
try:
    h5data = h5py.File(file, 'r')
except:
    raise IOError('Dataset not found. Please make sure the dataset was downloaded.')

train_image = h5data['image'][()]
train_label = h5data['label'][()]
train_set = MyDataset(train_image, train_label)

# 定义并初始化数据读取器
# !DataLoader会把数据打乱
train_loader = paddle.io.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=1, drop_last=False)
vgg16 = paddle.vision.models.vgg16(pretrained=True)
features = []
# 调用 DataLoader 迭代读取数据
for batch_id, data in enumerate(tqdm(train_loader(), desc="extract feature")):
    images, labels = data
    # *网络输入维度要求为[3,256,256]，而imread读取时存储为[256,256,3]，所以此处需要进行转置。
    # *但是2,3维顺序是否正确，或者有没有影响还未知
    # *图片输入网络前需要先转换为float32
    images = paddle.transpose(images.astype('float32'), perm=[0,3,1,2])
    feature = vgg16(images)
    # features.extend(feature)
    print("batch_id: {}, 训练数据shape: {}, 标签数据shape: {}".format(batch_id, images.shape, labels.shape))

# features = np.asarray(features, dtype = np.float32)
# print(features.shape)
# featuresh5py = h5py.File(os.path.join(__PATH__, 'features.hy'), 'w')
# featuresh5py.create_dataset("features", data=features)
# featuresh5py.close()

