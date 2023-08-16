import utils
import os
import h5py
import paddle
from matplotlib import pyplot as plt
from dataset import MyDataset


__PATH__ = "../datasets/NWPU-RESISC45/train"
file = os.path.join(__PATH__, "data.hy")
try:
    h5data = h5py.File(file, 'r')
except:
    raise IOError('Dataset not found. Please make sure the dataset was downloaded.')

train_image = h5data['imgae'][()]
train_label = h5data['label'][()]
train_set = MyDataset(train_image, train_label)

# 定义并初始化数据读取器
train_loader = paddle.io.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=1, drop_last=True)

# 调用 DataLoader 迭代读取数据
for batch_id, data in enumerate(train_loader()):
    images, labels = data
    print("batch_id: {}, 训练数据shape: {}, 标签数据shape: {}".format(batch_id, images.shape, labels))
    break

print(h5data['label'][()][0])
img = h5data['image'][()][0]
plt.imshow(img)
plt.show()


# # 调用内置vgg16模型
# vgg16 = paddle.vision.models.vgg16(pretrained=True)
# input = paddle.rand([5, 3, 224, 224])
# output = vgg16(input)
# print(output.shape)
