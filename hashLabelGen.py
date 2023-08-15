import utils
import os
import h5py
import paddle

__PATH__ = "../datasets/NWPU-RESISC45/train"
file = os.path.join(__PATH__, "data.hy")
try:
    data = h5py.File(file, 'r')
except:
    raise IOError('Dataset not found. Please make sure the dataset was downloaded.')

# 调用内置vgg16模型
vgg16 = paddle.vision.models.vgg16(pretrained=True)
input = paddle.rand([5, 3, 224, 224])
output = vgg16(input)
print(output.shape)
