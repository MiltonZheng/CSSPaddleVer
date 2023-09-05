'''
Author: Milton13 2677454592@qq.com
Date: 2023-08-29 16:31:56
LastEditors: Milton13 2677454592@qq.com
LastEditTime: 2023-09-01 10:51:21
FilePath: \CSSPaddleVer\hashCodeGen.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import paddle
import utils
from tqdm import tqdm
from paddle.vision.transforms import Resize
import numpy as np
import os
import h5py

h, w, c = [64, 64, 3]
batch_size = 64
path = "../output/dsth"
dsthModel = paddle.jit.load(path)
dsthModel.eval()

path = "../datasets/NWPU-RESISC45/test"
train_set = utils.build_testset(path)
data_loader = paddle.io.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)
hashcodes = []
for batch_id, data in enumerate(tqdm(data_loader(), desc="generating hashcodes")):
    images, label = data
    images_t = paddle.zeros([images.shape[0], c, h, w])
    for i in range(images.shape[0]):
        images_t[i] = Resize((h, w))(images[i]).astype("float32")
    images = images_t
    hashcode = dsthModel(images)
    hashcode = hashcode > 0.5
    if batch_id == 0:
        hashcodes = hashcode
    else:
        hashcodes = paddle.concat([hashcodes, hashcode], axis=0)
hashcodes = np.asarray(hashcodes).astype('int')
print(hashcodes.shape)
sum = hashcodes.sum(axis=0)
print(sum)

hashcodes_int64 = []
for i in range(hashcodes.shape[0]):
    hashcode = 0
    for j in range(hashcodes.shape[1]):
        hashcode = (hashcode << 1) + hashcodes[i][j].astype('int64')
    hashcodes_int64.append(hashcode)
hashcodes = hashcodes_int64

f = h5py.File(os.path.join(path, 'hashcodes.hy'), 'w')
f['hashcodes'] = hashcodes
f.close()