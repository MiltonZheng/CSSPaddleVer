import numpy as np
import paddle
import h5py
import os
import progressbar


# 加载hy文件
def load_hy(path):
    """
    :param path: 文件路径
    :return: 提取hy文件文件内容,此时数据未经处理,仍然以group、dataset的形式组织
    """
    file = os.path.join(path)
    try:
        data = h5py.File(file, 'r')
    except IOError:
        print("Failed while loading file, please make sure the right file or format is selected")
    else:
        print("File loaded successfully")
        return data


def cvt_2_h5py(image, label, data_dir, shape=None):
    """
    将图片和标签保存在同一个h5py文件中
    文件逻辑结构：先按照每张图片对应的内容和标签分为若干个group
    每一个group中包含image和label数据
    :param image: 图像
    :param label: 标签
    :param data_dir: 存储路径
    :param shape: 图片形状
    :return:
    """
    # 拼接样本和标签
    image = image.astype(np.uint8)
    label = label.astype(np.uint8)

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    f = h5py.File(os.path.join(data_dir, 'data.hy'), 'w')
    p = progressbar.ProgressBar()
    for i in p(range(image.shape[0])):
        grp = f.create_group(str(i))
        # 把图片按指定shape进行reshape，其中order='F'表示优先按列的维度进行索引
        if shape:
            grp['image'] = np.reshape(image[i], shape, order='F')
        else:
            grp['image'] = image[i]
        grp['label'] = label[i]
    f.close()
    return