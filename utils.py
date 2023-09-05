import numpy as np
from paddle.vision.transforms import Resize
import h5py
import os
from tqdm import tqdm
from dataset import MyDataset


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


def cvt_2_h5py(image, label, data_dir):
    """
    将图片和标签保存在同一个h5py文件中
    !(已更改)文件逻辑结构：先按照每张图片对应的内容和标签分为若干个group，每一个group中包含image和label数据
    *结构已更改为图片和标签分别存在两个dataset当中。
    :param image: 图像
    :param label: 标签
    :param data_dir: 存储路径
    :param shape: 图片形状
    :return:
    """
    # 拼接样本和标签
    # *灰度值只有256个取值，所以8位就够。
    # *标签此处也转换为了uint8，但是当类别多于256时不适合。
    image = image.astype(np.uint8)
    label = label.astype(np.uint8)

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    f = h5py.File(os.path.join(data_dir, 'data.hy'), 'w')
    f['image'] = image
    f['label'] = label
    f.close()
    return

def build_trainset(filepath):
    '''
    # *build a dataloader using paddlepaddle
    this one is different from the the one in hashLabelGen.py
    In hashLabelGen.py, we only need the images to extract features
    but we need both images and labels(hashtags) to train the model
    so we need to use the hashtags as pseudo labels here
    These two could be written in one file, but I think it's better to keep them separate
    So it's less confusing
    '''
    try:
        h5data = h5py.File(os.path.join(filepath, "data.hy"), 'r')
        hashtagsH5 = h5py.File(os.path.join(filepath, "hashtags.hy"), 'r')
    except:
        raise IOError('Dataset not found. Please make sure the dataset is stored in the right directory.')
    
    # *The model requires the input dimension to be [3, height, width](CHW)
    # *but imread reads images as [width, height, 3](HWC), so the data needs to be transposed
    train_image = h5data['image'][()]
    train_image = np.transpose(train_image, (0, 3, 1, 2))
    train_label = hashtagsH5['hashtags'][()]
    
    train_set = MyDataset(train_image, train_label)
    return train_set


def build_testset(filepath):
    try:
        h5data = h5py.File(os.path.join(filepath, "data.hy"), 'r')
    except:
        raise IOError('Dataset not found. Please make sure the dataset is stored in the right directory.')
    
    test_image = h5data['image'][()]
    test_image = np.transpose(test_image, (0, 3, 1, 2))
    test_label = h5data['label'][()]
    
    train_set = MyDataset(test_image, test_label)
    return train_set