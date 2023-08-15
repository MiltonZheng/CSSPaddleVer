import utils
import os
import numpy as np
import math
from matplotlib import pyplot as plt
import cv2
import progressbar


__PATH__ = "../datasets/NWPU-RESISC45"

labels = ['airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach', 
          'bridge', 'chaparral', 'church', 'circular_farmland', 'cloud', 
          'commercial_area', 'dense_residential', 'desert', 'forest', 'freeway', 
          'golf_course', 'ground_track_field', 'harbor', 'industrial_area', 'intersection', 
          'island', 'lake', 'meadow', 'medium_residential', 'mobile_home_park', 
          'mountain', 'overpass', 'palace', 'parking_lot', 'railway', 
          'railway_station', 'rectangular_farmland', 'river', 'roundabout', 'runway', 
          'sea_ice', 'ship', 'snowberg', 'sparse_residential', 'stadium', 
          'storage_tank', 'tennis_court', 'terrace', 'thermal_power_station', 'wetland']


total_num = 31500
class_num = 45
num_per_class = 700
train_num = 25000
test_num = 6500
train_image = []
train_label = []
test_image = []
test_label = []


def read_1_image_n_label(index):
    # *标签序号，按照labels中的顺序来，这里单独用变量是因为还要在train/test_label里面保存
    label_num = math.floor(index / num_per_class)
    # *标签
    label_ = labels[label_num]
    # 图片在类别内的序号
    num = index % num_per_class
    # 拼接文件路径，注意，图片名称为“类别_编号.jpg”，其中编号为固定的3位，例如第1张会表示为001，所以要用zfill补0
    path = os.path.join(__PATH__,"imgs", label_, str(label_ + "_" + str(num + 1).zfill(3) + ".jpg"))
    curr_img = cv2.imread(path)
    # cv2.imread默认以bgr模式读取数据，所以如果要可视化需要对通道进行转换，这样才能保证图片正常显示
    curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)
    return curr_img, label_num, path


if __name__ == "__main__":
    # 生成一个随机访问序列
    order = np.arange(total_num)
    np.random.shuffle(order)
    p1 = progressbar.ProgressBar()
    print("reading training set...")
    for i in p1(order[:25000]):
        img, label, _ = read_1_image_n_label(i)
        train_image.append(img)
        train_label.append(label)

    train_image = np.reshape(train_image, [train_num, 256, 256, 3])
    train_label = np.reshape(train_label, [train_num])
    print("training set size:", train_image.shape, train_label.shape)
    print("storing data:")
    utils.cvt_2_h5py(train_image, train_label, os.path.join(__PATH__, "train"), [256, 256, 3])

    test_path = os.path.join(__PATH__, "test")
    if not os.path.exists(test_path):
        os.mkdir(test_path)
    # !测试集图片需要保存一个编号列表，用于测试阶段
    test_path_txt = open(os.path.join(test_path, "test_path.txt"), 'a')
    print("reading test set...")
    p2 = progressbar.ProgressBar()
    for i in p2(order[25000:31500]):
        img, label, path = read_1_image_n_label(i)
        test_image.append(img)
        test_label.append(label)
        test_path_txt.write(os.path.abspath(path) + "\n")
    test_path_txt.close()
    test_image = np.reshape(test_image, [test_num, 256, 256, 3])
    test_label = np.reshape(test_label, [test_num])
    print("test set size:", test_image.shape, test_label.shape)
    print("storing data:")
    utils.cvt_2_h5py(test_image, test_label, test_path, [256, 256, 3])
    