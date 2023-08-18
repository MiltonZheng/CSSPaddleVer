'''
Author: MiltonZheng zheng12238@gmail.com
Date: 2023-08-18 10:25:31
LastEditors: MiltonZheng zheng12238@gmail.com
LastEditTime: 2023-08-18 15:53:19
FilePath: \CSSPaddleVer\hashLabelGen.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import h5py
import paddle
from dataset import MyDataset
from tqdm import tqdm
import numpy as np
from sklearn import decomposition
from sklearn import manifold


def build_data_loader(filepath):
    '''
    build a dataloader using paddlepaddle
    '''
    file = os.path.join(filepath, "data.hy")
    try:
        h5data = h5py.File(file, 'r')
    except:
        raise IOError('Dataset not found. Please make sure the dataset is stored in the right directory.')

    train_image = h5data['image'][()]
    train_label = h5data['label'][()]
    train_set = MyDataset(train_image, train_label)

    # define & initialize the data loader
    train_loader = paddle.io.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=1, drop_last=False)
    return train_loader


def extract_feature(data_loader, filepath):
    '''
    extract and store features using vgg16
    '''    
    vgg16 = paddle.vision.models.vgg16(pretrained=True)
    features = []
    # call the DataLoader to read data iteratively
    for batch_id, data in enumerate(tqdm(data_loader(), desc="extracting feature")):
        images, labels = data
        # *The model requires the input dimension to be [3, height, width]
        # *but imread reads images as [width, height, 3], so the data needs to be transposed
        images = paddle.transpose(images.astype('float32'), perm=[0,3,1,2])
        feature = vgg16(images)
        # !remember to move the extracted features from gpu memory to main memory or gpu memory will run out
        feature = paddle.Tensor(feature).detach().cpu().numpy()
        features.extend(feature)

    # store the features as hy file
    features = np.asarray(features)
    print("feature sieze:{}".format(features.shape))
    featuresh5py = h5py.File(os.path.join(filepath, 'features.hy'), 'w')
    featuresh5py.create_dataset("features", data=features)
    featuresh5py.close()

def create_hashtags(filepath, bits=48, nei=10):
    featuresfile = h5py.File(os.path.join(filepath, 'features.hy'), 'r')
    features = featuresfile["features"][()]
    print("features size:{}".format(features.shape))
    print("decompose features...")
    features_pca = manifold.SpectralEmbedding(n_components=bits, n_neighbors=nei).fit_transform(features)
    hashtags = features_pca >= 0
    hashtags = hashtags.astype(np.int32)
    print(hashtags.shape)

if __name__ == "__main__":
    path = "../datasets/NWPU-RESISC45/train"
    # data_loader = build_data_loader(filepath=path)
    # extract_feature(data_loader, filepath=path)
    create_hashtags(filepath=path)
    