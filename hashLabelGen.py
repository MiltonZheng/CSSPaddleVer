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
    raise IOError('Dataset not found. Please make sure the dataset is stored in the right directory.')

train_image = h5data['image'][()]
train_label = h5data['label'][()]
train_set = MyDataset(train_image, train_label)

# define & initialize the data loader
train_loader = paddle.io.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=1, drop_last=False)

vgg16 = paddle.vision.models.vgg16(pretrained=True)
features = []
# call the DataLoader to read data iteratively
for batch_id, data in enumerate(tqdm(train_loader(), desc="extracting feature")):
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
featuresh5py = h5py.File(os.path.join(__PATH__, 'features.hy'), 'w')
featuresh5py.create_dataset("features", data=features)
featuresh5py.close()


