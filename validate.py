import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
from clip import tokenize, load_model
from hashLabelGen import build_dataset
import paddle
from tqdm import tqdm
from paddle.vision.transforms import Resize


train_set = "../datasets/NWPU-RESISC45/train"


dataset = build_dataset(train_set)

def extract_features_w_clip(dataset):
    clip, transforms = load_model('ViT_B_32', pretrained=True)
    data_loader = paddle.io.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1, drop_last=False)
    features = []
    # call the DataLoader to read data iteratively
    for batch_id, data in enumerate(tqdm(data_loader(), desc="extracting features")):
        images, labels = data
        images_t = paddle.zeros([images.shape[0], 3, 224, 224])
        for i in range(images.shape[0]):
            images_t[i] = Resize((224, 224))(images[i]).astype("float32")
        images = images_t
        feature = clip.encode_image(images)
        feature = paddle.Tensor(feature).detach().cpu().numpy()
        features.extend(feature)

    # store the features as hy file
    features = np.asarray(features)
    print("feature size:{}".format(features.shape))
    features_h5 = h5py.File(os.path.join(train_set, 'clip_features.hy'), 'w')
    features_h5.create_dataset("features", data=features)
    features_h5.close()
    return features

def extract_text_feature(text):
    token = tokenize(text)
    clip, transforms = load_model('ViT_B_32', pretrained=True)
    text_feature = clip.encode_text(token)
    return np.asarray(text_feature)
    

train_paths = open(os.path.join(train_set, "train_path.txt"), 'r')
features = h5py.File(os.path.join(train_set, "clip_features.hy"), 'r')
features = features["features"][()]
hashtags = h5py.File(os.path.join(train_set, "hashtags.hy"), 'r')
hashtags = hashtags["hashtags"][()]
island_id = []
paths = []

# 统计类别为island的编号
count = 0
for line in train_paths.readlines():
    label = line.strip().split("/")[5]
    if label=='island':
        island_id.append(count)
    count += 1

# 统计island的哈希码并且转换为字符串形式
hashtag_strs = []
for i in range(hashtags.shape[0]):
    hashtag_str = "".join(str(i) for i in hashtags[i])
    hashtag_strs.append(hashtag_str)

# 计算island[0]与其他island的余弦距离
island0 = island_id[0]
# island_agent = features[island0]
island_agent = extract_text_feature("island").reshape(512)
hashtag_agent = hashtag_strs[island0]
norm2 = np.linalg.norm(island_agent)
distances = []
hamming_distances = []
indices = []
count = 0
for id in island_id:
    dot_product = np.dot(island_agent, features[id])
    norm1 = np.linalg.norm(features[id])
    distance = 1.0 - (dot_product / (norm1 * norm2))
    distances.append(distance)
    hamming_distance = bin(int(hashtag_agent,2)^int(hashtag_strs[id],2)).count('1')
    hamming_distances.append(hamming_distance)
    indices.append(count)
    count += 1

for i in range(features.shape[0]):
    if i in island_id:
        continue
    else:
        dot_product = np.dot(features[i], island_agent)
        norm1 = np.linalg.norm(features[i])
        distance = 1.0 - (dot_product / (norm1 * norm2))
        distances.append(distance)
        hamming_distance = bin(int(hashtag_agent,2)^int(hashtag_strs[i],2)).count('1')
        hamming_distances.append(hamming_distance)
        indices.append(count)
        count += 1
    if count > 1500:
        break

plt.scatter(indices, distances)
plt.savefig("./output/clip_features_validate.png")