import numpy as np
import os
import h5py
import matplotlib.pyplot as plt


train_set = "../datasets/NWPU-RESISC45/train"

train_paths = open(os.path.join(train_set, "train_path.txt"), 'r')
features = h5py.File(os.path.join(train_set, "features.hy"), 'r')
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
island_agent = features[island0]
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

plt.scatter(indices, hamming_distances)
plt.show()