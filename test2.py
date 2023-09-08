import os
import paddle
import numpy as np
from tqdm import tqdm
from paddle.io import DataLoader
from clip import tokenize, load_model
from paddle.vision.datasets import Cifar100
from sklearn.linear_model import LogisticRegression

# Load the model
model, transforms = load_model('ViT_B_32', pretrained=True)

# Load the dataset
train = Cifar100(mode='train', transform=transforms, backend='pil')
test = Cifar100(mode='test', transform=transforms, backend='pil')

# Get features
def get_features(dataset):
    all_features = []
    all_labels = []
    
    with paddle.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
            features = model.encode_image(images)
            all_features.append(features)
            all_labels.append(labels)

    return paddle.concat(all_features).numpy(), paddle.concat(all_labels).numpy()

# Calculate the image features
train_features, train_labels = get_features(train)
test_features, test_labels = get_features(test)

# Perform logistic regression
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=0)
classifier.fit(train_features, train_labels)

# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(np.float)) * 100.

# Print the result
print(f"Accuracy = {accuracy:.3f}")