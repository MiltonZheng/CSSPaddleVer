from paddle import nn
import paddle
from paddle.vision.transforms import Resize
import numpy as np
from paddle.vision.transforms import Normalize

class TestNet(nn.Layer):
    def __init__(self, height = 256, width = 256, channel = 3):
        super().__init__()
        self.input_height = height
        self.input_width = width
        self.input_channel = channel
        self.features = nn.Sequential(
            nn.Conv2D(self.input_channel, 6, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2),
            nn.Conv2D(6, 16, 5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(400, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10),
        )
        
    def forward(self, x):
        features = self.features(x)
        return features

class loss(nn.Layer):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, label):
        # loss = paddle.nn.functional.pairwise_distance(x, paddle.cast(label, dtype='float32'))
        loss = paddle.nn.functional.cross_entropy(x, label)
        # loss = paddle.mean(loss)
        return loss



h, w, c = [28, 28, 1]
batch_size = 1
testModel = TestNet(h, w, c)
params_info = paddle.summary(testModel, (batch_size, c, 28, 28))
print(params_info)

transform = Normalize(mean=[127.5], std=[127.5], data_format='CHW')
# 下载数据集并初始化 DataSet
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)

# 模型组网并初始化网络
model = paddle.Model(testModel)

# 模型训练的配置准备，准备损失函数，优化器和评价指标
model.prepare(paddle.optimizer.Adam(parameters=model.parameters()), 
              loss(),
              paddle.metric.Accuracy())

# 模型训练
model.fit(train_dataset, epochs=5, batch_size=64, verbose=1)
# 模型评估
model.evaluate(test_dataset, batch_size=64, verbose=1)

# 保存模型
model.save('./output/mnist')
# 加载模型
model.load('output/mnist')

# 从测试集中取出一张图片
img, label = test_dataset[0]
# 将图片shape从1*28*28变为1*1*28*28，增加一个batch维度，以匹配模型输入格式要求
img_batch = np.expand_dims(img.astype('float32'), axis=0)

# 执行推理并打印结果，此处predict_batch返回的是一个list，取出其中数据获得预测结果
out = model.predict_batch(img_batch)[0]
pred_label = out.argmax()
print('true label: {}, pred label: {}'.format(label[0], pred_label))
# 可视化图片
from matplotlib import pyplot as plt
plt.imshow(img[0])