from paddle import nn
import paddle
from paddle.vision.transforms import Resize

class DSTH(nn.Layer):
    def __init__(self, height = 256, width = 256, channel = 3, batch_size = 64):
        super().__init__()
        self.input_height = height
        self.input_width = width
        self.input_channel = channel
        self.batch_size = batch_size
        self.features = nn.Sequential(
            nn.Conv2D(self.input_channel, 32, 5, stride=1, padding=2),
            nn.MaxPool2D(kernel_size=3, stride=2, padding=0),
            nn.Conv2D(32, 32, 5, stride=1, padding=2),
            nn.AvgPool2D(kernel_size=3, stride=2, padding=0),
            nn.Conv2D(32, 64, 5, stride=1, padding=2),
            nn.BatchNorm(64, momentum=0.9, epsilon=1e-5),
            nn.ReLU(),
        )
        # * 此处计算feature子网络的输出特征图的高度和宽度
        self.f_height = int((int((self.input_height-1)/2)-1)/2)
        self.f_width = int((int((self.input_width-1)/2)-1)/2)
        self.linear = nn.Sequential(
            nn.Linear(64*self.f_height*self.f_width, 512),
            )
        self.slice = nn.Sequential(
            nn.Linear(512, 48),
            nn.BatchNorm(1, momentum=0.9, epsilon=1e-5),
            )
        
    def forward(self, x):
        transform = Resize((self.input_height, self.input_width))
        x_transformed = paddle.to_tensor(paddle.zeros([self.batch_size, self.input_channel, 
                                                       self.input_height, self.input_width]))
        for i in range(x.shape[0]):
            x_transformed[i] = transform(x[i])
        x = x_transformed
        features = self.features(x)
        features = paddle.reshape(features, [-1, 1, 64*self.f_height*self.f_width])
        linear = self.linear(features)
        logits = self.slice(linear)
        # logits = paddle.to_tensor([])
        # for i in range(16):
        #     slice1 = paddle.slice(linear, axes=[2], starts=[256*i], ends=[256*(i+1)])
        #     slice2 = self.slice(slice1)
        #     logits = paddle.concat(x=[logits, slice2], axis=2)
        # logits = paddle.reshape(logits, [-1, 48])
        return logits

class loss(nn.Layer):
    def __init__(self):
        super().__init__()
        
    def forward(self, logits, label):
        loss = paddle.nn.functional.pairwise_distance(logits, label)
        loss = paddle.mean(loss)
        return loss



h, w, c = [32, 32, 3]
batch_size = 64
dsthModel = DSTH(h, w, c, batch_size)
params_info = paddle.summary(dsthModel, (batch_size, c, 256, 256))
print(params_info)