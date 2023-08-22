from paddle import nn
import paddle

class DSTH(nn.Layer):
    def __init__(self, height = 256, width = 256, channel = 3, label = None):
        super().__init__()
        self.input_height = height
        self.input_width = width
        self.input_channel = channel
        self.label = label
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
            nn.Linear(64*self.f_height*self.f_width, 4096),
            )
        self.slice = nn.Sequential(
            nn.BatchNorm(1, momentum=0.9, epsilon=1e-5),
            nn.Linear(256, 3),
            )
        
    def forward(self, x):
        features = self.features(x)
        features = paddle.reshape(features, [-1, 1, 64*self.f_height*self.f_width])
        linear = self.linear(features)
        logits = paddle.to_tensor([])
        for i in range(16):
            slice1 = paddle.slice(linear, axes=[2], starts=[256*i], ends=[256*(i+1)])
            slice2 = self.slice(slice1)
            paddle.concat([logits, slice2], axis=2)
        loss = nn.functional.cross_entropy(logits, self.label)
        return logits




# h, w, c = [256, 256, 3]
# batch_size = 64
# dsthModel = DSTH(h, w, c)
# params_info = paddle.summary(dsthModel, (batch_size, c, h, w))
# print(params_info)