import os
from paddle.io import Dataset


class MyDataset(Dataset):
    """
    继承 paddle.io.Dataset 类
    """
    def __init__(self, data, label, transform=None):
        """
        实现 __init__ 函数，初始化数据集，将样本和标签映射到列表中
        """
        super().__init__()
        self.data_list = []
        for i in range(data.shape[0]):
            self.data_list.append([data[i], label[i]])
        # 传入定义好的数据处理方法，作为自定义数据集类的一个属性
        self.transform = transform

    def __getitem__(self, index):
        """
        实现 __getitem__ 函数，定义指定 index 时如何获取数据，并返回单条数据（样本数据、对应的标签）
        """
        # 根据索引，从列表中取出一个图像
        image, label = self.data_list[index]
        label = int(label)
        # 返回图像和对应标签
        return image, label

    def __len__(self):
        """
        实现 __len__ 函数，返回数据集的样本总数
        """
        return len(self.data_list)

# # 打印数据集样本数        
# train_custom_dataset = MyDataset('mnist/train','mnist/train/label.txt')
# test_custom_dataset = MyDataset('mnist/val','mnist/val/label.txt')
# print('train_custom_dataset images: ',len(train_custom_dataset), 'test_custom_dataset images: ',len(test_custom_dataset))