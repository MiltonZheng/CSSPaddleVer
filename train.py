from dsth import DSTH
import paddle
import utils
from paddle.vision.transforms import Normalize, Resize
from tqdm import tqdm


# logwriter = LogWriter(logdir='./output/dsth_experiment')
# ! h and w control the input size of the model
# ! all the input images will be resized to this size
h, w, c = [64, 64, 3]
batch_size = 64
EPOCH_NUM = 200
learning_rate = 0.0002
train_size = 25000
dsthModel = DSTH(h, w, c)
# * print the structural information of the model
print(paddle.summary(dsthModel, (batch_size, c, h, w)))

path = "../datasets/NWPU-RESISC45/train"
train_set = utils.build_trainset(path)
data_loader = paddle.io.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)
opt = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=dsthModel.parameters())
avg_loss = 0.0
for epoch_id in range(EPOCH_NUM):
    avg_loss_epoch = 0.0
    for batch_id, data in enumerate(tqdm(data_loader(), 
                                         desc="Epoch {}/{} [loss: {:.7f}]".format(epoch_id, EPOCH_NUM, float(avg_loss_epoch)))):
        images, labels = data
        
        # * resize the data
        # * also it needs to be converted into float32
        # * Instead of transforming the data type  when building the data loader,
        # * we do this when we actually load a data batch, so the data won't take up too much space
        images_t = paddle.zeros([images.shape[0], c, h, w])
        for i in range(images.shape[0]):
            images_t[i] = Resize((h, w))(images[i]).astype("float32")
        images = images_t
        labels = labels.astype('float32')
        
        predicts = dsthModel(images)
        # * compute the loss
        loss = paddle.nn.functional.pairwise_distance(predicts, paddle.cast(labels, dtype='float32'))
        avg_loss = paddle.mean(loss)
        avg_loss_epoch += paddle.sum(loss)
        # CrossEntropyLoss = paddle.nn.CrossEntropyLoss(soft_label=True)
        # avg_loss = CrossEntropyLoss(predicts, labels)
        
        # * back propagation
        avg_loss.backward()
        opt.step()
        opt.clear_grad()
    avg_loss_epoch = avg_loss_epoch / train_size

# * save
path = "../output/dsth"
paddle.jit.save(dsthModel, path)