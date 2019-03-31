# Pytorch 0.4.0 VGG16实现cifar10分类.
# @Time: 2018/6/23
# @Author: wxq

import torch
import torch.nn as nn
import math
import torchvision.transforms as transforms
import torchvision as tv
from torch.utils.data import DataLoader
import numpy as np

model_path = './model_pth/vgg16_bn-6c64b313.pth'  # 预训练模型的数据储存文件
# 当时LR=0.01遇到问题，loss逐渐下降，但是accuracy并没有提高，而是一直在10%左右，修改LR=0.00005后，该情况明显有所好转，准确率最终提高到了
# 当LR=0.0005时，发现准确率会慢慢提高，但是提高的速度很慢，这时需要增加BATCH_SIZE，可以加快训练的速度，但是要注意，BATCH_SIZE增大会影响最终训练的准确率，太大了还可能也会出现不收敛的问题
# 另外，注意每次进入下一个EPOCH都会让准确率有较大的提高，所以EPOCH数也非常重要，需要让网络对原有数据进行反复学习，强化记忆
#
# 目前，调试的最好的参数是BATCH_SIZE = 500  LR = 0.0005  EPOCH = 10  最终准确率为：69.8%    用时：
BATCH_SIZE = 500  # 将训练集划分为多个小批量训练，每个小批量的数据量为BATCH_SIZE
LR = 0.0005  # learning rate
EPOCH = 10  # 训练集反复训练的次数，每完整训练一次训练集，称为一个EPOCH
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class VGG(nn.Module):
    def __init__(self, features, num_classes=10):  # 构造函数 num_class根据最后分类的种类数量而定，cifar为10所以这里是10
        super(VGG, self).__init__()  # pytorch继承nn.Module模块的标准格式，需要继承nn.Module的__init__初始化函数
        self.features = features  # 图像特征提取网络结构（仅包含卷积层和池化层，不包含分类器）
        self.classifier = nn.Sequential(  # 图像特征分类器网络结构
            # FC4096 全连接层
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            # FC4096 全连接层
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            # FC1000 全连接层
            nn.Linear(4096, num_classes))
        # 初始化各层的权值参数
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']  # vgg16的网络结构参数，数字代表该层的卷积核个数，'M'代表该层为最大池化层


def make_layers(cfg, batch_norm=False):
    """利用cfg，生成vgg网络每层结构的函数"""
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':  # 最大池化层
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # 根据cfg设定卷积层的卷积核的数量v，根据论文，vgg网络中的卷积核尺寸均使用3x3xn，n是输入数据的通道数
            conv2d = nn.Conv2d(in_channels, v, 3, padding=1)  # 卷积层,in_channels是输入数据的通道数，初始RGB图像的通道数为3
            if batch_norm:  # 对batch是否进行标准化
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]  # 每次卷积完成后，需要使用激活函数ReLU激活一下，保持特征的非线性属性
            in_channels = v  # 下一层的输入数据的通道数，就是上一层卷积核的个数
    return nn.Sequential(*layers)  # 返回一个包含了网络结构的时序容器，加*是为了只传递layers列表中的内容，而不是传递列表本身


def vgg16(**kwargs):
    model = VGG(make_layers(cfg, batch_norm=True), **kwargs)  # batch_norm一定要等于True，如果不对batch进行标准化，那么训练结果的准确率一直无法提升
    # model.load_state_dict(torch.load(model_path))  # 如果需要使用预训练模型，则加入该代码
    return model


def getData():  # 定义数据预处理
    # transforms.Compose([...])就是将列表[]里的所有操作组合起来，返回所有操作组合的句柄
    transform = transforms.Compose([
        transforms.RandomResizedCrop(32),  # 将图像随机裁剪为不同大小(默认0.08~1.0和宽高比(默认3/4~4/3)，224是期望输出的图像大小
        transforms.RandomHorizontalFlip(),  # 以一定几率(默认为0.5)水平翻转图像
        transforms.ToTensor(),  # 将图像数据或数组数据转换为tensor数据类型
        transforms.Normalize(mean=[0.5, 0.5, 0.5],  # 标准化tensor数据类型的图像数据，其中mean[i],std[i]分别表示图像第i个通道的均值和标准差，
                             std=[1, 1, 1])])  #标准化公式为input[channel] =(input[channel] - mean[channel])/std[channel]
    trainset = tv.datasets.CIFAR10(root='G:\\Lerne\\machinelearning\\Deeplearning\\vgg16\\cifar_data', train=True, transform=transform, download=True)  # 获取CIFAR10的训练数据
    testset = tv.datasets.CIFAR10(root='G:\\Lerne\\machinelearning\\Deeplearning\\vgg16\\cifar_data', train=False, transform=transform, download=True)  # 获取CIFAR10的测试数据

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)  # 将数据集导入到pytorch.DataLoader中
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)  # 将测试集导入到pytorch.DataLoader中
    return train_loader, test_loader


def train():
    """创建网络，并开始训练"""
    trainset_loader, testset_loader = getData()  # 获取数据
    net = vgg16().cuda()  # 创建vgg16的网络对象
    net.train()
    print(net)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss().cuda()  # 定义网络的损失函数句柄,CrossEntropyLoss内部已经包含了softmax过程，所以神经网络只需要直接输出需要输入softmax层的数据即可
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)  # 定义网络的优化器句柄，使用Adam优化器

    # Train the model
    for epoch in range(EPOCH):
        true_num = 0.
        sum_loss = 0.
        total = 0.
        accuracy = 0.
        for step, (inputs_cpu, labels_cpu) in enumerate(trainset_loader):
            inputs = inputs_cpu.cuda()
            labels = labels_cpu.cuda()
            output = net(inputs)
            loss = criterion(output, labels)  # 计算一次训练的损失
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传递
            optimizer.step()  # 用Adam优化器优化网络结构参数

            _, predicted = torch.max(output, 1)  # predicted 当前图像预测的类别
            sum_loss += loss.item()
            total += labels.size(0)
            accuracy += (predicted == labels).sum()
            # tensor数据(在GPU上计算的)如果需要进行常规计算，必须要加.cpu().numpy()转换为numpy类型，否则数据类型无法自动转换为float
            print("epoch %d | step %d: loss = %.4f, the accuracy now is %.3f %%." % (epoch, step, sum_loss/(step+1), 100.*accuracy.cpu().numpy()/total))
        # 检测当前网络在训练集上的效果，并显示当前网络的训练情况
        acc = test(net, testset_loader)
        print("")
        print("___________________________________________________")
        print("epoch %d : training accuracy = %.4f %%" % (epoch, 100 * acc))
        print("---------------------------------------------------")


    print('Finished Training')
    return net


def test(net, testdata):
    """检测当前网络的效果"""
    correct, total = .0, .0
    for inputs_cpu, labels_cpu in testdata:
        inputs = inputs_cpu.cuda()
        labels = labels_cpu.cuda()
        net.eval()  # 有些模块在training和test/evaluation的时候作用不一样，比如dropout等等。
                    # net.eval()就是将网络里这些模块转换为evaluation的模式
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        #print(predicted)
    # net.train()
    # tensor数据(在GPU上计算的)如果需要进行常规计算，必须要加.cpu().numpy()转换为numpy类型，否则数据类型无法自动转换为float
    return float(correct.cpu().numpy()) / total


if __name__ == '__main__':
    net = train()