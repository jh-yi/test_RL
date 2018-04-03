# -*- coding:utf-8 -*-
'''
ResNet50 Graph: http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006
http://blog.csdn.net/kongshuchen/article/details/72285709
可用transform.Compose([...])扩大训练数据集
short cut上stride=2时需要downsample
'''
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import dataloader
import torch.utils.model_zoo as model_zoo
import math
import numpy as np

# from keras.preprocessing import image
# from keras.applications.resnet50 import preprocess_input
# from keras.applications.resnet50 import decode_predictions
# import cv2
# from PIL import Image

# weights download website
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def get_data(download=False):
    '''download, proprecess and load the pics data set

    :param download: bool, True or False
    :return:
        train_loader: object:DataLoader>
        test_loader: object:DataLoader>
    '''

    # Image Preprocessing       transform<object:Compose>  attribute:<list> of objects
    transform = transforms.Compose([  # Composes several transforms together.为了扩大训练数据集
        # transforms.Scale(40),      #UserWarning: The use of the transforms.Scale transform is deprecated, please use transforms.Resize instead.
        transforms.Resize(40),  # for new model
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),  # for new model
        transforms.ToTensor()])

    # CIFAR-10 Dataset
    train_dataset = dsets.CIFAR10(root='./data/',  # <object:CIFAR10>
                                  train=True,
                                  transform=transform,
                                  download=download)

    test_dataset = dsets.CIFAR10(root='./data/',
                                 train=False,
                                 transform=transforms.ToTensor())

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,  # <object:DataLoader>
                                               batch_size=100,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=100,
                                              shuffle=False)
    return train_loader, test_loader


def conv3x3_new(in_channels, out_channels, stride=1):
    # 3x3 convolution with padding
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    '''
    即论文中的两层残差块
    '''

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3_new(in_channels, out_channels, stride)  # Stride可变
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_new(out_channels, out_channels)  # stride总为1
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:  # short cut上stride=2时需要downsample
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# ResNet_new Module
class ResNet_new(nn.Module):
    # 与源文件略有不同
    def __init__(self, block, layers, num_classes=10):  # num_classes because of cifr10
        super(ResNet_new, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3_new(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16,
                                      layers[0])  # return Sequential model(两个ResidualBlock), ?16-RB-16 16-RB-16
        self.layer2 = self.make_layer(block, 32, layers[1],
                                      2)  # return Sequential model(两个ResidualBlock),?16-RB/2-32 32-RB-32
        self.layer3 = self.make_layer(block, 64, layers[2],
                                      2)  # return Sequential model(两个ResidualBlock),?32-RB/2-64 64-RB-64
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

        # for m in self.modules()...原文件中这一段不懂

    def make_layer(self, block, out_channels, blocks, stride=1):
        '''get Sequential , consist of blocks X ResidualBlock
        :param block: e.g. class ResidualBlock, __init__(self, in_channels, out_channels, stride=1, downsample=None):
        :param out_channels:
        :param blocks: e.g. layers = [2,2,2,2]; blocks = layers[0]=2
        :param stride:(stride != 1) or (self.in_channels != out_channels) 说明short cut需要downsample
        :return:nn.Sequential模型
        '''
        downsample = None
        if (stride != 1) or (
                self.in_channels != out_channels):  # 源文件中out_channels*block.expansion，只是此时block.expansion=1不考虑
            downsample = nn.Sequential(
                conv3x3_new(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride, downsample))  # return a list of object(class ResidualBlock)
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)  # *args可以当作可容纳多个变量组成的list

    def forward(self, x):
        '''(conv-bn-relu) - (blocks X ResidualBlock) - blocks X ResidualBlock/2 - blocks X ResidualBlock/2 - avg_pol-Flatten-fc
        :param x: Variable  e.g.   .data>>torch.Size([100,3,32,32])  #100是以为batch_size=100
        :return: out      .data>>torch.Size([100,10])
        '''
        out = self.conv(x)  # conv =conv3x3_new(3,16)>>  .data>>torch.Size([100,16,32,32])
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)  # out_channels=16>>  .data>>torch.Size([100,16,32,32])
        out = self.layer2(out)  # out_channels=132 + subsample>>  .data>>torch.Size([100,32,16,16])
        out = self.layer3(out)  # out_channels=164 +subsample>>  .data>>torch.Size([100,64,8,8])
        out = self.avg_pool(out)  # AvgPool2d(8)>>>>  .data>>torch.Size([100,64,1,1])
        out = out.view(out.size(0), -1)  # >>>>  .data>>torch.Size([100,64])
        out = self.fc(out)  # fc = nn.Linear(64, num_classes)>>  .data>>torch.Size([100,num_classes])
        return out


# ------------------以下为源文件-----------------------------------------------

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    '''
    downsample=None: IdentityBlock
    downsample!=None: ConvBlock
    规律：--in_channels->out_channels>out_channles>out_channdels*expansion
    '''
    expansion = 4  # 在class ResNet的_make_layer中expansion决定了shortcut路有conv层

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# [3,4,6,3]个Bottleneck
def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet_new-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


# ------------------------------------------------------------------------------

def main():
    # -----------------------------------built in ResNet50------------------------------
    '''
    transform1 = transforms.Compose([transforms.Resize(224,224),
                                     transforms.ToTensor()])
    pic_name = 'cat.jpg'
    #pic_name = 'cat.jpg'

    print("preprocessing")

    img = cv2.imread(pic_name)
    img = cv2.resize(img, (224,224))
    img = img.reshape(-1,3,224,224)     #RuntimeError: Given groups=1, weight[64, 3, 7, 7], so expected input[1, 224, 224, 3] to have 3 channels, but got 224 channels instead
    img = img.astype('float32')
    img = img[..., ::-1]
    img = img/255.
    img -= 0.5
    img *= 2
    img = torch.from_numpy(img)


    resnet = resnet50(pretrained=True)
    '''

    # ------------------------------------new model---------------------------------------------------

    download = False
    train_loader, test_loader = get_data(download=download)
    # resnet = ResNet_new(ResidualBlock, [2, 2, 2, 2]).cuda()
    resnet = ResNet_new(ResidualBlock, [2, 2, 2, 2])

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)

    # -----------------------------Training-----------------------------------------------------------
    '''
    print("Training")
    epochs = 80  # 80
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(
                train_loader):  # images<Float Tensor> .data>>torch.Size([100,3,32,32]) labels<long Tensor> torch.Size([100])
            # if (i+1)%10==0:     #只是为了数据集缩小10倍
            print("Training pics:", i)
            # images = Variable(images.cuda())
            images = Variable(images)  # <Variable> .data>>torch.Size([100,3,32,32])
            # labels = Variable(labels.cuda())
            labels = Variable(labels)  # <Variable>  .data>>torch.Size([100])

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = resnet(images)  # <Variable> .data>>torch.Size([100,10])
            loss = criterion(outputs, labels)  # <Variable> .data>>torch.Size([1])
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" % (epoch + 1, epochs, i + 1, 500, loss.data[0]))

        # Decaying Learning Rate
        if (epoch + 1) % 20 == 0:
            lr /= 3
            optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)

            # Test
    torch.save(resnet.state_dict(), 'resnet_cifar10_' + str(epochs) + ' epochs.pkl')
    '''
#--------------------------------------Testing-----------------------------------------------------------------
    resnet.load_state_dict(torch.load('resnet_cifar10_80 epochs.pkl'))

    correct = 0  # total correct times of prediction
    total = 0  # total numbers of test pics
    # for images, labels in test_loader:          #由此看出test_loader为多个tuple
    print('Testing')
    for i, (images, labels) in enumerate(test_loader):
        # images = Variable(images.cuda())
        images = Variable(images)
        outputs = resnet(images)
        _, predicted = torch.max(outputs.data,
                                 1)  # <Variable> .data>>torch.Size([100,10]) | torch.max() >>(每行最大值Torch, 最大值下标Torch)
        total += labels.size(0)
        correct += (predicted == labels).sum()  # 每次+=100次里面正确预测的次数
        print("current testing pics_batch:", i, "\ncurrent acc:%d %%" % (100 * correct / total))
    print('Accuracy of the model on the test images: %d %%' % (100 * correct / total))

    # Save the Model    only for new model
    # torch.save(resnet.state_dict(), 'resnet.pkl')

    '''
    #for single image
    image = Variable(img)
    preds= resnet(image)
    preds = preds.data.numpy()      #单单preds.data.numpy()并未改变
    print(type(preds),preds.shape, len(preds.shape),preds.shape[1])
    print('Predicted:', decode_predictions(preds, top=3)[0])
    '''

    #test the acc in training dataset
    correct = 0  # total correct times of prediction
    total = 0  # total numbers of test pics
    # for images, labels in test_loader:          #由此看出test_loader为多个tuple
    print('Trainingdata testing')
    for i, (images, labels) in enumerate(train_loader):
        # images = Variable(images.cuda())
        images = Variable(images)
        outputs = resnet(images)
        _, predicted = torch.max(outputs.data,
                                 1)  # <Variable> .data>>torch.Size([100,10]) | torch.max() >>(每行最大值Torch, 最大值下标Torch)
        total += labels.size(0)
        correct += (predicted == labels).sum()  # 每次+=100次里面正确预测的次数
        print("current training pics_batch:", i, "\ncurrent acc:%d %%" % (100 * correct / total))
    print('Accuracy of the model on the train images: %d %%' % (100 * correct / total))

if __name__ == "__main__":
    main()
