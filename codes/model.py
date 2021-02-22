import torch.nn as nn
import torch

# 首先说明一下,两种残差结构:卷积+实线残差;卷积+虚线残差
class BasicBlock(nn.Module): # 定义的是18层和34层resnet中每一个残差结构(有两层卷积+一个残差)
    expansion = 1   # 18层和34层的残差结构中的第一层和第二层的卷积核的个数是一样的(也就是上一层的卷积核的个数(卷积层的输出的feature map个数)==下一层的卷积核的个数(卷积层的输出的feature map个数))
                    # expation设为1就是因为在18，34层中每一个卷积层中卷积核的个并没有发生变化
    def __init__(self, in_channel, out_channel, stride=1, downsample=None): # download是指虚线的捷径中设置的卷积层，简称“下采样”
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,    # 一个残差结构中的第一个卷积层
                               kernel_size=3, stride=stride, padding=1, bias=False) # 当使用的是虚线的残差结构时，传入的stride就会是2,此时这个卷积层起到的作用就是将input的size所谓原来的一半
        self.bn1 = nn.BatchNorm2d(out_channel) # bn层
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample  # 下采样默认赋值为空

    def forward(self, x):
        identity = x  # 先预留输入的特征矩阵
        if self.downsample is not None: # 如果有下采样，说明是实线残差
            identity = self.downsample(x) # 处理输入的特征矩阵

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out) 

        out += identity  # 加上初始矩阵
        out = self.relu(out)

        return out


class Bottleneck(nn.Module): # 定义的50，101，152层resnet中每一个残差结构(有三层卷积+一个残差)
    expansion = 4  # 同一个残差结构中的第三层的卷积核的个数等于第1,2层的卷积核的个数的4倍

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1) # 这里同理，如果是作为残差层的第一个残差结构，stride=2，从而将input_size缩小为原来的一般
        self.bn2 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion, # 因为第三层卷积层的输出是输入的4倍
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        '''
        参数解释
        block：ResNet中使用的残差结构类型，如果resnet是18,34层的就使用BasicBlock,50,101,152使用Bottleneck类型
        blocks_num：每一个残差层(包含多个残差结构)里边包含的残差结构的个数，然后这个参数是这些个数的列表
        num_classes: 分类个数
        include_top: 方便为了在resnet基础上搭建更加复杂的网络的基础
        '''
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64 # 输入特征图的深度(经过初始的maxpooling之后的特征图)

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, # 注意哦，self.in_channel是作为第一个卷积层的输出个数
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 最大池化操作 
        self.layer1 = self._make_layer(block, 64,  blocks_num[0])           # 第一个残差层 Conv_2
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2) # 第二个残差层 Conv_3
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2) # 第三个残差层 Conv_4
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2) # 第四个残差层 Conv_5
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1): # stride=1指的是stride默认是1的意思
        '''
        参数解释
        block: 残差结构的类型
        channel: 残差结构中卷积层所用到卷积核的个数(定义的是第一层卷积核的个数)
        block_nun： 当前残差层的残差结构的个数
        '''

        downsample = None # 下采样函数初始化
        '''
        因为Conv_2,Conv_3...都是通过_make_layer函数来生成的，但不是每一层都需要有下采样函数的，也不是每一层的每一个残差结构都需要下采样的函数（也就是带虚线残差结构）的情况有以下几种:
        1. 如果是ResNet中的50,101,152,那么每一个残差层的第一个的残差结构就一定是带虚线的残差结构
        2. 如果是ResNet中的34，那么只有从Conv_3开始的每一个残差层的第一个残差结构才是带虚线的残差结构

        现在用构造ResNet34，ResNet50的构造过程举例说明
        1. ResNet34
        (1)Conv_2：因为此时stride=1,且此时self.in_channel和从外面输入的channel都是64，所以不会生成下采样函数，然后通过if模块的下面那几行代码构建了第一个残差结构，此时传入的stride=1,
        后面是循环，继续构建Conv_2中的其他的残差结构，注意，此时传入的stride也是1，说明没有改变width,height
        (2)Conv_3: 此时的self.in_channel还是64但是外面输入的channel是128了，并且因为传入的stride变成了2，两个条件都满足了所以构造了下采样函数，下采样函数的卷积层的输出是channel*1，
        所以还是128因为此时传入的stride是2，所以下采样函数中也是2，起到将原输入的width和height都降一半的作用，因此在构造主线的第一个残差结构的时候，第一个卷积层的stride是用的2，起到
        缩小shape的作用，第二个卷积层就直接stride=1,维持原样，还要注意构造完第一个残差结构之后，将self.in_channel更新，所以self.in_channel变成了128，然后在构造剩余的残差结构的时候，
        就输入输出都是128，stride默认是1，就一直保持原样。
        (3)Conv_4,Conv_5同理

        2. ResNet50
        (1)Conv_2: 输入的stride=1，但是此时self.in_channel=64,channel=64,block.expansion=4,所以构造了下采样函数，但是因为Conv_2并没有改变input的width和height，所以下采样函数的stride也是1
        在构造第一个残差结构时，第一个卷积层的stride是固定是1的，然后第二个卷积层因为传入的stride还是1所以还是1，第三层卷层的stride还是固定1,构造完第一个残差结构之后，修改self.in_channel=256，
        在构造Conv_2的其他残差结构时，传入默认stride都是1，且每一个残差结构的卷积核的输入输出都是(256,64),(64,64),(64,256)
        (2)Conv_3: 输入的stride=2，且self.in_channel=256,channel=128,block.expansion=4,所以构造输入是256，输出是512，stride=2的卷积层作为下采样，然后在构建第一个残差结构时，主线的第一个卷积层
        的输入是256，输出是128，同时因为stride=1所以shape没变，第二层卷积层输入是128，输出是128，但是因为此时stride的是传入的2，所以shape降为原来的一半，第三层，因为stride=1，shape不变，输入channel
        为128，输出为128*4=512，这样就构造好了第一个残差结构，这一层的其他残差结构就是直接保持shape不变，输入输出channels数依次是(512,128),(128,128),(128,512)。
        (3)Conv_4,Conv_5跟Conv_3同理。
        '''
        
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential( # 生成下采样函数
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False), # 虚线残差卷积核个数要乘4倍
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion # 要及时修正作为下一个残差结构的输入值

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

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

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)

def resnet50(num_classes=1000,include_top=True):
    return ResNet(Bottleneck,[3,4,6,3],num_classes=num_classes,include_top=include_top)