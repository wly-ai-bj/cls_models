import os
import numpy as np 
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from tensorboardX import SummaryWriter
from torch.autograd import Variable

#7x7 convolution
def conv7x7(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, dilation=1, groups=1, bias=False, padding_mode='zeros')

#3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros')

#1x1 convolution
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros')

#Batch_Normalization
def Batch_Norm(num_features):
    return nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)

class BasicBlock(nn.Module):
    # in_expansion = 1
    out_expansion = 1
    def __init__(self, in_channels, out_channels, stride=1,downsample=None):
        super(BasicBlock,self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = Batch_Norm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    
    def forward(self,x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample :
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class Bottelneck(nn.Module):
    # in_expansion = 2
    out_expansion = 4
    def __init__(self,in_channels,out_channels,stride=1,downsample=None):
        super(Bottelneck,self).__init__()
        self.conv1 = conv1x1(in_channels, out_channels, stride=stride)
        self.bn1 = Batch_Norm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels,out_channels,stride=1)
        self.bn2 = Batch_Norm(out_channels)
        self.conv3 = conv1x1(out_channels,out_channels*self.out_expansion,stride=1)
        self.bn3 = Batch_Norm(out_channels*self.out_expansion)
        self.downsample = downsample

    def forward(self,x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample :
            identity = self.downsample(x)
        
        out += identity
        return out


class ResNet(nn.Module):
    def __init__(self,block,in_channels,layers,num_classes,ds_4):
        super(ResNet,self).__init__()
        # self.in_channels = 64
        # self.batch_size = batch_size
        # self.conv1 = conv7x7(3, 64, 2)
        # self.bn1 = Batch_Norm(64)
        # self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            conv3x3(3, 64, 2),Batch_Norm(64),nn.ReLU(inplace=True),
            conv3x3(64,64, 1),Batch_Norm(64),nn.ReLU(inplace=True),
            conv3x3(64,64, 1),Batch_Norm(64),nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
        self.block_layer1 = self.block_layers(in_channels[0],64,1,block,layers[0])
        self.block_layer2 = self.block_layers(in_channels[1],128,2,block,layers[1])
        self.block_layer3 = self.block_layers(in_channels[2],256,2,block,layers[2])

        self.block_layer4_1 = self.block_layers(in_channels[3],512,ds_4,block,layers[3])
        self.block_layer4_2 = self.block_layers(in_channels[3],512,ds_4,block,layers[3])
        self.block_layer4_3 = self.block_layers(in_channels[3],512,ds_4,block,layers[3])
        self.block_layer4_4 = self.block_layers(in_channels[3],512,ds_4,block,layers[3])
        self.GAP1 = nn.AdaptiveAvgPool2d((1,1))
        self.GAP2 = nn.AdaptiveAvgPool2d((1,1))
        self.GAP3 = nn.AdaptiveAvgPool2d((1,1))
        self.GAP4 = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(512*block.out_expansion, num_classes[0], bias=True)
        self.fc2 = nn.Linear(512*block.out_expansion, num_classes[1], bias=True)
        self.fc3 = nn.Linear(512*block.out_expansion, num_classes[2], bias=True)
        self.fc4 = nn.Linear(512*block.out_expansion, num_classes[3], bias=True)

    def block_layers(self,in_channels,out_channels,stride,block,block_num):
        if (in_channels != out_channels*block.out_expansion) or (stride != 1):
            downsample = nn.Sequential(
                conv1x1(in_channels, out_channels*block.out_expansion, stride=stride),
                Batch_Norm(out_channels*block.out_expansion)
            )
        else:
            downsample = None
        layers= []
        layers.append(block(in_channels, out_channels, stride,downsample))
        in_channels = out_channels*block.out_expansion
        for _ in range(1,block_num):
            layers.append(block(out_channels*block.out_expansion,out_channels))
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)
        out = self.maxpool(out)
        out = self.block_layer1(out)
        out = self.block_layer2(out)
        out = self.block_layer3(out)
        out1 = self.block_layer4_1(out)
        out2 = self.block_layer4_2(out)
        out3 = self.block_layer4_3(out)
        out4 = self.block_layer4_4(out)
        out1 = self.GAP1(out1)
        out2 = self.GAP2(out2)
        out3 = self.GAP3(out3)
        out4 = self.GAP4(out4)
        out1 = out1.view(out1.size(0), -1)
        out2 = out2.view(out2.size(0), -1)
        out3 = out3.view(out3.size(0), -1)
        out4 = out4.view(out4.size(0), -1)
        out1 = self.fc1(out1)
        out2 = self.fc2(out2)
        out3 = self.fc3(out3)
        out4 = self.fc4(out4)     
        return out1,out2,out3,out4
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


def resnet18(num_classes,ds_4):
    return ResNet(BasicBlock,[64,64,128,256],[2,2,2,2],num_classes=num_classes,ds_4=ds_4)

def resnet34(num_classes,ds_4):
    return ResNet(BasicBlock,[64,64,128,256],[3,4,6,3],num_classes=num_classes,ds_4=ds_4)

def resnet50(num_classes,ds_4):
    return ResNet(Bottelneck,[64,256,512,1024],[3,4,6,3],num_classes=num_classes,ds_4=ds_4)

def resnet101(num_classes,ds_4):
    return ResNet(Bottelneck,[64,256,512,1024],[3,4,23,3],num_classes=num_classes,ds_4=ds_4)

def resnet152(num_classes,ds_4):
    return ResNet(Bottelneck,[64,256,512,1024],[3,8,36,3],num_classes=num_classes,ds_4=ds_4)

def model_sel(model_arc,num_classes,ds_4):
    if model_arc == 'resnet18':
        return resnet18(num_classes,ds_4)
    elif model_arc == 'resnet34':
        return resnet34(num_classes,ds_4)
    elif model_arc == 'resnet50':
        return resnet50(num_classes,ds_4)
    elif model_arc == 'resnet101':
        return resnet101(num_classes,ds_4)
    elif model_arc == 'resnet152':
        return resnet152(num_classes,ds_4)
    else:
        print('The model is not exist!')
        return None

def main():
    graph_path = "./summary/"
    if not os.path.exists(graph_path):
        os.mkdir(graph_path)
    
    dummy_img = Variable(torch.rand(size=(80,3,112,112)))  #假设输入32张3*224*224的图片
    model = resnet18([2,2,2,5],1)#gender,glasses,mask，age（0～9 幼儿，10～19 少年，20～39 青年，40～69 中年，70以上 老年）
    # dummy_img = torch.from_numpy(np.random.randn(3,224,224))
    writer = SummaryWriter(logdir=graph_path,comment='resnet18')
    writer.add_graph(model,(dummy_img,),verbose=True)
    writer.close()



if __name__ == "__main__":
    main()