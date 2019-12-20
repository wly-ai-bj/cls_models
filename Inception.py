import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

import os

def conv7x7(in_channels, out_channels, stride=2, padding=3):
    return nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=stride, padding=padding, dilation=1, groups=1, bias=False)

def conv5x5(in_channels, out_channels, stride=1, padding=2):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=padding, dilation=1, groups=1, bias=False)

def conv3x3(in_channels, out_channels, stride=1,padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, dilation=1, groups=1, bias=False)

def conv1x1(in_channels, out_channels,stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, dilation=1, groups=1, bias=False)

def conv1xn(in_channels, out_channels, kernel_size, padding):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, groups=1, bias=False)

def convnx1(in_channels, out_channels, kernel_size, padding):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, groups=1, bias=False)

def bn(num_features):
    return nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)

def maxpool(stride=2, padding=1):
    return nn.MaxPool2d(kernel_size=3, stride=stride, padding=padding, dilation=1, return_indices=False, ceil_mode=False)

def fc(in_features, out_features):
    return nn.Linear(in_features, out_features, bias=True)

def gap(kernel_size=7,stride=None):
    return nn.AvgPool2d(kernel_size=kernel_size,stride=stride)

class Inception_v1(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Inception_v1,self).__init__()
        self.conv1 = nn.Sequential(
            conv1x1(in_channels,out_channels[0]),
            bn(out_channels[0]),
            nn.ReLU()
        )
        self.conv2_1 = nn.Sequential(
            conv1x1(in_channels,out_channels[1][0]),
            bn(out_channels[1][0]),
            nn.ReLU()
        )
        self.conv2_2 = nn.Sequential(
            conv3x3(out_channels[1][0],out_channels[1][1]),
            bn(out_channels[1][1]),
            nn.ReLU()
        )
        self.conv3_1 = nn.Sequential(
            conv1x1(in_channels,out_channels[2][0]),
            bn(out_channels[2][0]),
            nn.ReLU()
        )
        self.conv3_2 = nn.Sequential(
            conv5x5(out_channels[2][0],out_channels[2][1]),
            bn(out_channels[2][1]),
            nn.ReLU()
        )
        self.maxpool = maxpool(1,1)
        self.conv4 = nn.Sequential(
            conv1x1(in_channels,out_channels[3]),
            bn(out_channels[3]),
            nn.ReLU()
        )
        
    def forward(self,x):
        out1 = self.conv1(x)

        out2 = self.conv2_1(x)
        out2 = self.conv2_2(out2)

        out3 = self.conv3_1(x)
        out3 = self.conv3_2(out3)

        out4 = self.maxpool(x)
        out4 = self.conv4(out4)

        out = torch.cat([out1,out2,out3,out4],dim=1)
        return out

class Inception_v2_1(nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super(Inception_v1,self).__init__()
        self.conv1 = nn.Sequential(
            conv1x1(in_channels,out_channels[0]),
            bn(out_channels[0]),
            nn.ReLU()
        )
        self.conv2_1 = nn.Sequential(
            conv1x1(in_channels,out_channels[1][0]),
            bn(out_channels[1][0]),
            nn.ReLU()
        )
        self.conv2_2 = nn.Sequential(
            conv3x3(out_channels[1][0],out_channels[1][1]),
            bn(out_channels[1][1]),
            nn.ReLU()
        )
        self.conv3_1 = nn.Sequential(
            conv1x1(in_channels,out_channels[2][0]),
            bn(out_channels[2][0]),
            nn.ReLU()
        )
        self.conv3_2 = nn.Sequential(
            conv3x3(out_channels[2][0],out_channels[2][1]),
            bn(out_channels[2][1]),
            nn.ReLU(),
            conv3x3(out_channels[2][1],out_channels[2][1], stride=stride,padding=1),
            bn(out_channels[2][1]),
            nn.ReLU()
        )
        self.maxpool = maxpool(stride=stride,1)
        self.conv4 = nn.Sequential(
            conv1x1(in_channels,out_channels[3]),
            bn(out_channels[3]),
            nn.ReLU()
        )
        
    def forward(self,x):
        out1 = self.conv1(x)

        out2 = self.conv2_1(x)
        out2 = self.conv2_2(out2)

        out3 = self.conv3_1(x)
        out3 = self.conv3_2(out3)

        out4 = self.maxpool(x)
        out4 = self.conv4(out4)

        out = torch.cat([out1,out2,out3,out4],dim=1)
        return out


class Inception_v2_2(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_frac_conv):
        super(Inception_v1,self).__init__()
        self.conv1 = nn.Sequential(
            conv1x1(in_channels,out_channels[0]),
            bn(out_channels[0]),
            nn.ReLU()
        )
        self.conv2_1 = nn.Sequential(
            conv1x1(in_channels,out_channels[1][0]),
            bn(out_channels[1][0]),
            nn.ReLU()
        )
        self.conv2_2 = nn.Sequential(
            conv1xn(out_channels[1][0],out_channels[1][1],(1,kernel_frac_conv)),
            bn(out_channels[1][1]),
            nn.ReLU()
            convnx1(out_channels[1][1],out_channels[1][1],(kernel_frac_conv,1)),
            bn(out_channels[1][1]),
            nn.ReLU()
        )
        self.conv3_1 = nn.Sequential(
            conv1x1(in_channels,out_channels[2][0]),
            bn(out_channels[2][0]),
            nn.ReLU()
        )
        self.conv3_2 = nn.Sequential(
            conv1xn(out_channels[2][0],out_channels[2][1],(1,kernel_frac_conv)),
            bn(out_channels[2][1]),
            nn.ReLU(),
            convnx1(out_channels[2][1],out_channels[2][1],(kernel_frac_conv,1)),
            bn(out_channels[2][1]),
            nn.ReLU(),
            conv1xn(out_channels[2][1],out_channels[2][1],(1,kernel_frac_conv)),
            bn(out_channels[2][1]),
            nn.ReLU(),
            convnx1(out_channels[2][1],out_channels[2][1],(kernel_frac_conv,1)),
            bn(out_channels[2][1]),
            nn.ReLU()
        )
        self.maxpool = maxpool(1,1)
        self.conv4 = nn.Sequential(
            conv1x1(in_channels,out_channels[3]),
            bn(out_channels[3]),
            nn.ReLU()
        )
        
    def forward(self,x):
        out1 = self.conv1(x)

        out2 = self.conv2_1(x)
        out2 = self.conv2_2(out2)

        out3 = self.conv3_1(x)
        out3 = self.conv3_2(out3)

        out4 = self.maxpool(x)
        out4 = self.conv4(out4)

        out = torch.cat([out1,out2,out3,out4],dim=1)
        return out


class Inception_v2_3(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_frac_conv):
        super(Inception_v1,self).__init__()
        self.conv1 = nn.Sequential(
            conv1x1(in_channels,out_channels[0]),
            bn(out_channels[0]),
            nn.ReLU()
        )
        self.conv2_1 = nn.Sequential(
            conv1x1(in_channels,out_channels[1][0]),
            bn(out_channels[1][0]),
            nn.ReLU()
        )
        self.conv2_2_1 = nn.Sequential(
            conv1xn(out_channels[1][0],out_channels[1][1],(1,kernel_frac_conv)),
            bn(out_channels[1][1]),
            nn.ReLU()
        )
        self.conv2_2_2 = nn.Sequential(
            convnx1(out_channels[1][1],out_channels[1][1],(kernel_frac_conv,1)),
            bn(out_channels[1][1]),
            nn.ReLU()
        )

        self.conv3_1 = nn.Sequential(
            conv1x1(in_channels,out_channels[2][0]),
            bn(out_channels[2][0]),
            nn.ReLU()
        )
        self.conv3_2 = nn.Sequential(
            conv3x3(out_channels[2][0],out_channels[2][1]),
            bn(out_channels[2][1]),
            nn.ReLU()
        )
        self.conv3_3_1 = nn.Sequential(
            conv1xn(out_channels[2][1],out_channels[2][1],(1,kernel_frac_conv)),
            bn(out_channels[2][1]),
            nn.ReLU()
        )
        self.conv3_3_2 = nn.Sequential(
            convnx1(out_channels[2][1],out_channels[2][1],(kernel_frac_conv,1)),
            bn(out_channels[2][1]),
            nn.ReLU()
        )
        self.maxpool = maxpool(1,1)
        self.conv4 = nn.Sequential(
            conv1x1(in_channels,out_channels[3]),
            bn(out_channels[3]),
            nn.ReLU()
        )
        
    def forward(self,x):
        out1 = self.conv1(x)

        out2 = self.conv2_1(x)
        out2_1 = self.conv2_2_1(out2)
        out2_2 = self.conv2_2_2(out2)

        out3 = self.conv3_1(x)
        out3 = self.conv3_2(out3)
        out3_1 = self.conv3_3_1(out3)
        out3_2 = self.conv3_3_2(out3)

        out4 = self.maxpool(x)
        out4 = self.conv4(out4)

        out = torch.cat([out1,out2_1,out2_2,out3_1,out3_2,out4],dim=1)
        return out


class GoogLeNet(nn.Module):
    def __init__(self,classes_num):
        super(GoogLeNet,self).__init__()
        self.conv1 = conv7x7(3,64)
        self.bn1 = bn(64)
        self.relu = nn.ReLU()
        self.maxpool1 = maxpool()

        self.conv2 = conv1x1(64,64)
        self.bn2 = bn(64)
        self.conv3 = conv3x3(64,192,1,1)
        self.bn3 = bn(192)
        self.maxpool2 = maxpool()

        self.inception3_a = Inception_v1(192,[64,[96,128],[16,32],32])
        self.inception3_b = Inception_v1(256,[128,[128,192],[32,96],64])
        self.maxpool3 = maxpool()

        self.inception4_a = Inception_v1(480,[192,[96,208],[16,48],64])
        self.inception4_b = Inception_v1(512,[160,[112,224],[24,64],64])
        self.inception4_c = Inception_v1(512,[128,[128,256],[24,64],64])
        self.inception4_d = Inception_v1(512,[112,[144,288],[32,64],64])
        self.inception4_e = Inception_v1(528,[256,[160,320],[32,128],128])
        self.maxpool4 = maxpool()

        self.inception5_a = Inception_v1(832,[256,[160,320],[32,128],128])
        self.inception5_b = Inception_v1(832,[384,[192,384],[48,128],128])
        
        self.avgpool1 = gap(5,3)
        self.conv4 = conv1x1(512,128)
        self.bn4 = bn(128)
        self.dropout1 = nn.Dropout2d(p=0.7)
        self.fc1_1 = fc(4*4*128,1024)
        self.fc1_2 = fc(1024,classes_num)

        self.avgpool2 = gap(5,3)
        self.conv5 = conv1x1(528,128)
        self.bn5 = bn(128)
        self.dropout2 = nn.Dropout2d(p=0.7)
        self.fc2_1 = fc(4*4*128,1024)
        self.fc2_2 = fc(1024,classes_num)

        self.avgpool3 = gap(7)
        self.dropout3 = nn.Dropout2d(p=0.4)
        self.fc3 = fc(1024,classes_num)

    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.maxpool2(out)

        out = self.inception3_a(out)
        out = self.inception3_b(out)
        out = self.maxpool3(out)

        out = self.inception4_a(out)

        out1 = self.avgpool1(out)
        out1 = self.conv4(out1)
        out1 = self.bn4(out1)
        out1 = self.relu(out1)
        out1 = out1.view(out1.size(0),-1)
        out1 = self.dropout1(out1)
        out1 = self.fc1_1(out1)
        out1 = self.fc1_2(out1)

        out = self.inception4_b(out)
        out = self.inception4_c(out)
        out = self.inception4_d(out)

        out2 = self.avgpool2(out)
        out2 = self.conv5(out2)
        out2 = self.bn5(out2)
        out2 = self.relu(out2)
        out2 = out2.view(out2.size(0),-1)
        out2 = self.dropout2(out2)
        out2 = self.fc2_1(out2)
        out2 = self.fc2_2(out2)

        out = self.inception4_e(out)
        out = self.maxpool4(out)

        out = self.inception5_a(out)
        out = self.inception5_b(out)

        out = self.avgpool3(out)
        out =  out.view(out.size(0),-1)
        out = self.dropout3(out)
        out = self.fc3(out)
        return out1, out2, out

    def initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data,0,0.01)
                m.bias.data.zero_()

def main():
    model = GoogLeNet(1000)
    x = Variable(torch.rand((8,3,224,224)))
    log_path = './summary/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    writer = SummaryWriter(log_dir=log_path,comment='GoogLeNet')
    writer.add_graph(model,(x,),verbose=True)
    writer.close()
    return

if __name__ == "__main__":
    main()