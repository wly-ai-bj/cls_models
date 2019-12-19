import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import os

def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False)

def conv1x1(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)

def bn(num_features):
    return nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)

def maxpool():
    return nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)

def fc(in_features, out_features):
    return nn.Linear(in_features, out_features, bias=True)

class stage(nn.Module):
    def __init__(self,in_channels, out_channels, has_conv1x1=False):
        super(stage,self).__init__()
        self.conv1 = conv3x3(in_channels,out_channels)
        self.bn1 = bn(out_channels)
        self.relu = nn.ReLU()       
        self.conv2 = conv3x3(out_channels,out_channels)
        self.bn2 = bn(out_channels)      
        self.maxpool = maxpool()
        self.has_conv1x1 = has_conv1x1
        self.conv1x1 = conv1x1(out_channels,out_channels)
        self.bn3 = bn(out_channels)
       
    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.has_conv1x1:
            out = self.conv1x1(out)
            out = self.bn3(out)
            out = self.relu(out)
        out = self.maxpool(out)
        return out


class VGG(nn.Module):
    def __init__(self,classes_num):
        super(VGG,self).__init__()
        self.stage1 = stage(3,64)
        self.stage2 = stage(64,128)
        self.stage3 = stage(128,256,has_conv1x1=True)
        self.stage4 = stage(256,512,has_conv1x1=True)
        self.stage5 = stage(512,512,has_conv1x1=True)
        self.fc1 = fc(7*7*512,4096)
        self.fc2 = fc(4096,4096)
        self.fc3 = fc(4096,classes_num)
        self.dropout = nn.Dropout2d()

    def forward(self,x):
        out = self.stage1(x)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)
        out = out.view(out.size(0),-1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

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
    log_path = './summary/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    model = VGG(1000)

    x = Variable(torch.rand((8,3,224,224)))
    writer = SummaryWriter(log_dir=log_path,comment='VGG16')
    writer.add_graph(model,(x,))
    writer.close()
    return

if __name__ == "__main__":
    main()