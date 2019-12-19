
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

import os

def conv11x11(in_channels, out_channels, stride=4, padding=2):
    return nn.Conv2d(in_channels, out_channels, kernel_size=11, stride=stride, padding=padding, dilation=1, groups=1, bias=True)

def conv5x5(in_channels, out_channels, stride=2, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=padding, dilation=1, groups=1, bias=True)

def conv3x3(in_channels, out_channels, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, dilation=1, groups=1, bias=True)

def overlap_pool(kernel_size, stride, padding=2):
    return nn.MaxPool2d(kernel_size, stride=stride, padding=padding, dilation=1, return_indices=False, ceil_mode=False)

def fc(in_features, out_features):
    return nn.Linear(in_features, out_features, bias=True)

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.conv1 = nn.Sequential(
            conv11x11(3,48),
            nn.ReLU(inplace=True),
            # overlap_pool(3,1)
            # conv5x5(48,128),
            # nn.ReLU(inplace=True),
            # # overlap_pool(3,1)
            # conv3x3(128,192,2,0),
            # nn.ReLU(inplace=True),
            # # overlap_pool(3,2)
            # conv3x3(192,192),
            # nn.ReLU(inplace=True),
            # # overlap_pool(3,1)
            # conv3x3(192,128),
            # nn.ReLU(inplace=True)
            # # overlap_pool(3,1)
        )

        self.conv2 = nn.Sequential(
            conv5x5(48,128),
            nn.ReLU(inplace=True)
            # overlap_pool(3,1)
        )
        
        self.conv3 = nn.Sequential(
            conv3x3(128,192,2,0),
            nn.ReLU(inplace=True)
            # overlap_pool(3,2)
        )
        
        self.conv4 = nn.Sequential(
            conv3x3(192,192),
            nn.ReLU(inplace=True)
            # overlap_pool(3,1)
        )

        self.conv5 = nn.Sequential(
            conv3x3(192,128),
            nn.ReLU(inplace=True)
            # overlap_pool(3,1)
        )

        self.fc = nn.Sequential(
            nn.Dropout2d(),
            fc(13*13*128,2048),
            nn.Dropout2d(),
            fc(2048,2048)
        )

        self.cls = fc(2048,1000)

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        out = self.cls(out)

        return out

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
        return

def main():
    x = torch.rand(size=(8,3,224,224))
    x = Variable(x)
    log_path = './summary/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    model = AlexNet()
    writer = SummaryWriter(log_dir=log_path, comment='AlexNet')
    writer.add_graph(model,input_to_model=(x,),verbose=False)
    writer.close()
    return

if __name__ == "__main__":
    main()
    