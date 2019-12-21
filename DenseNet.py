import torch 
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

class CONV(nn.Module):
    def __init__(self,in_channels, out_channels,kernel_size,stride,padding):
        super(CONV, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self,x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class Dense_B_layer(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(Dense_B_layer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv1 = CONV(in_channels, out_channels, 1, 1, 0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

    def forward(self,x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.conv2(out)
        out = torch.cat([out, x],dim=1)
        return out

class Dense_block(nn.Module):
    def __init__(self,in_channels, out_channels, block_scale):
        super(Dense_block,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_scale = block_scale
        self.layers = self.make_layer()      
             
    def forward(self,x):
        # out = self.layer1(x)
        # layers = self.make_layer()
        for i in range(len(self.layers)):
            if i == 0:
                out = self.layers[i](x)
            else:
                out = self.layers[i](out)
        return out[:,0: self.out_channels,:,:]
        
    def make_layer(self):
        layer = []
        for i in range(self.block_scale):
            layer.append(Dense_B_layer((self.in_channels + i*self.out_channels), self.out_channels))
        return nn.Sequential(*layer)


class Transition(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Transition,self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels,out_channels,1,1,0,bias=False)
        self.avgpool = nn.AvgPool2d(2,2)

    def forward(self,x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)
        out = self.avgpool(out)
        return out


class DenseNet(nn.Module):
    def __init__(self, k, theta, block_scale, class_num):
        super(DenseNet,self).__init__()
        self.conv1 = CONV(3,int(2*k),7,2,3)
        self.pool = nn.MaxPool2d(3,2,1)
        self.denseblock1 = Dense_block(int(2*k), k, block_scale[0])
        self.transition1 = Transition(k, int(theta*k))
        self.denseblock2 = Dense_block(int(theta*k), k, block_scale[1])
        self.transition2 = Transition(k, int(theta*k))
        self.denseblock3 = Dense_block(int(theta*k), k, block_scale[2])
        self.transition3 = Transition(k, int(theta*k))
        self.denseblock4 = Dense_block(int(theta*k), k, block_scale[3])
        self.bn = nn.BatchNorm2d(int(theta*k))
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout2d(p=0.5)
        self.fc = nn.Linear(k,class_num)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        out = self.conv1(x)
        out = self.pool(out)
        out = self.denseblock1(out)
        out = self.transition1(out)
        out = self.denseblock2(out)
        out = self.transition2(out)
        out = self.denseblock3(out)
        out = self.transition3(out)
        out = self.denseblock4(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.avgpool(out)
        # out = out.view(out.size(0),-1)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out 

def DenseNet_xxx(model_name, k, theta, class_num):
    if model_name == 'DenseNet-121':
        return DenseNet(k, theta, [6, 12, 24, 16], class_num)
    elif model_name == 'DenseNet-169':
        return DenseNet(k, theta, [6, 12, 32, 32], class_num)
    elif model_name == 'DenseNet-201':
        return DenseNet(k, theta, [6, 12, 48, 32], class_num)
    elif model_name == 'DenseNet-264':
        return DenseNet(k, theta, [6, 12, 64, 48], class_num)
    else:
        raise Exception('The {} model does not exsit!'.format(model_name))

def main():
    log_path = './summary/'
    writer = SummaryWriter(log_dir=log_path, comment='DenseNet')
    x = Variable(torch.rand(size=(8,3,224,224)))
    model = DenseNet_xxx('DenseNet-121', 12, 1, 1000)
    # model_conv = CONV(3,24,7,2,3)
    # model_dense_layer = Dense_B_layer(3,12)
    # model_dense_block = Dense_block(3,12,6)
    # model_trans = Transition(3,12)
    # model_dense = DenseNet(12,1,[6, 12, 24, 16],1000)
    writer.add_graph(model, (x,), verbose=True)
    writer.close()
    return

if __name__ == "__main__":
    main()