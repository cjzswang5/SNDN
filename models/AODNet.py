from torch import nn, cat, relu

class AODNet(nn.Module):
    def __init__(self):
        super(AODNet, self).__init__()
        self.conv1 = nn.Conv2d( 3, 3, 1, 1, 0)
        self.conv2 = nn.Conv2d( 3, 3, 3, 1, 1)
        self.conv3 = nn.Conv2d( 6, 3, 5, 1, 2)
        self.conv4 = nn.Conv2d( 6, 3, 7, 1, 3)
        self.conv5 = nn.Conv2d(12, 3, 3, 1, 1)
        self.b = 1

    def forward(self, x):
        x1 = relu(self.conv1(x))
        x2 = relu(self.conv2(x1))
        cat1 = cat((x1, x2), 1)
        x3 = relu(self.conv3(cat1))
        cat2 = cat((x2, x3), 1)
        x4 = relu(self.conv4(cat2))
        cat3 = cat((x1, x2, x3, x4), 1)
        k = relu(self.conv5(cat3))
        return k

if __name__ == '__main__':
    import torch
    tmp = torch.rand(3, 3, 5, 6)
    net = AODNet()
    tmp_o = net(tmp)
    print(tmp.size(), '->', tmp_o.size())

