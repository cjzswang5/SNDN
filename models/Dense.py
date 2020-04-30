#from .NonLocalBlock2D    import NonLocalBlock2D as NonLocalBlock2D
from .SimpleNLBlock2D   import SimpleNLBlock2D as NonLocalBlock2D
from torch              import nn, cat
from torchvision.models import densenet121
from torch.nn           import functional as F

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
#        self.nonlocalblock = NonLocalBlock2D(in_planes)
        self.conv1 = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_planes, inter_planes, 1, 1, 0, bias=False))
        self.conv2 = nn.Sequential(
                nn.BatchNorm2d(inter_planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_planes, out_planes, 3, 1, 1, bias=False))
        self.droprate = dropRate

    def forward(self, x):
#        out = self.conv1(self.nonlocalblock(x))
        out = self.conv1(x)
        if self.droprate > 0.0:
            out = F.dropout(out, self.droprate, self.training, False)
        out = self.conv2(out)
        if self.droprate > 0.0:
            out = F.dropout(out, self.droprate, self.training, False)
        return cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.conv1 = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_planes, out_planes, 1, 1, 0, bias=False))
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(x)
        if self.droprate > 0:
            out = F.dropout(out, self.droprate, self.training, False)
        return out

class Dense(nn.Module):
    def __init__(self):
        super(Dense, self).__init__()

        self.nonlocalblock1 = NonLocalBlock2D(64)
        self.nonlocalblock2 = NonLocalBlock2D(128)
        self.nonlocalblock3 = NonLocalBlock2D(256)

        ############# H -> ceil(H/4) ##############
        haze_class = densenet121(pretrained=True)

        self.conv0=haze_class.features.conv0
        self.norm0=haze_class.features.norm0
        self.relu0=haze_class.features.relu0
        self.pool0=haze_class.features.pool0

        ############# Block1-down H -> floor(H/2) ##############
        self.dense_block1=haze_class.features.denseblock1
        self.trans_block1=haze_class.features.transition1

        ############# Block2-down H -> floor(H/2) ##############
        self.dense_block2=haze_class.features.denseblock2
        self.trans_block2=haze_class.features.transition2

        ############# Block3-down H -> floor(H/2) ##############
        self.dense_block3=haze_class.features.denseblock3
        self.trans_block3=haze_class.features.transition3

        ############# Block4-up H -> 2H ##############
        self.dense_block4=BottleneckBlock(512,256)
        self.trans_block4=TransitionBlock(512+256,128)

        ############# Block5-up H -> 2H ##############
        self.dense_block5=BottleneckBlock(384,256)
        self.trans_block5=TransitionBlock(384+256,128)

        ############# Block6-up H -> 2H ##############
        self.dense_block6=BottleneckBlock(256,128)
        self.trans_block6=TransitionBlock(256+128,64)


        ############# Block7-up H -> 2H ##############
        self.dense_block7=BottleneckBlock(64,64)
        self.trans_block7=TransitionBlock(64+64,32)

        ############# Block8-up H -> 2H ##############
        self.dense_block8=BottleneckBlock(32,32)
        self.trans_block8=TransitionBlock(32+32,16)

        self.conv_refin=nn.Conv2d(19,20,3,1,1)
        self.tanh=nn.Tanh()


        self.conv1010 = nn.Conv2d(20, 1, 1, 1, 0)
        self.conv1020 = nn.Conv2d(20, 1, 1, 1, 0)
        self.conv1030 = nn.Conv2d(20, 1, 1, 1, 0)
        self.conv1040 = nn.Conv2d(20, 1, 1, 1, 0)

        self.refine3 = nn.Conv2d(20+4, 1, 3, 1, 1)
        self.refine4 = nn.Conv2d(   4, 3, 1, 1, 0)

        self.upsample = F.interpolate

        self.relu=nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        #print('x ', list(x.size()))
        # HxW -> ceil(H/4) x ceil(W/4)
        x0 = self.pool0(self.relu0(self.norm0(self.conv0(x))))
        #print('x0', list(x0.size()))

        # HxW -> floor(H/2) x floor(W/2)
        x1 = self.trans_block1(self.dense_block1(self.nonlocalblock1(x0)))
        #print('x1', list(x1.size()))

        # HxW -> floor(H/2) x floor(W/2)
        x2 = self.trans_block2(self.dense_block2(self.nonlocalblock2(x1)))
        #print('x2', list(x2.size()))

        # HxW -> floor(H/2) x floor(W/2)
        x3 = self.trans_block3(self.dense_block3(self.nonlocalblock3(x2)))

        x4 = self.trans_block4(self.dense_block4(x3))
        x4 = self.upsample(x4, size=x2.size()[2:])

        x42 = cat([x4,x2], 1)

        # HxW -> floor(H/2) x floor(W/2)
        x5=self.trans_block5(self.dense_block5(x42))
        x5 = self.upsample(x5, size=x1.size()[2:])

        x51=cat([x5,x1], 1)
        ##  32 X 32
        x6=self.trans_block6(self.dense_block6(x51))
        x6 = self.upsample(x6, scale_factor=2)

        ##  64 X 64
        x7=self.trans_block7(self.dense_block7(x6))
        x7 = self.upsample(x7, scale_factor=2)

        ##  128 X 128
        x8=self.trans_block8(self.dense_block8(x7))
        x8 = self.upsample(x8, size=x.size()[2:])

        x8=cat([x8,x], 1)

        x9=self.relu(self.conv_refin(x8))

        shape_out = x9.data.size()
        shape_out = shape_out[2:]

        x101 = F.avg_pool2d(x9, 32)
        x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9,  8)
        x104 = F.avg_pool2d(x9,  4)

        x1010 = self.upsample(self.relu(self.conv1010(x101)), size=shape_out)
        x1020 = self.upsample(self.relu(self.conv1020(x102)), size=shape_out)
        x1030 = self.upsample(self.relu(self.conv1030(x103)), size=shape_out)
        x1040 = self.upsample(self.relu(self.conv1040(x104)), size=shape_out)

        dehaze = cat((x1010, x1020, x1030, x1040, x9), 1)
        dehaze = self.refine3(dehaze)

        x11 = cat((x, dehaze), 1)

        k = self.tanh(self.refine4(x11))

        return k

if __name__ == '__main__':
    import torch, time
    tmp = torch.rand(4, 3, 256, 256)
    net = Dense()
    t = time.time()
    tmp_o = net(tmp)
    print(tmp.size(), '->', tmp_o.size())
    print(time.time() - t)

