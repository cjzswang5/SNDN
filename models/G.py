from torch                  import nn, cat, relu
from .Dense                 import Dense
from .AODNet                import AODNet
from torch.nn.functional    import interpolate

class Generator(nn.Module):
    def __init__(self, b=1.0):
        super(Generator, self).__init__()
        self.Dense = Dense()
        self.b = b
        self.upsample = interpolate
        self.AODNet2 = AODNet()
        self.conv1 = nn.Conv2d(6, 3, 3, 1, 1)
        self.conv2 = nn.Conv2d(3, 3, 3, 1, 1)
        self.conv3 = nn.Conv2d(3, 3, 3, 1, 1)

    def forward(self, x):
        # downsample into 1/4 size
        x_2 = self.upsample(x, scale_factor=0.5 )
        #x_4 = self.upsample(x, scale_factor=0.25)

        tmp_k1 = self.Dense(x)
        tmp_k2 = self.upsample(self.AODNet2(x_2), size=tmp_k1.size()[2:])
        #tmp_k4 = self.upsample(self.AODNet4(x_4), size=tmp_k1.size()[2:])

        # k = tmp_k1
        #k = cat((tmp_k1, tmp_k2, tmp_k4), 1)
        k = cat((tmp_k1, tmp_k2), 1)

        k = relu(self.conv1(k))
        k = relu(self.conv2(k))
        k = relu(self.conv3(k))

        output = k * x - k + self.b
        output = relu(output)

        # for convinet show K(x). rescale k to [0, 1].
        # It doesn't affect tarining and testing.
        k = k - k.min()
        k = k / k.max()

        return output, k

if __name__ == '__main__':
    import torch, time
    tmp = torch.rand(4, 3, 256, 256)
    t = time.time()
    for i in range(10):
        tmp, tmp_k = Generator()(tmp)
    print((time.time() - t) / 10)

