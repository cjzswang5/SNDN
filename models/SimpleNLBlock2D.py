from torch import nn, matmul

class SimpleNLBlock2D(nn.Module):
    """ Simplified Non-local module.

    See GCNet.

    Args:
        in_channels (int ): Channels of the input feature map.
    """

    def __init__(self, in_channels):
        super(SimpleNLBlock2D, self).__init__()

        self.in_c  = in_channels
        self.conv1 = nn.Conv2d(self.in_c, 1,         1, 1, 0)
        self.conv2 = nn.Conv2d(self.in_c, self.in_c, 1, 1, 0)

    def forward(self, x):
        n, c, h, w = x.shape
        # x1: [N, C, HW]
        x1 = x.view(n, c, -1)
        # x2: [N, HW, 1]
        x2 = self.conv1(x).view(n, -1, 1).softmax(dim=1)
        # x3: [N , C, 1, 1]
        x3 = matmul(x1, x2).unsqueeze(3)
        return  x + self.conv2(x3)

if __name__ == '__main__':

    # test for NonLocalBlock2D, 'tmp' shape should be same as 'tmp_o'.
    import torch
    tmp = torch.rand(5, 12, 16, 8)
    tmp_o = SimpleNLBlock2D(tmp.shape[1])(tmp)
    print(tmp.size(), '->', tmp_o.size())

