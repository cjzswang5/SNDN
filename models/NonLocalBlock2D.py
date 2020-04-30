from torch import nn, matmul

class NonLocalBlock2D(nn.Module):
    """Non-local module.

    See https://arxiv.org/abs/1711.07971 for details.

    Args:
        in_channels (int ): Channels of the input feature map.
        reduction   (int ): Channel reduction ratio.
        use_scale   (bool): Whether to scale pairwise_weight by
                            1/inter_channels.
        mode        (str ): Options are `embedded_gaussian` and `dot_product`.
    """

    def __init__(self, in_channels, reduction=2, use_scale=True,
                 mode='embedded_gaussian'):
        super(NonLocalBlock2D, self).__init__()

        self.in_channels    = in_channels
        self.reduction      = reduction
        self.use_scale      = use_scale
        self.inter_channels = in_channels // reduction
        self.mode           = mode

        assert mode in ['embedded_gaussian', 'dot_product']

        self.conv_g     = self.conv1x1(self.in_channels, self.inter_channels)
        self.conv_theta = self.conv1x1(self.in_channels, self.inter_channels)
        self.conv_phi   = self.conv1x1(self.in_channels, self.inter_channels)
        self.conv_out   = self.conv1x1(self.inter_channels, self.in_channels)

    def conv1x1(self, in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def embedded_gaussian(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = matmul(theta_x, phi_x)
        if self.use_scale:
            # theta_x.shape[-1] is `self.inter_channels`
            pairwise_weight /= theta_x.shape[-1] ** -0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def dot_product(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def forward(self, x):
        n, _, h, w = x.shape

        # g_x: [N, HxW, C_inter]
        g_x = self.conv_g(x).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta_x: [N, HxW, C_inter]
        theta_x = self.conv_theta(x).view(n, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        # phi_x: [N, C_inter, HxW]
        phi_x = self.conv_phi(x).view(n, self.inter_channels, -1)

        # pairwise_weight: [N, HxW, HxW]
        pairwise_func = getattr(self, self.mode)
        pairwise_weight = pairwise_func(theta_x, phi_x)

        # y: [N, HxW, C_inter]
        y = matmul(pairwise_weight, g_x)
        # y: [N, C_inter, H, W]
        y = y.permute(0, 2, 1).reshape(n, self.inter_channels, h, w)

        output = x + self.conv_out(y)

        return output

if __name__ == '__main__':

    # test for NonLocalBlock2D, 'tmp' shape should be same as 'tmp_o'.
    import torch
    tmp = torch.rand(5, 12, 16, 8)
    net = NonLocalBlock2D(tmp.shape[1])
    tmp_o = net(tmp)
    print(tmp_o.size())

