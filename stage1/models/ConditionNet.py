import torch
import torch.nn as nn
import torch.nn.functional as F
from models.spectral_normalization import SpectralNorm


class Self_Attention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = SpectralNorm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 1, kernel_size=1))
        self.key_conv = SpectralNorm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 1, kernel_size=1))
        self.value_conv = SpectralNorm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1))
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
        inputs :
            x : input feature maps( B X C X W X H)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


class ConditionNet(nn.Module):
    """
    Condition Network, input GT img and output color prior for colorization.
    """

    def __init__(self, ndf=32):
        super(ConditionNet, self).__init__()
        self.in_size = 3
        self.ndf = ndf

        self.layer1 = nn.Sequential(
            nn.Conv2d(self.in_size, self.ndf, 4, 2, 1), 
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.ndf, self.ndf, 4, 2, 1),
            nn.InstanceNorm2d(self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1),
            nn.InstanceNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1),
            nn.InstanceNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.attention = Self_Attention(self.ndf*4)
        self.layer5 = nn.Sequential(
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1),
            nn.InstanceNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(self.ndf * 8, self.ndf * 16, 4, 2, 1),
            nn.InstanceNorm2d(self.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.last_new = nn.Conv2d(self.ndf * 16, 4, 3, 1, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        feature1 = self.layer1(input)     # 216 384 -> 108 192
        feature2 = self.layer2(feature1)     # 108 192->54 96
        feature3 = self.layer3(feature2)  # 54 96 -> 27 48
        feature4 = self.layer4(feature3)           # 27 48 -> 13 24
        feature_attention = self.attention(feature4)
        feature5 = self.layer5(feature_attention)           # 13 24 -> 6 12
        feature6 = self.layer6(feature5)           # 6 12 -> 3 6
        output = self.last_new(feature6)  #b c 3 6 -> b 4 1 4 
        output = F.avg_pool2d(output, output.size()[2:]).view(output.size()[0], 4,1,1)  # b 1 1 1
        output = self.sigmoid(output)       # normalize to 0~1
        return output
