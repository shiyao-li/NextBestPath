import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class NBP(nn.Module):
    def __init__(self, img_ch=5, output_ch1=8, output_ch2=1):
        super(NBP, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)  # New encoder layer

        # decoder 1
        self.Up5_1 = up_conv(ch_in=1024, ch_out=512)  # New decoder layer
        self.Att5_1 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5_1 = conv_block(ch_in=1024, ch_out=512)

        self.Up4_1 = up_conv(ch_in=512, ch_out=256)
        self.Att4_1 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4_1 = conv_block(ch_in=512, ch_out=256)

        self.Final1 = nn.Conv2d(256, output_ch1, kernel_size=1)

        # decoder 2
        self.Up5_2 = up_conv(ch_in=1024, ch_out=512)  # New decoder layer
        self.Att5_2 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5_2 = conv_block(ch_in=1024, ch_out=512)

        self.Up4_2 = up_conv(ch_in=512, ch_out=256)
        self.Att4_2 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4_2 = conv_block(ch_in=512, ch_out=256)

        self.Up3_2 = up_conv(ch_in=256, ch_out=128)
        self.Att3_2 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3_2 = conv_block(ch_in=256, ch_out=128)

        self.Up2_2 = up_conv(ch_in=128, ch_out=64)
        self.Att2_2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2_2 = conv_block(ch_in=128, ch_out=64)

        self.Final2 = nn.Sequential(
            nn.Conv2d(64, output_ch2, kernel_size=1),
            nn.Sigmoid()
        )
        self.log_vars = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)  # New encoder layer
        x5 = self.Conv5(x5)

        # Decoder 1（64x64x8）
        d5_1 = self.Up5_1(x5)  # New decoder layer
        x4_1 = self.Att5_1(g=d5_1, x=x4)
        d5_1 = torch.cat((x4_1, d5_1), dim=1)
        d5_1 = self.Up_conv5_1(d5_1)

        d4_1 = self.Up4_1(d5_1)
        x3_1 = self.Att4_1(g=d4_1, x=x3)
        d4_1 = torch.cat((x3_1, d4_1), dim=1)
        d4_1 = self.Up_conv4_1(d4_1)
        out1 = self.Final1(d4_1)

        # Decoder 2
        d5_2 = self.Up5_2(x5)  # New decoder layer
        x4_2 = self.Att5_2(g=d5_2, x=x4)
        d5_2 = torch.cat((x4_2, d5_2), dim=1)
        d5_2 = self.Up_conv5_2(d5_2)

        d4_2 = self.Up4_2(d5_2)
        x3_2 = self.Att4_2(g=d4_2, x=x3)
        d4_2 = torch.cat((x3_2, d4_2), dim=1)
        d4_2 = self.Up_conv4_2(d4_2)

        d3_2 = self.Up3_2(d4_2)
        x2_2 = self.Att3_2(g=d3_2, x=x2)
        d3_2 = torch.cat((x2_2, d3_2), dim=1)
        d3_2 = self.Up_conv3_2(d3_2)

        d2_2 = self.Up2_2(d3_2)
        x1_2 = self.Att2_2(g=d2_2, x=x1)
        d2_2 = torch.cat((x1_2, d2_2), dim=1)
        d2_2 = self.Up_conv2_2(d2_2)

        out2 = self.Final2(d2_2)

        return out1, out2

    def loss(self, pred1, target1, pred2, target2):
        # Calculate σ₁² and σ₂²
        sigma1_squared = torch.exp(2 * self.log_vars[0])  # σ₁² = exp(2 * log_vars[0])
        sigma2_squared = torch.exp(2 * self.log_vars[1])  # σ₂² = exp(2 * log_vars[1])
        
        # MSE loss term: (1/(2σ₁²)) * L_MSE - added missing 1/2 coefficient
        loss1 = (1.0 / (2.0 * sigma1_squared)) * F.mse_loss(pred1, target1) + self.log_vars[0]
        
        # BCE loss term: (1/σ₂²) * L_BCE - corrected coefficient  
        loss2 = (1.0 / sigma2_squared) * F.binary_cross_entropy(pred2, target2) + self.log_vars[1]

        return loss1 + loss2
