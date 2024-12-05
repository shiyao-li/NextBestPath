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

class UNet(nn.Module):
    def __init__(self, img_ch=5, output_ch1=8, output_ch2=1):
        super(UNet, self).__init__()

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
            precision1 = torch.exp(-self.log_vars[0])
            loss1 = precision1 * F.mse_loss(pred1, target1) + self.log_vars[0]

            precision2 = torch.exp(-self.log_vars[1])
            loss2 = precision2 * F.binary_cross_entropy(pred2, target2) + self.log_vars[1]

            return loss1 + loss2
# class UNet(nn.Module):
#     def __init__(self, img_ch=5, output_ch1=8, output_ch2=1):
#         super(UNet, self).__init__()

#         self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
#         self.Conv2 = conv_block(ch_in=64, ch_out=128)
#         self.Conv3 = conv_block(ch_in=128, ch_out=256)
#         self.Conv4 = conv_block(ch_in=256, ch_out=512)

#         # 解码器1：用于64x64x8的输出
#         self.Up4_1 = up_conv(ch_in=512, ch_out=256)
#         self.Att4_1 = Attention_block(F_g=256, F_l=256, F_int=128)
#         self.Up_conv4_1 = conv_block(ch_in=512, ch_out=256)

#         self.Final1 = nn.Conv2d(256, output_ch1, kernel_size=1)

#         # 解码器2：用于256x256x1的二值化输出
#         self.Up4_2 = up_conv(ch_in=512, ch_out=256)
#         self.Att4_2 = Attention_block(F_g=256, F_l=256, F_int=128)
#         self.Up_conv4_2 = conv_block(ch_in=512, ch_out=256)

#         self.Up3_2 = up_conv(ch_in=256, ch_out=128)
#         self.Att3_2 = Attention_block(F_g=128, F_l=128, F_int=64)
#         self.Up_conv3_2 = conv_block(ch_in=256, ch_out=128)

#         self.Up2_2 = up_conv(ch_in=128, ch_out=64)
#         self.Att2_2 = Attention_block(F_g=64, F_l=64, F_int=32)
#         self.Up_conv2_2 = conv_block(ch_in=128, ch_out=64)

#         self.Final2 = nn.Sequential(
#             nn.Conv2d(64, output_ch2, kernel_size=1),
#             nn.Sigmoid()
#         )
#         self.log_vars = nn.Parameter(torch.zeros(2))

#     def forward(self, x):
#         # 编码路径
#         x1 = self.Conv1(x)

#         x2 = self.Maxpool(x1)
#         x2 = self.Conv2(x2)

#         x3 = self.Maxpool(x2)
#         x3 = self.Conv3(x3)
        
#         x4 = self.Maxpool(x3)
#         x4 = self.Conv4(x4)

#         # 解码器1路径（64x64x8输出）
#         d4_1 = self.Up4_1(x4)
#         x3_1 = self.Att4_1(g=d4_1, x=x3)
#         d4_1 = torch.cat((x3_1, d4_1), dim=1)
#         d4_1 = self.Up_conv4_1(d4_1)
#         out1 = self.Final1(d4_1)

#         # 解码器2路径（256x256x1二值化输出）
#         d4_2 = self.Up4_2(x4)
#         x3_2 = self.Att4_2(g=d4_2, x=x3)
#         d4_2 = torch.cat((x3_2, d4_2), dim=1)
#         d4_2 = self.Up_conv4_2(d4_2)

#         d3_2 = self.Up3_2(d4_2)
#         x2_2 = self.Att3_2(g=d3_2, x=x2)
#         d3_2 = torch.cat((x2_2, d3_2), dim=1)
#         d3_2 = self.Up_conv3_2(d3_2)

#         d2_2 = self.Up2_2(d3_2)
#         x1_2 = self.Att2_2(g=d2_2, x=x1)
#         d2_2 = torch.cat((x1_2, d2_2), dim=1)
#         d2_2 = self.Up_conv2_2(d2_2)

#         out2 = self.Final2(d2_2)

#         return out1, out2
    

    


#########################################################################################################

# unet transformer

####################################################################################################

# class conv_block(nn.Module):
#     def __init__(self, ch_in, ch_out):
#         super(conv_block, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.conv(x)

# class up_conv(nn.Module):
#     def __init__(self, ch_in, ch_out):
#         super(up_conv, self).__init__()
#         self.up = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.up(x)

# class Attention_block(nn.Module):
#     def __init__(self, F_g, F_l, F_int):
#         super(Attention_block, self).__init__()
#         self.W_g = nn.Sequential(
#             nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(F_int)
#         )

#         self.W_x = nn.Sequential(
#             nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(F_int)
#         )

#         self.psi = nn.Sequential(
#             nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )

#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, g, x):
#         g1 = self.W_g(g)
#         x1 = self.W_x(x)
#         psi = self.relu(g1 + x1)
#         psi = self.psi(psi)
#         return x * psi

# class TransformerBlock(nn.Module):
#     def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
#         super(TransformerBlock, self).__init__()
#         self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
#         self.ffn = nn.Sequential(
#             nn.Conv2d(embed_dim, ff_dim, 1),
#             nn.ReLU(),
#             nn.Conv2d(ff_dim, embed_dim, 1)
#         )
#         self.norm1 = nn.InstanceNorm2d(embed_dim)
#         self.norm2 = nn.InstanceNorm2d(embed_dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         b, c, h, w = x.shape
#         x_flat = x.flatten(2).permute(2, 0, 1)
#         attn_output, _ = self.attn(x_flat, x_flat, x_flat)
#         attn_output = attn_output.permute(1, 2, 0).view(b, c, h, w)
#         x = x + self.dropout(attn_output)
#         x = self.norm1(x)
        
#         ffn_output = self.ffn(x)
#         x = x + self.dropout(ffn_output)
#         x = self.norm2(x)
#         return x

# class PatchEmbedding(nn.Module):
#     def __init__(self, in_channels, embed_dim, patch_size=1):
#         super(PatchEmbedding, self).__init__()
#         self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

#     def forward(self, x):
#         return self.conv(x)
    
# class UNetTransformer(nn.Module):
#     def __init__(self, img_ch=5, output_ch1=8, output_ch2=1, embed_dim=512, num_heads=8, ff_dim=1024, num_transformer_layers=2):
#         super(UNetTransformer, self).__init__()

#         self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
#         self.Conv2 = conv_block(ch_in=64, ch_out=128)
#         self.Conv3 = conv_block(ch_in=128, ch_out=256)
#         self.Conv4 = conv_block(ch_in=256, ch_out=512)

#         self.patch_embed = PatchEmbedding(in_channels=512, embed_dim=embed_dim)

#         self.transformers = nn.Sequential(
#             *[TransformerBlock(embed_dim, num_heads, ff_dim, dropout=0.1) for _ in range(num_transformer_layers)]
#         )

#         # 解码器1：用于64x64x8的输出
#         self.Up4_1 = up_conv(ch_in=512, ch_out=256)
#         self.Att4_1 = Attention_block(F_g=256, F_l=256, F_int=128)
#         self.Up_conv4_1 = conv_block(ch_in=512, ch_out=256)

#         self.Final1 = nn.Conv2d(256, output_ch1, kernel_size=1)

#         # 解码器2：用于256x256x1的二值化输出
#         self.Up4_2 = up_conv(ch_in=512, ch_out=256)
#         self.Att4_2 = Attention_block(F_g=256, F_l=256, F_int=128)
#         self.Up_conv4_2 = conv_block(ch_in=512, ch_out=256)

#         self.Up3_2 = up_conv(ch_in=256, ch_out=128)
#         self.Att3_2 = Attention_block(F_g=128, F_l=128, F_int=64)
#         self.Up_conv3_2 = conv_block(ch_in=256, ch_out=128)

#         self.Up2_2 = up_conv(ch_in=128, ch_out=64)
#         self.Att2_2 = Attention_block(F_g=64, F_l=64, F_int=32)
#         self.Up_conv2_2 = conv_block(ch_in=128, ch_out=64)

#         self.Final2 = nn.Sequential(
#             nn.Conv2d(64, output_ch2, kernel_size=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         # 编码路径
#         x1 = self.Conv1(x)
#         x2 = self.Maxpool(x1)
#         x2 = self.Conv2(x2)
#         x3 = self.Maxpool(x2)
#         x3 = self.Conv3(x3)
#         x4 = self.Maxpool(x3)
#         x4 = self.Conv4(x4)

#         # # # Transformer
#         # x4 = self.patch_embed(x4)
#         # x4 = self.transformers(x4)

#         # 解码器1路径（64x64x8输出）
#         d4_1 = self.Up4_1(x4)
#         x3_1 = self.Att4_1(g=d4_1, x=x3)
#         d4_1 = torch.cat((x3_1, d4_1), dim=1)
#         d4_1 = self.Up_conv4_1(d4_1)

#         out1 = self.Final1(d4_1)

#         # 解码器2路径（256x256x1二值化输出）
#         d4_2 = self.Up4_2(x4)
#         x3_2 = self.Att4_2(g=d4_2, x=x3)
#         d4_2 = torch.cat((x3_2, d4_2), dim=1)
#         d4_2 = self.Up_conv4_2(d4_2)

#         d3_2 = self.Up3_2(d4_2)
#         x2_2 = self.Att3_2(g=d3_2, x=x2)
#         d3_2 = torch.cat((x2_2, d3_2), dim=1)
#         d3_2 = self.Up_conv3_2(d3_2)

#         d2_2 = self.Up2_2(d3_2)
#         x1_2 = self.Att2_2(g=d2_2, x=x1)
#         d2_2 = torch.cat((x1_2, d2_2), dim=1)
#         d2_2 = self.Up_conv2_2(d2_2)

#         out2 = self.Final2(d2_2)

#         return out1, out2

#########################################################################################################

# Attention unet

####################################################################################################

# class Attention_block(nn.Module):
#     def __init__(self, F_g, F_l, F_int):
#         super(Attention_block, self).__init__()
#         self.W_g = nn.Sequential(
#             nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(F_int)
#         )
        
#         self.W_x = nn.Sequential(
#             nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(F_int)
#         )

#         self.psi = nn.Sequential(
#             nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )
        
#         self.relu = nn.ReLU(inplace=True) # save the memory
        
#     def forward(self, g, x):
#         g1 = self.W_g(g)
#         x1 = self.W_x(x)
#         psi = self.relu(g1 + x1)
#         psi = self.psi(psi)
#         return x * psi

# class conv_block(nn.Module):
#     def __init__(self, ch_in, ch_out):
#         super(conv_block, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         x = self.conv(x)
#         return x

# class up_conv(nn.Module):
#     def __init__(self, ch_in, ch_out):
#         super(up_conv, self).__init__()
#         self.up = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         x = self.up(x)
#         return x

# class UNet(nn.Module):
#     def __init__(self, img_ch=5, output_ch=8):
#         super(UNet, self).__init__()
        
#         self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
#         self.Conv2 = conv_block(ch_in=64, ch_out=128)
#         self.Conv3 = conv_block(ch_in=128, ch_out=256)
#         self.Conv4 = conv_block(ch_in=256, ch_out=512)
#         self.Conv5 = conv_block(ch_in=512, ch_out=1024)

#         self.Up5 = up_conv(ch_in=1024, ch_out=512)
#         self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
#         self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

#         self.Up4 = up_conv(ch_in=512, ch_out=256)
#         self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
#         self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
#         self.Up3 = up_conv(ch_in=256, ch_out=128)
#         self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
#         self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
#         self.Up2 = up_conv(ch_in=128, ch_out=64)
#         self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
#         self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

#         self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

#         self.Conv_binary = nn.Sequential(
#                                     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#                                     nn.BatchNorm2d(64),
#                                     nn.ReLU(inplace=True),
#                                     nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
#                                     nn.BatchNorm2d(32),
#                                     nn.ReLU(inplace=True),
#                                     nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
#                                     nn.Sigmoid()
#                                 )

#     def forward(self, x):
#         # encoding path
#         x1 = self.Conv1(x)
#         x2 = self.Maxpool(x1)
#         x2 = self.Conv2(x2)
        
#         x3 = self.Maxpool(x2)
#         x3 = self.Conv3(x3)

#         x4 = self.Maxpool(x3)
#         x4 = self.Conv4(x4)

#         x5 = self.Maxpool(x4)
#         x5 = self.Conv5(x5)

#         # decoding + concat path
#         d5 = self.Up5(x5)
#         x4 = self.Att5(g=d5, x=x4)
#         d5 = torch.cat((x4, d5), dim=1)        
#         d5 = self.Up_conv5(d5)
        
#         d4 = self.Up4(d5)
#         x3 = self.Att4(g=d4, x=x3)
#         d4 = torch.cat((x3, d4), dim=1)
#         d4 = self.Up_conv4(d4)

#         d3 = self.Up3(d4)
#         x2 = self.Att3(g=d3, x=x2)
#         d3 = torch.cat((x2, d3), dim=1)
#         d3 = self.Up_conv3(d3)

#         d2 = self.Up2(d3)
#         x1 = self.Att2(g=d2, x=x1)
#         d2 = torch.cat((x1, d2), dim=1)
#         d2 = self.Up_conv2(d2)

#         output1 = self.Conv_1x1(d2)
#         output2 = self.Conv_binary(d2)  # 新增的二值图输出

#         return output1, output2
    

# def count_parameters(model):
#     """计算模型的总参数数量，包括可训练和非可训练参数。"""
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     return {'Total parameters': total_params, 'Trainable parameters': trainable_params}

# def estimate_memory_usage(model, input_size, device):
#     """估算模型的参数内存占用和前向传播时的激活内存占用。"""
#     # 参数内存占用
#     param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
#     param_memory_MB = param_memory / (1024 ** 2)  # 转换为MB

#     # 激活内存占用
#     input_tensor = torch.randn(*input_size).to(device)
#     input_tensor.requires_grad_(True)
#     with torch.no_grad():
#         try:
#             outputs = model(input_tensor)
#             activation_memory = sum(torch.numel(x) * x.element_size() for x in outputs)
#             activation_memory_MB = activation_memory / (1024 ** 2)  # 转换为MB
#             return param_memory_MB, activation_memory_MB, True
#         except RuntimeError as e:
#             print(e)
#             return param_memory_MB, None, False

# def find_max_batch_size(model, device):
#     batch_size = 1
#     max_batch_size = 0
#     input_channels = 5
#     height, width = 256, 256
#     success = True

#     while success:
#         input_size = (batch_size, input_channels, height, width)
#         _, _, success = estimate_memory_usage(model, input_size, device)
#         if success:
#             max_batch_size = batch_size
#             batch_size *= 2
#         else:
#             if batch_size > 1:
#                 batch_size //= 2  # 减少batch size尝试更精细的探测
#             else:
#                 break

#     return max_batch_size

# def main():
#     # 设定设备
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # 实例化UNet模型
#     model = UNetTransformer().to(device)
    
#     # 找到最大的batch size
#     max_batch_size = find_max_batch_size(model, device)
    
#     print(f"Maximum batch size on current GPU: {max_batch_size}")

# def test_model_output_shape(model, input_size, device):
#     """测试模型的输出形状是否符合预期。"""
#     input_tensor = torch.randn(*input_size).to(device)
#     with torch.no_grad():
#         outputs, out2 = model(input_tensor)
#     return outputs.shape, out2.shape

# def main():
#     # 设定设备
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # 实例化UNet模型
#     model = UNet().to(device)
    
#     # 测试模型输入输出的形状
#     test_input_size = (1, 5, 256, 256)
#     output_shape, out2 = test_model_output_shape(model, test_input_size, device)
#     print(f"Output shape for input size {test_input_size}: {output_shape}")
#     print(out2)

# # 运行main函数
# if __name__ == "__main__":
#     main()