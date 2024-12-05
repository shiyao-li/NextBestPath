import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=16, emb_size=512, img_size=128):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, E, H/P, W/P]
        x = x.flatten(2)  # [B, E, N]
        x = x.transpose(1, 2)  # [B, N, E]
        return x

class VisionTransformer(nn.Module):
    def __init__(self, in_channels=1, patch_size=16, emb_size=512, img_size=128, num_heads=8, depth=3):
        super().__init__()
        self.patch_emb = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        transformer_layers = nn.TransformerEncoderLayer(d_model=emb_size, nhead=num_heads)
        self.transformer_enc = nn.TransformerEncoder(transformer_layers, num_layers=depth)

    def forward(self, x):
        x = self.patch_emb(x)
        x = self.transformer_enc(x)
        return x

class GainPredictionModel(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(GainPredictionModel, self).__init__()
        self.feature_extractor = VisionTransformer()
        self.layer_norm1 = nn.LayerNorm(512)  # 层归一化1
        self.layer_norm2 = nn.LayerNorm(512)  # 层归一化2
        self.diff_feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer_norm_diff = nn.LayerNorm(64 * 32 * 32)  # 根据实际输出大小调整
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(512 * 2 + 64 * 32 * 32, 256) 
        self.fc2 = nn.Linear(256, 1)

    def forward(self, grid_a, grid_b):
        feat_a = self.feature_extractor(grid_a)
        feat_b = self.feature_extractor(grid_b)

        # 应用层归一化
        feat_a = self.layer_norm1(feat_a)
        feat_b = self.layer_norm2(feat_b)

        feat_a = torch.mean(feat_a, dim=1)
        feat_b = torch.mean(feat_b, dim=1)

        # 计算差异特征
        diff = torch.abs(grid_a - grid_b)
        diff_feat = self.diff_feature_extractor(diff)
        diff_feat = diff_feat.view(diff_feat.size(0), -1)
        diff_feat = self.layer_norm_diff(diff_feat)

        combined_features = torch.cat((feat_a, feat_b), dim=1)
        combined_features = self.dropout(combined_features)

        x = F.relu(self.fc1(combined_features))
        gain = self.fc2(x)
        return gain


# class PatchEmbedding(nn.Module):
#     def __init__(self, in_channels=1, patch_size=16, emb_size=768, img_size=128):
#         super().__init__()
#         self.patch_size = patch_size
#         self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

#     def forward(self, x):
#         x = self.proj(x)  # [B, E, H/P, W/P]
#         x = x.flatten(2)  # [B, E, N]
#         x = x.transpose(1, 2)  # [B, N, E]
#         return x

# class VisionTransformer(nn.Module):
#     def __init__(self, in_channels=1, patch_size=16, emb_size=768, img_size=128, num_heads=8, depth=3):
#         super().__init__()
#         self.patch_emb = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
#         transformer_layers = nn.TransformerEncoderLayer(d_model=emb_size, nhead=num_heads)
#         self.transformer_enc = nn.TransformerEncoder(transformer_layers, num_layers=depth)

#     def forward(self, x):
#         x = self.patch_emb(x)
#         x = self.transformer_enc(x)
#         return x

# class GainPredictionModel(nn.Module):
#     def __init__(self, dropout_rate=0.1):
#         super(GainPredictionModel, self).__init__()
#         self.feature_extractor = VisionTransformer()
#         self.layer_norm1 = nn.LayerNorm(768)  # 层归一化1
#         self.layer_norm2 = nn.LayerNorm(768)  # 层归一化2
#         self.layer_norm_after_combine = nn.LayerNorm(768 * 2 + 1)  # 组合特征后的层归一化
#         self.dropout = nn.Dropout(dropout_rate)
#         self.fc1 = nn.Linear(768 * 2 + 1, 512) 
#         self.fc2 = nn.Linear(512, 1)

#     def forward(self, grid_a, grid_b, distance):
#         feat_a = self.feature_extractor(grid_a)
#         feat_b = self.feature_extractor(grid_b)

#         # 应用层归一化
#         feat_a = self.layer_norm1(feat_a)
#         feat_b = self.layer_norm2(feat_b)

#         feat_a = torch.mean(feat_a, dim=1)
#         feat_b = torch.mean(feat_b, dim=1)

#         feat_a_flat = feat_a.view(feat_a.size(0), -1)
#         feat_b_flat = feat_b.view(feat_b.size(0), -1)

#         if distance.dim() == 1:
#             distance = distance.unsqueeze(1)

#         combined_features = torch.cat((feat_a_flat, feat_b_flat, distance), dim=1)
#         combined_features = self.layer_norm_after_combine(combined_features)

#         x = F.relu(self.fc1(self.dropout(combined_features)))
#         gain = self.fc2(x)
#         return gain

