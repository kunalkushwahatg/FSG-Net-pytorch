import torch
import torch.nn as nn
import torch.fft


# --- Frequency Domain Encoder Block ---
class FrequencyEncoder(nn.Module):
    def __init__(self, in_channels):
        super(FrequencyEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # FFT to get real and imaginary parts
        fft = torch.fft.fft2(x)
        real = fft.real
        imag = fft.imag
        freq_feat = torch.cat([real, imag], dim=1)  # Shape: (B, 2*C, H, W)
        return self.conv(freq_feat)


# --- Attention Gate (Squeeze-and-Excitation Style) ---
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(F_int, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(g1 + x1)
        return x * psi


# --- Double Convolution Block ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# --- Frequency-Guided U-Net ---
class FrequencyAttentionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(FrequencyAttentionUNet, self).__init__()
        self.freq_encoder = FrequencyEncoder(in_channels)

        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        prev_channels = in_channels
        for feature in features:
            self.encoder_blocks.append(DoubleConv(prev_channels, feature))
            prev_channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        self.upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()

        for feature in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.attention_blocks.append(AttentionGate(F_g=feature, F_l=feature, F_int=feature // 2))
            self.decoder_blocks.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        freq_feat = self.freq_encoder(x)
        x = x + freq_feat  # frequency-guided input fusion

        skip_connections = []

        for encoder in self.encoder_blocks:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            skip = skip_connections[i]
            attn = self.attention_blocks[i](x, skip)
            x = torch.cat((attn, x), dim=1)
            x = self.decoder_blocks[i](x)

        return self.final_conv(x)
