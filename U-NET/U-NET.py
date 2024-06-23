import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

# Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

# Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

# Decoder
        self.dec4 = self.conv_block(1024 + 512, 512)
        self.dec3 = self.conv_block(512 + 256, 256)
        self.dec2 = self.conv_block(256 + 128, 128)
        self.dec1 = self.conv_block(128 + 64, 64)

# Final Convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

# Max Pooling
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

# Up Sampling
        self.up_sample = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_sample2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_sample3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_sample4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
# Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.max_pool(enc1))
        enc3 = self.enc3(self.max_pool(enc2))
        enc4 = self.enc4(self.max_pool(enc3))

# Bottleneck
        bottleneck = self.bottleneck(self.max_pool(enc4))

# Decoder
        dec4 = self.up_sample(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.up_sample2(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.up_sample3(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.up_sample4(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

# Final Convolution
        return self.final_conv(dec1)

model = UNet(in_channels=3, out_channels=1).cuda()
print(model)