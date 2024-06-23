import torch
import torch.nn as nn

class UNetClassification(nn.Module):
	def __init__(self, in_channels=3, num_classes=2):
		super(UNetClassification, self).__init__()

    # Encoder
    self.enc1 = self.conv_block(in_channels, 64)
    self.enc2 = self.conv_block(64, 128)
    self.enc3 = self.conv_block(128, 256)
    self.enc4 = self.conv_block(256, 512)

    # Bottleneck
    self.bottleneck = self.conv_block(512, 1024)

    # Fully Connected Layer
    self.fc = nn.Sequential(
        nn.Linear(1024 * 8 * 8, 256),  # assuming input image size is 128x128
        nn.ReLU(inplace=True),
        nn.Linear(256, num_classes)
    )

    # Max Pooling
    self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

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

    # Flatten and Fully Connected Layer
    bottleneck_flat = bottleneck.view(bottleneck.size(0), -1)
    return self.fc(bottleneck_flat)

model_cls = UNetClassification(in_channels=3, num_classes=2).cuda()
print(model_cls)