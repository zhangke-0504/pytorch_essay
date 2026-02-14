import torch
# 输入的图像通道数，以及输出的结果的通道数
in_channels, out_channels = 5, 10
# 图像大小
width, height = 100, 100
# 卷积核hw
kernel_size = 3
batch_size = 1

input = torch.randn(
    batch_size,
    in_channels,
    width,
    height
)

conv_layer = torch.nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=kernel_size
)

output = conv_layer(input)

print(input.shape)
print(output.shape)
print(conv_layer.weight.shape)