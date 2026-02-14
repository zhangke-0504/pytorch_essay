import torch

input = [
    3,4,5,6,7,
    2,4,6,8,2,
    1,6,7,8,4,
    9,7,4,6,2,
    3,7,5,4,1
]

input = torch.Tensor(input).view(1, 1, 5, 5)
# 此处第一个1是输出通道数，第二个1是输入通道数
conv_layer = torch.nn.Conv2d(
    1, 1, 
    kernel_size=3,
    stride=2, 
    padding=1, 
    bias=False,
)

# 123456789是卷积核的数值
# 第一个1是输出通道数， 第二个1是输入通道数，然后是卷积核的wh
kernel = torch.Tensor([1,2,3,4,5,6,7,8,9]).view(1, 1, 3, 3)
# 卷积核权重人为初始化
conv_layer.weight.data = kernel.data

output = conv_layer(input)
print(output)