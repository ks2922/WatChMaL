import torch
from watchmal.model.pointnet2 import PointNet2

model = PointNet2(num_input_channels=5, num_output_channels=7)
x = torch.randn(4, 5, 256)
out = model(x)
print('output shape:', out.shape)  # should be [4, 7]
print('forward pass successful')
