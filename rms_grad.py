import torch
from torch import nn
# compute derivative of RMS of a vector x and compare gradient calculated by autograd against manual calculation
# Set the seed for reproducibility
torch.manual_seed(42)
n = 5
x = torch.randn(1, n, requires_grad=True)
x_squared = torch.pow(x, 2)
# 2. Calculate the mean of the squared elements
mean_x_squared = torch.mean(x_squared)
rms = torch.sqrt(mean_x_squared)
rms.backward()

# manual calc. See (rms.png)
drms_dx = torch.divide(x, n*rms)
print("drms_dx  (autograd)")
print(x.grad)

print("drms_dx  (manual)")
print(drms_dx)
