import torch
# Compute the derivative of RMS Norm of x multiplied by a scalar lambda and compare gradient calculated by autograd
# against manual calculation
# See swiGLU section in https://arxiv.org/pdf/2410.10989 and swiglue{i}.jpg for derivation of manual derivative
# Set the seed for reproducibility
torch.manual_seed(42)
n = 5
# 1. using autograd
x = torch.randn(1, n, requires_grad=True)
lmbda = torch.randn(1, n, requires_grad=True)
x_squared = torch.pow(x, 2)
# 2. Calculate the mean of the squared elements
mean_x_squared = x_squared.mean()
rms = torch.sqrt(mean_x_squared)
x_rms_norm = torch.divide(x, rms) # this is our x_hat
y = torch.mul(lmbda, x_rms_norm)
L = torch.sum(y)
L.backward()

# manual calculation.. see rmsnorm{i}.png images
dL_dx = (1/rms) * (lmbda - 1/n*torch.matmul(lmbda, torch.matmul(torch.transpose(x_rms_norm, 0, 1), x_rms_norm)))
print("dL_dx (autograd)")
print(x.grad)

print("dL_dx (manual)")
print(dL_dx)