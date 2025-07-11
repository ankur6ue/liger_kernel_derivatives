import torch
from torch import nn
# compute derivative of swiGLU of a vector x and compare gradient calculated by autograd against manual calculation
# See swiGLU section in https://arxiv.org/pdf/2410.10989 and swiglue{i}.jpg for derivation of manual derivative
n = 10
x = torch.randn(5, 1, requires_grad=True)
W = torch.randn(10,5)
V = torch.randn(10,5)
b = torch.randn(10, 1)
c = torch.randn(10, 1)
A1 = W @ x + b
A2 = V @ x + c
y = torch.mul(A1, A2)
L = torch.sum(y)

sigma_A1 = torch.divide(1, (1 + torch.exp(-A1)))
A3 = A1*sigma_A1
A4 = torch.mul(A3, A2)
L = torch.sum(A4)
L.backward()
print("dL_dx (autograd)")
print(x.grad)

# manual calculation
dL_dx = torch.ones(1,10) @ (torch.diag(A3.squeeze())  @ V + torch.diag(A2.squeeze()) @  torch.diag(torch.mul(sigma_A1, (1 + A1 - torch.mul(A1, sigma_A1))).squeeze()) @ W)
print("dL_dx (manual)")
print(dL_dx)