import torch
from torch import nn
import torch.nn.functional as F
# compute derivative of swiGLU of a vector x and compare gradient calculated by autograd against manual calculation
# See swiGLU section in https://arxiv.org/pdf/2410.10989 and swiglue{i}.jpg for derivation of manual derivative
# Set the seed for reproducibility
torch.manual_seed(42)
N = 10
B = 3
M = 5
x = torch.randn(M, B, requires_grad=True)
W = torch.randn(N,M, requires_grad=True)
V = torch.randn(N,M)
b = torch.randn(N, B, requires_grad=True )
c = torch.randn(N, B)
A1 = W @ x + b
A2 = V @ x + c

sigma_A1 = torch.divide(1, (1 + torch.exp(-A1)))
A3 = A1*sigma_A1
A4 = torch.mul(A3, A2)
A4.retain_grad()
L = torch.sum(A4)
L.backward()
print("dL_dx (autograd)")
print(x.grad)

# Manual calculation
# This expression is more explicit as it calculates the derivative of output of the hadamard product (A4 above) wrt
# its inputs (A3, A2) explicitly (a N*N diagonal matrix) and then takes a dot product with dL_dA4 (all ones), but
# will only work for a batch size of 1.. because of the torch.diag(A3.squeeze()), which doesn't translate to
# > 1 batch sizes
if B == 1:
    dL_dx = V.T @ (torch.diag(A3.squeeze()) @ torch.ones(N, B)) + W.T @ (torch.diag(A2.squeeze()) @ torch.ones(N, B) * (sigma_A1 * (1 + A1 - A1 * sigma_A1)))

# This should work for all batch sizes.. because for vectors of size A and V of dimensions N*B, A * V calculates the hadamard product of A and V
# without using the diagonal matrix
# @ means matrix multiplication, * means hadamard product
dL_dx = V.T @ (A3 * torch.ones(N, B)) + W.T @ (A2 * (sigma_A1 * (1 + A1 - A1 * sigma_A1)) * torch.ones(N, B))

print("dL_dx (manual)")
print(dL_dx)