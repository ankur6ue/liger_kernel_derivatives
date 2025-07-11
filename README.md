This repo contains mathematical derivations for some of the gradients in the paper ["Liger Kernel: Efficient Triton Kernels for LLM Training"](https://arxiv.org/pdf/2410.10989) and code for comparing the gradient calculated using the derived formulas against gradient calculated by pytorch autograd.

Currently the calculations include:
- RMS (Root Mean Square)
- RMSNorm
- SwiGLU

RMSNorm and SwiGLU output a vector, while Pytorch .backward() operates on a scalar. To convert the vector output to a scalar, I'm using the sum operation. 