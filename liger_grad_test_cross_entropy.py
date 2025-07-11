import torch
from torch import nn
logits = torch.randn(1, 10, requires_grad=True)
_softmax = nn.Softmax(dim=1)
probs = _softmax(logits)
probs.retain_grad() # to retain gradients during the backward pass
t = torch.zeros(1,10)
t[0, 0] = 1 # can set any element to 1
loss = torch.sum(torch.mul(t, torch.log(probs)))
loss.backward()

# manual calculation
dL_dlogits = t - probs
dL_dy = torch.div(t, probs)

print("dL_dlogits (autograd)")
print(logits.grad)
print("dL_dprobs (autograd)")
print(probs.grad)

print("dL_dlogits (manual)")
print(dL_dlogits)
print("dL_dprobs (manual)")
print(dL_dy)
print('done')