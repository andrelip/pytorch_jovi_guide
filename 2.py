# %%
import numpy as np
import torch

# Input (temp, rainfall, humidity)

inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70]], dtype='float32')


# Targets (apples, oranges)
targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119]], dtype='float32')

# to pytorch tensor
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)
# %%
def model(x):
    return x @ w.t() + b

# %%
def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()
# %%
preds = model(inputs)
loss = mse(preds, targets)
loss.backward()
print(loss)

# %%
with torch.no_grad():
    w -= w.grad * 1e-5
    b -= b.grad * 1e-5
w.grad.zero_()
b.grad.zero_()
# %%

w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)
num_epochs = 250_000

for epoch in range(num_epochs):
    preds = model(inputs)
    loss = mse(preds, targets)
    loss.backward()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    with torch.no_grad():
        w -= w.grad * 1e-7
        b -= b.grad * 1e-7
        w.grad.zero_()
        b.grad.zero_()

# %%
calculate(inputs, targets, 100)
# %%
