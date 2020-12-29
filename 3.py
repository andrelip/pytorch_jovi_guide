# %%
import torch
import torchvision
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F


%matplotlib inline


# %%
dataset = MNIST(root='data/', download=True)

# %%
len(dataset)

# %%
test_dataset = MNIST(root='data/', train=False)
len(test_dataset)

# %%
dataset[0]

# %%
image, label = dataset[0]
plt.imshow(image, cmap='gray')
print('Label:', label)

# %%
dataset = MNIST(root='data/',
                train=True,
                transform=transforms.ToTensor())

img_tensor, label = dataset[0]
print(img_tensor.shape, label)

# %5 
img_tensor
# %%
print(img_tensor[0, 10:15, 10:15])
# %%
print(torch.max(img_tensor), torch.min(img_tensor))

# %%
plt.imshow(img_tensor[0, 10:15, 10:15], cmap='gray')

# %%
train_ds, val_ds = random_split(dataset, [50000, 10000])
len(train_ds), len(val_ds)

# %%
batch_size = 128
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)
# %%
input_size = 28*28
num_classes = 10
# %%
model = nn.Linear(input_size, num_classes)
print(model.weight.shape)
model.weight

# %%
print(model.bias.shape)
model.bias

# %%

# image could be reshaped
img_tensor, label = dataset[0]
print(img_tensor.shape)
img_tensor.reshape(784)

class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out


model = MnistModel()


# %%
print(model.linear.weight.shape, model.linear.bias.shape)
list(model.parameters())

# %%
for images, labels in train_loader:
    print(images.shape)
    outputs = model(images)
    break

print('outputs.shape : ', outputs.shape)
print('Sample outputs :\n', outputs[:2].data)
# %%
# Apply softmax for each output row
probs = F.softmax(outputs, dim=1)

# Look at sample probabilities
print("Sample probabilities:\n", probs[:2].data)

# Add up the probabilities of an output row
print("Sum: ", torch.sum(probs[0]).item())

# %%
max_probs, preds = torch.max(probs, dim=1)
print(preds)
print(max_probs)

# %%
labels

# %%
torch.sum(preds == labels)

# %%


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# %%
accuracy(outputs, labels)

# %% 
loss_fn = F.cross_entropy
# %%
loss = loss_fn(outputs, labels)
print(loss)



# %%
labels
# %%
