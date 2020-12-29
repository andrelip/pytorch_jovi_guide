# %%
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

inputs = torch.tensor([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70]], dtype=torch.float32)


# Targets (apples, oranges)
targets = torch.tensor([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119]], dtype=torch.float32)

# %%
model = nn.Linear(3, 2)
print(model.weight)
print(model.bias)
# %%
list(model.parameters())

# %%
preds = model(inputs)
preds
# %%
loss_fn = F.mse_loss

# %%
loss = loss_fn(model(inputs), targets)
print(loss)
# %%
opt = torch.optim.Adam(model.parameters(), lr=1e-1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min')


# %%
train_ds = TensorDataset(inputs, targets)
train_ds[0:3]
# %%
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
for xb, yb in train_dl:
    print(xb)
    print(yb)
    break
# %%
def fit(num_epochs, model, loss_fn, opt, train_dl):

    # Repeat for given number of epochs
    for epoch in range(num_epochs):

        # Train with batches of data
        for xb, yb in train_dl:

            # 1. Generate predictions
            pred = model(xb)

            # 2. Calculate loss
            loss = loss_fn(pred, yb)

            # 3. Compute gradients
            loss.backward()

            # 4. Update parameters using gradients
            opt.step()
            scheduler.step(loss)

            # 5. Reset the gradients to zero
            opt.zero_grad()

        # Print the progress
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch +
                                                       1, num_epochs, loss.item()))


fit(200000, model, loss_fn, opt, train_dl)

# %%

# %%
