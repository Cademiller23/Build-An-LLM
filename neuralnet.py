import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt 
from torch.utils.data import TensorDataset, DataLoader


# 2 blocks -> each has 2 rows -> each row has 3 #'s
t4 = torch.tensor([
    [[1,2,3],
     [3,4,5]],
    [[5,6,7],
      [7,8,9.]],
])
print(t4)
print(t4.shape) # torch
print(t4[1,1,1]) # 1

# Building the neural network
model = nn.Sequential(
    nn.Linear(in_features=2, out_features=4), # first hidden layer
    nn.ReLU(), # activation function
    nn.Linear(4,3), # second hidden layer
    nn.ReLU(), # activation function
    nn.Linear(3,1) # linear output layer
)
print(model)
# nn.Linear(2,4) takes a 2-value input vector and maps it to 4 numbers

# To get results feed data as a tensor of shape (batch_size, 2) and call output = model(x) to get the result.


# Loss function measures the difference between targets true valuea nd the model prediction

# Optimizer is an algorithm that adjusts the weights to minimize the loss. 

"""
One step: Stochastic Gradient Descent
1. Sample some training data and run it through the network to make predictions.
2. Measure the loss between the predictions and the true values.
3. Finally, adjust the weights in a direction that makes the loss smaller."""

# Data prep -> model -> optimizer -> Training -> loss curve
# DMOTL

# Load, split, and scale data
csv_path = "red-wine.csv"
red_wine = pd.read_csv(csv_path)

# Train / Valid split (70% T/ 30% V)
df_train = red_wine.sample(frac=0.7, random_state=0)
df_valid = red_wine.drop(df_train.index)

# min-max scaling to [0, 1] (fit on train only!)
min_, max_ = df_train.min(), df_train.max()
df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)

# Separate features and target / thus train drops quality column so it can be a target
X_train = df_train.drop('quality', axis=1).values.astype('float32')
y_train = df_train['quality'].values.astype('float32').reshape(-1, 1)
X_valid = df_valid.drop('quality', axis=1).values.astype('float32')
y_valid = df_valid['quality'].values.astype('float32').reshape(-1,1)

# Wraps NumPy arrays in TensorDatasets then DataLoaders for batching/shuffling
train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
valid_ds = TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid))

train_dl = DataLoader(train_ds, batch_size=256, shuffle=True) # Shuffle only training
valid_dl = DataLoader(valid_ds, batch_size=256) 

# Define the NETWORK
model = nn.Sequential(
    nn.Linear(11, 512), # Layer 1: 11->512
    nn.ReLU(), # Non-Linear activation
    nn.Linear(512, 512), # Later 2: 512->512
    nn.ReLU(), 
    nn.Linear(512, 512), # Layer 3: 512 -> 512
    nn.ReLU(),
    nn.Linear(512, 1) # Output layer: 512->1 (linear regression)

)

# 4) Set UP Loss & Optimizer
loss_fn = nn.L1Loss() # Mean absolute Error (MAE)
optimizer = torch.optim.Adam(model.parameters()) #Adam with default settings

# 5) Training Loop
num_epochs = 10 # How many passes over the dataset
train_loss_hist = [] # Store loss for plotting

for epoch in range(num_epochs):
    # Training phase
    model.train() # Set layers to training mode (e.g., dropout)
    running = 0.0 # accumulate total loss *per sample*
    for xb, yb in train_dl: # Iterate over mini-batches
        optimizer.zero_grad() # Reset gradients to zero
        preds = model(xb) # Forward pass
        loss = loss_fn(preds, yb) # Forward pass
        loss.backward() # Gradient computation (backward pass)
        optimizer.step() # Update weights
        running += loss.item() * xb.size(0) # sum loss over batch
    
    epoch_loss = running / len(train_dl.dataset) # average MAE this epoch
    train_loss_hist.append(epoch_loss)

    # Validation phase -- (no grad, just monitoring)
    model.eval() # Evaluaton mode (disables dropout)
    with torch.no_grad(): # No gradient tracking
        val_running = 0.0
        for xv, yv in valid_dl:
            val_running += loss_fn(model(xv), yv).item() * xv.size(0)
        val_loss = val_running / len(valid_dl.dataset)
    
    print(f"Epoch {epoch+1}/{num_epochs}"
          f"- loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f}")
    
# Plotting the loss curve
pd.Series(train_loss_hist, name='loss').plot(title='Training MAE')
plt.xlabel('epoch')
plt.ylabel('MAE')
plt.show()