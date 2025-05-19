"""
Binary Classification with PyTorch Review

Accuracy = number_correct / number_total
Cross-entropy is a sort of measure for the distance from one probability distribution to another.
Cross-entropy helps measure classification loss

Sigmoid activation 
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

csv_path = 'ionosphere.csv'
ionosphere = pd.read_csv(csv_path, header=None)

"""
1 - Basic Pre-Processing
    a) Feature / label sepraration
    b) Train-Validation split (70/30)
    c) min-max scaling using *train* stats only
"""

# A) split into X (features) and y (labels)
X = ionosphere.iloc[:, :-1].astype("float32").values # all columns exxcept last
y = (ionosphere.iloc[:, -1] == "g").astype("float32").values # last col: 'g'->1, 'b'->0

# B) Create reproducible 70/30 split
rng = np.random.default_rng(seed = 0) # RNG with fixed seed
index = rng.permutation(len(X)) # Shuffled indices
split = int(len(X) * 0.7) # 70% of data
train_idx, valid_idx = index[:split], index[split:] # indices for each set

X_train, X_valid = X[train_idx], X[valid_idx] # feature matrices
y_train, y_valid = y[train_idx, None], y[valid_idx, None] # labels -> shape (N, 1)


# C) Min-Max Scale *by train stats only*
xmin, xmax = X_train.min(0), X_train.max(0) # column-wise min/max
X_train = (X_train - xmin) / (xmax - xmin + 1e-9) # Scale train to [0,1]
X_valid = (X_valid - xmin) / (xmax - xmin + 1e-9) # Scale valid to [0,1]

train_dl = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy((y_train)),
                                    batch_size=512, shuffle=True))
valid_dl = DataLoader(TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid)),
                      batch_size=512)


# Model
in_dim = X_train.shape[1] # 33 features in this file
model = nn.Sequential(
    nn.Linear(in_dim, 4), nn.ReLU(),
    nn.Linear(4,4), nn.ReLU(),
    nn.Linear(4,1) # Logits; sigmoid handled by loss/metric
)

# Loss, metric, optimiser (Keras .compile equivalent)
loss_fn = nn.BCEWithLogitsLoss() # Binary Cross-Entropy with Logits
optimizer = torch.optim.Adam(model.paramters())

def accuracy(outputs, targets):
    preds = (torch.sigmoid(outputs) >= 0.5).float() # Sigmoid + threshold (0.5)
    return (preds == targets).float().mean().item() # Accuracy = % correct predictions

# Training with early stopping
patience, min_delta = 10, 1e-3
best_val, wait, best_state = np.inf, 0, None 
history = {"loss": [], "val_loss": [], 'acc': [], 'val_acc': []} # Store metrics

for epoch in range(1000): # Max epochs
    # Train
    model.train() # Set layers to training mode (e.g., dropout)
    running_loss, running_acc = 0.0, 0.0 # Accumulate loss/acc
    for xb, yb in train_dl: # Iterate over mini-batches what is xb, yb?
        # xb = torch.from_numpy(X_train[train_idx])
        # yb = torch.from_numpy(y_train[train_idx])
        optimizer.zero_grad() # Reset gradients to zero
        out = model(xb) # Forward pass
        loss = loss_fn(out, yb) # Forward pass
        loss.backward() # Gradient computation (backward pass)
        optimizer.step() # Update weights
        running_loss += loss.item() * xb.size(0) # sum loss over batch
        running_acc += accuracy(out, yb) * xb.size(0) # sum acc over batch

    train_loss = running_loss / len(train_dl.dataset) # average MAE this epoch
    train_acc = running_acc / len(train_dl.dataset) # average acc this epoch

    # Validate
    model.eval() # Set layers to evaluation mode (e.g., dropout)
    with torch.no_grad():
        v_loss, v_acc = 0.0, 0.0 # Accumlate loss/acc
        for xb, yb in valid_dl:
            out = model(xb) # Forward pass
            v_loss += loss_fn(out, yb).item() * xb.size(0) # Sum loss over batch
            v_acc += accuracy(out, yb) * xb.size(0) # Sum acc over batch
    val_loss = v_loss / len(valid_dl.dataset) # average MAE this epoch
    val_acc = v_acc / len(valid_dl.dataset) # Average acc this epoch

    # Record
    history['loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    # Early stopping check
    if val_loss + min_delta < best_val:
        best_val, wait, best_state = val_loss, 0, model.state_dict() # save best state
    else:
        wait += 1
    if wait >= patience:
        print(f"Early stop at epoch {epoch + 1}")
        break
# Restore best weights
model.load_state_dict(best_state)

# Plot learning curves
pd.DataFrame(history).iloc[5:][['loss', 'val_loss']].plot()
pd.DataFrame(history).iloc[5:][['acc', 'val_acc']].plot()

