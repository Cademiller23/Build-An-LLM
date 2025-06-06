import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from matplotlib import pyplot as plt
# Number
t1 = torch.tensor(4.)
print(t1)
# Vector
t2 = torch.tensor([1., 2, 3, 4])
print(t2)
# Matrice
t3 = torch.tensor([[1., 2],
                   [3, 4],
                   [5, 6]])
print(t3)
# 3D Tensor
t4 = torch.tensor([
    [[1,2,3],
     [3,4,5]],
    [[5,6,7],
      [7,8,9.]],
])
print(t4)
# .shape property
print(t4.shape) # torch

# pytorch tensors have a majpr advantage: GPU optimization and automatic derivative calculation.
u = torch.tensor(3., requires_grad=True)
v = torch.tensor(4., requires_grad=True)
print("Value of u:", u, "Value of v:", v)

f = u**3 + v**2
print("Value of f:", f)

f.backward()
print("\ndf/du ", u.grad)
print("df/dv :", v.grad)

# Gradient descent in Pytorch

def function(a,b,x):
    return a*x**3 + b*x**2

x = torch.linspace(-2.1, 2.1, 20)[:,None]
y = function(10, 3, x)
plt.scatter(x,y)
plt.title('Plot for $a=10$, $b=3$')
plt.show()
# Mean absolute error, which is the distance from each data point to teh curve.
def mae(truth, preds): 
    return (torch.abs(preds - truth)).mean()

y2 = function(1,1,x)
y3 = function(2,1,x)
plt.scatter(x,y)
plt.plot(x,y2,c='orange')
plt.plot(x,y3,c='green')

plt.legend(['Data',
            'a = 1, b = 1, MAE = {:.3f}'.format(mae(y, y2).item()),
            'a = 4, b = 1, MAE = {:.3f}'.format(mae(y,y3).item())
            ])
plt.show()

def calc_mae(args):
    y_new = function(*args, x)
    return mae(y, y_new)
print(calc_mae([5,1]))
print()

ab = torch.tensor([1.1, 1], requires_grad=True)
print(ab)
print()
# Calculate the loss
loss = calc_mae(ab)
print("Loss:", loss)
# Calculate the gradient
loss.backward()
print("Gradient:", ab.grad) 
print("Gradient a:", ab.grad[0])
print("Gradient b:", ab.grad[1])
print()
# Since we have a negative gradient means that we need to increase the value of the parameters. 

with torch.no_grad():
    ab -= ab.grad * 0.02 # Learning rate
    loss = calc_mae(ab)
print(f'loss={loss:.2f}')

for i in range(15):
    loss = calc_mae(ab) # calculate the loss
    loss.backward() # calculate the gradient
    with torch.no_grad(): # update the parameters and gradient based on the learning rate and the gradient
        ab -= ab.grad * 0.02
    print(f'step={i}, loss={loss:.2f}')

print(ab)

a, b = ab.detach()
y2 = function(1,1,x)
y3 = function(2,1,x)
y4 = function(a,b,x)
plt.scatter(x,y)
plt.plot(x,y2,c='orange')
plt.plot(x,y3,c='green')
plt.plot(x,y4,c='red')

plt.legend(['Data',
            'a = 1, b = 1, MAE = {:.3f}'.format(mae(y,y2).item()),
            'a = 2, b = 1, MAE = {:.3f}'.format(mae(y,y3).item()),
            'a = {:.1f}, b = {:.1f}, MAE = {:.3f}'.format(a,b, mae(y,y3).item()),
            ])
plt.show()