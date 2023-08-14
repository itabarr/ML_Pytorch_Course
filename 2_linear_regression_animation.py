import torch 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Example for doing linear regression in pytorch
sample_size = 100
x_dim = 1
y_dim = 1
learning_rate = 0.02

#generate random data sample X and W , and B to be learned by the model 
X = torch.linspace(0, 10, sample_size)
W = torch.tensor([[4]])
B = torch.tensor([60])


def model(x, w, b):
    if x.ndim == 1:
        return (w * x + b).T

    return x @ w.T + b

def mse(y, y_hat):
    return torch.mean((y - y_hat) ** 2)

def fit(x , w , b):
    pred = model(X, w, b)
    loss = mse(Y, pred)
    loss.backward()

    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        w.grad.zero_()
        b.grad.zero_()
    
    return w , b 


# generatr Y by applying model(X, W, B) with noise
Y_NO_NOISE = model(X, W, B)
NOISE = torch.normal(0, 2, (sample_size, y_dim))
Y = Y_NO_NOISE + NOISE


#generate ransom guess
w = torch.randn(y_dim, x_dim , requires_grad=True)
b = torch.rand(y_dim ,  requires_grad=True)

#matplotlib notebook
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(X, Y, label='Data')
ax.plot(X, Y_NO_NOISE, 'r', label='Model without noise')
line, = ax.plot(X, model(X, w, b).detach().numpy(), 'g', label='Predicted')

w = torch.randn(y_dim, x_dim, requires_grad=True) 
b = torch.rand(y_dim, requires_grad=True) 
pred = model(X, w, b)

ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')


def update(frame):
    global w, b, line  # Use global to modify these values
    
    # Update model parameters using fit function
    w, b = fit(X, w, b)
    
    # Get the updated prediction
    pred = model(X, w, b)
    
    # Update the line data for animation
    line.set_ydata(pred.detach().numpy())
    
    return line,

ani = FuncAnimation(fig, update, frames=range(100), blit=True)

plt.show()
         











