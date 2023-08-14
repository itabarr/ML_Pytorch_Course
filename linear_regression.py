import torch 

# Example for doing linear regression in pytorch
sample_size = 10
x_dim = 5
y_dim = 2
learning_rate = 0.01

#generate random data sample X and W , and B to be learned by the model 
X = torch.rand(sample_size, x_dim)
W = torch.rand(y_dim, x_dim)
B = torch.rand(y_dim)

def model(x, w, b):
    return x @ w.T + b

def mse(y, y_hat):
    return torch.mean((y - y_hat) ** 2)

# generatr Y by applying model(X, W, B) with noise
Y = model(X, W, B)
NOISE = torch.normal(0, 0.001, (sample_size, y_dim))
Y = Y + NOISE


#generate ransom guess
w = torch.randn(y_dim, x_dim , requires_grad=True)
b = torch.rand(y_dim ,  requires_grad=True)

for i in range(30000):
    pred = model(X, w, b)
    loss = mse(Y, pred)
    loss.backward()

    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        w.grad.zero_()
        b.grad.zero_()
    
    if i % 100 == 0:
        print(f"Iteration: {i}, Loss: {loss.item()}")

        # calculate the diffrence between model prediction and ground truth
        dw  = ((W - w)**2).sum()
        db = ((B - b)**2).sum() 

        print(f"dw: {dw.item()}, db: {db.item()}")
         











