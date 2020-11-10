import torch 
#nn package has a set of modules. A Module receives a Input and computes output
import torch.nn as nn

batch_size = 4
dim_in, hidden_dim, dim_out = 64, 32, 16

x = torch.randn(batch_size, dim_in)
y = torch.randn(batch_size, dim_out)


#nn.Sequential - Similar to Keras Sequential 
model = nn.Sequential(
    nn.Linear(dim_in, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, dim_out)
)

#Default reduction is 'mean' (it's MEAN squared loss). For loss, we gotta take total
loss_function = nn.MSELoss(reduction='sum')
num_iterations = 100
learning_rate = 10e-6

for _ in range(num_iterations): 
    y_pred = model(x)
    loss = loss_function(y_pred, y)

    model.zero_grad()

    #Backward Pass : get grads of loss wrt all learnable params. 
    loss.backward()
    
    #Update weights with gradient descent
    with torch.no_grad(): 
        for param in model.parameters(): 
            param -= learning_rate * param.grad


#Using Optim
#First argument is the tensors it should update
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
for _ in range(num_iterations): 
    y_pred = model(x)
    loss = loss_function(y_pred, y)

    #Zero out all the gradients for the variables it will update
    #On calling loss.backward(), gradients are not overwritten, but are accumulated in buffers. 
    optimizer.zero_grad()

    #Backward Pass: Compute del(loss) wrt del(parameters)
    loss.backward()

    #Calling step function updates the parameters. I imagine tensors as nodes in the computational graph
    #We update the node.val wherener we do stuff. It's inplace?
    optimizer.step()