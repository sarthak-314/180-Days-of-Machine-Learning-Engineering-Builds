#Neural Network in PyTorch Using Autograd

import torch 
batch_size = 4
dim_in, hidden_dim, dim_out = 64, 32, 16
device = torch.device('cpu')
#Make tensor of specific data type
x = torch.randn(batch_size, dim_in, device=device, dtype=torch.float)
y = torch.randn(batch_size, dim_out, device=device, dtype=torch.dtype)

#With autograd, the forward pass makes a computation graph with tensors as nodes 
# and edges as mappings from input to output functions
#If x.requires_grad = True, find grad by x.grad

#Requires grad = True, because we wanna compute the gradient in backward pass backpropogation
weights_1 = torch.randn(dim_in, hidden_dim, device=device, dtype=torch.float, requires_grad=True)
weights_2 = torch.randn(hidden_dim, dim_out, device=device, dtype=torch.float, requires_grad=True)

learning_rate = 1e-6
num_iterations = 500

for _ in range(num_iterations): 
    h = x.mm(weights_1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(weights_2)

    loss = (y_pred - y).pow(2).sum()
    #Autograd for the backward pass. 
    #Calculates gradient of loss wrt all the tensors which have .requires_grad = True
    #Get the gradient of loss wrt tensor in .grad
    loss.backward()

    #Wrap in torch.no_grad() because to not track the operations
    with torch.no_grad(): 
        weights_1 -= learning_rate * weights_1.grad
        weights_2 -= learning_rate * weights_2.grad

        #Manually set the gradient to zero for next iteration
        weights_1.grad.zero_()
        weights_2.grad.zero_()

