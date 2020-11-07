#Custom Autograd for ReLU
import torch 
class CustomReLU(torch.autograd.Function): 
    #We don't use any properties of the class (self) so we can use @staticmethod
    #@staticmethod does not require access of class, just the parameters
    @staticmethod
    def forward(ctx, input): 
        #Instead of __init__ and self, we use ctx. I think of it as self in init??
        ctx.save_for_backward(input)
        #Can save anything in ctx.save_for_backward()
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output): 
        #We receive del(loss) / del(output), we need del(loss) / del(input)
        input, = ctx.saved_tensors()
        grad_input = grad_output.copy()
        grad_input[input < 0] = 0
        return grad_input




dtype = torch.float
device = torch.device("cpu")
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs.
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Create random Tensors for weights.
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for _ in range(500):
    #Instead of instantiating, we do .apply for some reason?
    relu = MyReLU.apply

    # Forward pass: compute predicted y. Wrap the operation with autograd boi
    # ReLU with the custom autograd operation.
    y_pred = relu(x.mm(w1)).mm(w2)

    loss = (y_pred - y).pow(2).sum()

    loss.backward()

    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        w1.grad.zero_()
        w2.grad.zero_()

