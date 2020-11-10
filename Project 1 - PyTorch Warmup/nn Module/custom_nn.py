#Class Based Custom nn model
import torch.nn as nn 

class TwoLayerModel(nn.Module): 
    def __init__(self, dim_in = 64, hidden_dim = 32, dim_out = 16): 
        self.linear_1 = nn.Linear(dim_in, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, dim_out)
    
    def forward(self, x):
        h = self.linear_1(x)
        h_relu = nn.ReLU()
        y_pred = self.linear_2(h_relu)
        return y_pred 

model = TwoLayerModel()

#loss_function is written as criterion ?
criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr = 10e-6)

for _ in range(num_iterations): 
    y_pred = model(x)

    loss = criterion(y_pred, y)
    #Zero gradients 
    optimizer.zero_grad()
    #Backward Pass
    loss.backward()
    #Update the parameters
    optimizer.step()


