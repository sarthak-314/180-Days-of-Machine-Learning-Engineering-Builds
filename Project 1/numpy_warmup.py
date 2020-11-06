#Neural Network in Numpy
import numpy as np

batch_size = 32
dim_in, hidden_dim, dim_out = 64, 32, 16

#Random input and output data - we try to make the network predict x to y mapping
x = np.random.randn(batch_size, dim_in)
y = np.random.randn(batch_size, dim_out)

#Randomly intitalize weights - we try to learn the weights through backpropogation
weights_1 = np.random.randn(dim_in, hidden_dim)
weights_2 = np.random.randn(hidden_dim, dim_out)

learning_rate = 10e-3
num_iterations = 10
for _ in range(num_iterations): 
    #Get the predicted y (Forward)
    hidden = x.dot(weights_1)
    #Relu Function np.max() => 0 is broadcasted, take element wise maximum
    h_relu = np.max(hidden, 0)
    y_pred = h_relu.dot(weights_2)
    #Element wise difference => square => sum all
    loss = np.square(y_pred - y).sum()

    #x => weights_1 => hidden => h_relu => weights_2 => y_pred => y
    #Points from y to y_pred
    grad_y_pred = 2 * (y_pred - y)
    #y_pred = h_relu.weights_2
    #del(y_pred)(with weights_2) = h_relu . del(weights_2)
    grad_weights_2 = np.dot(h_relu.T, grad_y_pred)
    grad_h_relu = np.dot(weights_2.T, grad_y_pred)
    grad_h = grad_h_relu.copy()
    #Relu only for positive numbers. Ignore gradient for negative nums
    grad_h[h < 0] = 0
    grad_weights_1 = np.dot(x.T, grad_h)
    
    #Update weights
    weights_1 -= learning_rate * grad_weights_1
    weights_2 -= learning_rate * grad_weights_2


    #Update weights_2
    weights_2 += gradient * learning_rate
    weights_1 += gradient 
    #Error = (y - y_pred) ^ 2