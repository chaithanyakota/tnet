# tnet

A simple neural network from scratch in Python using only NumPy. The neural network is designed to predict the gender of a person based on their height and weight data. 

The [sigmoid activation function](https://en.wikipedia.org/wiki/Sigmoid_function) is used for the neurons in the hidden layer. It squashes the output of each neuron between 0 and 1, introducing non-linearity to the model.

$`σ(z) = \frac{1}{1 + e^{-z}}`$

[Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) is used to train the neural network.

The [Mean Squared Error](https://en.wikipedia.org/wiki/Mean_squared_error) is used as a metric to determine the model's performance. It quantifies the average squared difference between the predicted values and the true labels. 
Lower MSE values indicate a better fit of the model to the data.

$`MSE = \frac{1}{n} \displaystyle\sum_{i=1}^{n} {(y_{true} - y_{pred})}^2`$

