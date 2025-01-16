import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    """Compute softmax probabilities for the positive class."""
    # x is a series of probabilities for the positive class
    #assume the negative class has a constant score of 0
    # return the softmax probabilities for the positive class only
    exp = np.exp(x)
    softmaxes = exp / (exp+1)
    return softmaxes

def tanh(x):
    return np.tanh(x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def step_function(x, tau=0):
    return np.where(x > tau, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivativeofsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Generate input values between -1 and 1
x = np.linspace(-0.1, 1.1, 1000)

# Evaluate activation functions
relu_output = relu(x)
tanh_output = tanh(x)
leaky_relu_output = leaky_relu(x)
step_output = step_function(x, tau=0)  # You can adjust tau as needed
sigmoid_output = sigmoid(x)

# For Softmax, compute the probability of the positive class
# Assume the negative class has a constant score of 0
scores = np.vstack((x, np.zeros_like(x)))  # Positive class scores and negative class scores
softmax_probs = np.array([softmax(score) for score in scores.T])  # Transpose to iterate over columns
softmax_positive = softmax_probs[:, 0]  # Extract positive class probabilities

# Plot all activation functions in one plot
plt.figure(figsize=(6, 4))
plt.plot(x, relu_output, label='ReLU', color='red',linewidth=2)
plt.plot(x, step_output, label='Step Function (τ=0)', color='green', linestyle='--',linewidth=2)
plt.plot(x, softmax_positive, label='Softmax (Negative Class = 0)', color='blue',linewidth=2)
# plt.plot(x, sigmoid_output, label='Sigmoid', color='purple',linewidth=2)
plt.xlabel('Input', fontsize=14)
plt.ylabel('Output', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('activation_functions.png')
plt.show()
plt.close()
