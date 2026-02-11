import numpy as np
import matplotlib.pyplot as plt

# STEP 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# STEP 2
X = np.array([[0],
              [1]])

Y = np.array([[0],
              [1]])

# STEP 3
weight_input_hidden = np.random.rand(1, 1)
bias_hidden = np.random.rand(1)

weight_hidden_output = np.random.rand(1, 1)
bias_output = np.random.rand(1)

# STEP 4
hidden_input = np.dot(X, weight_input_hidden) + bias_hidden
hidden_output = sigmoid(hidden_input)

final_input = np.dot(hidden_output, weight_hidden_output) + bias_output
final_output = sigmoid(final_input)

loss = np.mean((Y - final_output) ** 2)

# STEP 5
output_error = Y - final_output
output_delta = output_error * sigmoid_derivative(final_output)

hidden_error = np.dot(output_delta, weight_hidden_output.T)
hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

learning_rate = 0.5

weight_hidden_output = weight_hidden_output + np.dot(hidden_output.T, output_delta) * learning_rate
bias_output = bias_output + np.sum(output_delta) * learning_rate

weight_input_hidden = weight_input_hidden + np.dot(X.T, hidden_delta) * learning_rate
bias_hidden = bias_hidden + np.sum(hidden_delta) * learning_rate

# STEP 6
epochs = 1000
loss_values = []

for i in range(epochs):

    hidden_input = np.dot(X, weight_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weight_hidden_output) + bias_output
    final_output = sigmoid(final_input)

    loss = np.mean((Y - final_output) ** 2)
    loss_values.append(loss)

    output_error = Y - final_output
    output_delta = output_error * sigmoid_derivative(final_output)

    hidden_error = np.dot(output_delta, weight_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

    weight_hidden_output = weight_hidden_output + np.dot(hidden_output.T, output_delta) * learning_rate
    bias_output = bias_output + np.sum(output_delta) * learning_rate

    weight_input_hidden = weight_input_hidden + np.dot(X.T, hidden_delta) * learning_rate
    bias_hidden = bias_hidden + np.sum(hidden_delta) * learning_rate

# STEP 7
plt.plot(loss_values)
plt.title("Loss During Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

hidden_input = np.dot(X, weight_input_hidden) + bias_hidden
hidden_output = sigmoid(hidden_input)

final_input = np.dot(hidden_output, weight_hidden_output) + bias_output
final_output = sigmoid(final_input)

print("Final Predictions:")
print(final_output)
