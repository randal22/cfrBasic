import torch
import torch.nn as nn
import torch.optim as optim
from loadEncode import encoded_arrays,sets_of_values,string_values
import numpy as np
# 1. Define the XOR dataset
# XOR inputs and corresponding labels
inputs = encoded_arrays
labels = sets_of_values

# 2. Define the neural network model
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.hidden1 = nn.Linear(39, 64)  # Hidden layer with 64 neurons
        self.hidden2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, 3)  # Output layer receiving 64

    def forward(self, x):
        x=x.view(-1,3*13)
        x = torch.relu(self.hidden1(x))  # Activation function for hidden layer
        x = torch.relu(self.hidden2(x))  # Activation function for hidden layer
        x=torch.nn.functional.softmax(self.output(x), dim=1)
        return x

# Instantiate the model
model = XORModel()

# 3. Define the loss function and the optimizer
#criterion = nn.cro
optimizer = optim.AdamW(model.parameters())  # Stochastic Gradient Descent #default learning rate

# 4. Train the model
num_epochs = 10000  # Number of epochs to train

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(inputs)
    loss = torch.nn.functional.binary_cross_entropy(outputs, labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss every 1000 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. Evaluate the model
with torch.no_grad():  # No need to track gradients for evaluation
    predictions = model(inputs)
    rand_idx = np.random.randint(len(inputs), size=100)
    x_sample = [string_values[r]for r in rand_idx]
    y_sample = predictions[rand_idx]

    for x,y in zip(x_sample,y_sample):
        print (f"input: {x} \n output: {[f for f in y]}")
