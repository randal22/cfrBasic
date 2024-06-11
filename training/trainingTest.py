import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define the XOR dataset
# XOR inputs and corresponding labels
inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
labels = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# 2. Define the neural network model
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.hidden = nn.Linear(2, 8)  # Hidden layer with 8 neurons
        self.output = nn.Linear(8, 1)  # Output layer receiving 8

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))  # Activation function for hidden layer
        x = torch.sigmoid(self.output(x))  # Activation function for output layer
        return x

# Instantiate the model
model = XORModel()

# 3. Define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)  # Stochastic Gradient Descent

# 4. Train the model
num_epochs = 10000  # Number of epochs to train

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss every 1000 epochs
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. Evaluate the model
with torch.no_grad():  # No need to track gradients for evaluation
    predictions = model(inputs)
    print(f'\nPredictions:\n{predictions}')
    print(f'Rounded Predictions:\n{torch.round(predictions)}')
    print(f'Actual Labels:\n{labels}')