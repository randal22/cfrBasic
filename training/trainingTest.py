import torch
import torch.nn as nn
import torch.optim as optim
from loadEncode import encoded_arrays,sets_of_values,string_values
import numpy as np
from torch.nn.utils import clip_grad_norm_
#this code is the training of the model, using load encode to be able to read the CFR data


#Define the XOR dataset
# XOR inputs and corresponding labels
inputs = encoded_arrays
labels = sets_of_values

#Define the neural network model
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
    def clip_weights(self, min_val, max_val):
        for param in self.parameters():
            param.data.clamp_(min_val, max_val)

# Instantiate the model
model = XORModel()

# Define the optimiser
optimiser = optim.AdamW(model.parameters(),weight_decay=0.1)  # Stochastic Gradient Descent

# Train the model
num_epochs = 10000  # Number of epochs to train
model_save_path = 'best_model.pth'
max_grad_norm = 0.5 # Set the maximum norm of the gradient
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(inputs)
    loss = torch.nn.functional.binary_cross_entropy(outputs, labels)

    # Backward pass and optimization
    optimiser.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), max_grad_norm)
    optimiser.step()
    model.clip_weights(-1,1)

    # Print loss to allow user to see if model is progressing as expected
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

#Evaluate the model
with torch.no_grad():  # No need to track gradients for evaluation
    predictions = model(inputs)
    rand_idx = np.random.randint(len(inputs), size=100)
    x_sample = [string_values[r]for r in rand_idx]
    y_sample = predictions[rand_idx]

    for x,y in zip(x_sample,y_sample):
        print (f"input: {x} \n output: {[f for f in y]}")

torch.save(model.state_dict(), model_save_path)#save model