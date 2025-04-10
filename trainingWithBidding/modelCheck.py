import torch
from loadEncode5 import encoded_states, string_values
#this code is to check the  model, to allow user to spot a bad model by eye before doing proper testing against agents etc

# Define the model architecture (must match the saved model)
class XORModel(torch.nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.hidden1 = torch.nn.Linear(65, 128)
        self.hidden2 = torch.nn.Linear(128, 64)
        self.output = torch.nn.Linear(64, 6)

    def forward(self, x):
        x = x.view(-1, 5*13)
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.nn.functional.softmax(self.output(x), dim=1)
        return x

# Load the saved model
model = XORModel()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Generate some test examples
test_inputs = encoded_states[:20]  # Use the first 10 examples from encoded_arrays
test_strings = string_values[:20]  # Corresponding string values

# Run the model on test examples
with torch.no_grad():
    predictions = model(test_inputs)

# Print results
for input_string, prediction in zip(test_strings, predictions):
    print(f"Input: {input_string}")
    print(f"Output: {prediction.tolist()}")
    print("---")