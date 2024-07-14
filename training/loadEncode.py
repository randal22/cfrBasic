import csv
import torch
import torch.nn as nn
import torch.optim as optim

uniform_two_vec = torch.tensor([1, 1, 0]) / 2
uniform_three_vec = torch.tensor([1, 1, 1]) / 3
def load_csv_file(file_path):
    strings = []
    values_sets = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            # Assuming the format is string double double or string double double double
            if len(row) >= 2:  # At least 2 columns
                try:
                    string_val = ':'.join(row[:-len(row)+1])  # Join all elements except the last ones
                    double_vals = [float(val) for val in row[-len(row)+1:]]  # Convert remaining values to float
                    double_labels_tensor=torch.zeros(3)
                    for i, v in enumerate (double_vals):
                        double_labels_tensor[i]=v
                    #ignore single visited games, as they negatively impact the net data
                    if torch.allclose(uniform_two_vec,double_labels_tensor) or torch.allclose(uniform_three_vec,double_labels_tensor):
                        continue
                    strings.append(string_val)
                    values_sets.append(double_labels_tensor)
                except ValueError:
                    print("Error converting values in row:", row)
            else:
                print("Invalid row format:", row)
    values_sets=torch.stack(values_sets,dim=0)
    return strings, values_sets

def encode_strings(string_values, substrings):
    # Encode the strings into a list of 3 by 13 arrays based on the presence of substrings
    encoded_arrays = torch.zeros(len(string_values),3,13)
    for tensorIndex, string_values in enumerate(string_values):
        #encoded_array = [[0] * 13, [0] * 13, [0] * 13]  # Initialize the array with zeros
        PostCounter=0
        # Check for presence of substrings before and after the colon
        for index, sub in enumerate(substrings):
            
            if sub in string_values.lower():
                
                if ':' in string_values and sub in string_values.lower().split(':')[0]:
                    encoded_arrays[tensorIndex][0][index] = 1  # Card type before colon
                elif ':' in string_values and sub in string_values.lower().split(':')[1]:
                        ##count and print number of substrings after the colon
                        after_colon_substrings = string_values.split(':')[1].lower()
                        count_after_colon = sum(1 for sub in substrings if sub in after_colon_substrings)
                        #print(f"Number of substrings after the colon: {count_after_colon}")

                        if (count_after_colon%2==0):
                              encoded_arrays[tensorIndex][1][index] = 1  #middle row because no card to respond to
                        else:
                            if  count_after_colon==1: #1 card
                                encoded_arrays[tensorIndex][2][index] = 1  # Last row as it's a card to respond to
                            else: #3 substrings after, if not last, middle row, else last and 3rd row
                                PostCounter+=1
                                #print(PostCounter)
                                if PostCounter==3:
                                    encoded_arrays[tensorIndex][2][index] = 1
                                    
                                else:
                                    encoded_arrays[tensorIndex][1][index] = 1
                
          
        #encoded_arrays.append(encoded_array)

    return encoded_arrays


file_path = 'Strategy.csv'
string_values, sets_of_values = load_csv_file(file_path)
#print("String Values:", string_values)
#print("Sets of Values:", sets_of_values)
substrings = ['two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'jack', 'queen', 'king', 'ace']

encoded_arrays = encode_strings(string_values, substrings)
counter=0
#for encoded_array in encoded_arrays:
    #print (string_values[counter])
    #print (sets_of_values[counter])
    #for row in encoded_array:
        #print(row)
    #print("---")
    #counter+=1
    #if(counter>5):
        #break
    
