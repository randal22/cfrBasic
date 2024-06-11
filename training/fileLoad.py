import csv
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
                    strings.append(string_val)
                    values_sets.append(double_vals)
                except ValueError:
                    print("Error converting values in row:", row)
            else:
                print("Invalid row format:", row)
    return strings, values_sets

# Example usage:
file_path = 'Strategy.csv'
string_values, sets_of_values = load_csv_file(file_path)
#print("String Values:", string_values)
#print("Sets of Values:", sets_of_values)
substrings = ['Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Jack', 'Queen', 'King', 'Ace']