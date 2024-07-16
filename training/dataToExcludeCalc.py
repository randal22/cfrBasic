import csv
from decimal import Decimal, InvalidOperation
import os
#this code is used to count the number of single visit instances in the cfr data
def count_distributions(file_path):
    count_50_50 = 0
    count_33_33_33 = 0

    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip the header row
        for row_num, row in enumerate(csv_reader, start=2):  # Start from 2 as we skipped the header
            try:
                # Split the first column by colon
                label, *values = row[0].split(':')
                # If there were no additional columns, use the split values
                # Otherwise, use the additional columns
                probabilities = values if len(row) == 1 else row[1:]
                
                # Convert to Decimal, skipping empty strings
                probabilities = [Decimal(p) for p in probabilities if p.strip()]
                
                if len(probabilities) == 2 and all(p == Decimal('0.5') for p in probabilities):
                    count_50_50 += 1
                elif len(probabilities) == 3 and all(p == Decimal('0.3333333432674408') for p in probabilities):
                    count_33_33_33 += 1
            except (InvalidOperation, IndexError) as e:
                print(f"Error in row {row_num}: {row}")
                print(f"Error message: {str(e)}")

    return count_50_50, count_33_33_33

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to strategy.csv
file_path = os.path.join(current_dir, 'strategy.csv')

count_50_50, count_33_33_33 = count_distributions(file_path)

print(f"Number of 0.5, 0.5 distributions: {count_50_50}")
print(f"Number of 0.333..., 0.333..., 0.333... distributions: {count_33_33_33}")