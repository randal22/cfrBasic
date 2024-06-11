def encode_strings(string_vals, substrings):
    # Encode the strings into a list of 3 by 13 arrays based on the presence of substrings
    encoded_arrays = []
    for string_val in string_vals:
        encoded_array = [[0] * 13, [0] * 13, [0] * 13]  # Initialize the array with zeros
        PostCounter=0
        # Check for presence of substrings before and after the colon
        for index, sub in enumerate(substrings):
            
            if sub in string_val.lower():
                
                if ':' in string_val and sub in string_val.lower().split(':')[0]:
                    encoded_array[0][index] = 1  # Card type before colon
                elif ':' in string_val and sub in string_val.lower().split(':')[1]:
                        ##count and print number of substrings after the colon
                        after_colon_substrings = string_val.split(':')[1].lower()
                        count_after_colon = sum(1 for sub in substrings if sub in after_colon_substrings)
                        #print(f"Number of substrings after the colon: {count_after_colon}")

                        if (count_after_colon%2==0):
                              encoded_array[1][index] = 1  #middle row because no card to respond to
                        else:
                            if  count_after_colon==1: #1 card
                                encoded_array[2][index] = 1  # Last row as it's a card to respond to
                            else: #3 substrings after, if not last, middle row, else last and 3rd row
                                PostCounter+=1
                                print(PostCounter)
                                if PostCounter==3:
                                    encoded_array[2][index] = 1
                                    
                                else:
                                    encoded_array[1][index] = 1
                
          
        encoded_arrays.append(encoded_array)

    return encoded_arrays

# Example usage:
string_vals = ["TwoNineKing:Ten", "JackQueen:KingAce", "ThreeFour:FiveSixTen","Six,Jack:EightTenSeven"]  # Example array of strings
substrings = ['two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'jack', 'queen', 'king', 'ace']
encoded_arrays = encode_strings(string_vals, substrings)
# Print the encoded arrays
counter=0
for encoded_array in encoded_arrays:
    print (string_vals[counter])
    for row in encoded_array:
        print(row)
    print("---")
    
    counter+=1