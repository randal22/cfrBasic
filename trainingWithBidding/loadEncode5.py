import csv
import torch
import random

def loadDataFile(file_path):
    strings = []
    values_sets = []
    with open(file_path, 'r') as file:
        next(file)  # Skip the header row
        for line in file:
            line = line.strip()
            if not line:
                continue
            # Split the line into game string and float values at the first colon
            parts = line.split(':', 1)
            if len(parts) < 2:
                print(f"Skipping invalid line: {line}")
                continue
            game_str, values_str = parts[0], parts[1]
            # Convert values to floats
            try:
                double_vals = list(map(float, values_str.split(',')))
            except ValueError as e:
                print(f"Error converting values in line: {line} - {e}")
                continue
            # Store the game string and values
            strings.append(game_str)
            # Ensure exactly 6 values, fill with 0 if necessary
            double_labels_tensor = torch.zeros(6)
            for i in range(min(len(double_vals), 6)):
                double_labels_tensor[i] = double_vals[i]
            values_sets.append(double_labels_tensor)
    if values_sets:
        values_sets = torch.stack(values_sets, dim=0)
    else:
        values_sets = torch.empty(0, 6)
    return strings, values_sets


def encodeGameStrings(string_values, substrings):
    # Initialize tensor with shape (num_strings, 5, 13)
    encoded_arrays = torch.zeros(len(string_values), 5, 13, dtype=torch.float)
    
    # Card name to index mapping
    card_to_idx = {
        'Two': 0, 'Three': 1, 'Four': 2, 'Five': 3, 'Six': 4, 
        'Seven': 5, 'Eight': 6, 'Nine': 7, 'Ten': 8, 'Jack': 9, 
        'Queen': 10, 'King': 11, 'Ace': 12
    }
    
    # Bid string to value mapping
    bid_to_value = {f'Bid{i}': i for i in range(6)}  # Creates mapping for Bid0 through Bid5
    total = len(string_values)
    update_interval = max(1, total // 100)  # Update every 1% of total
    
    for tensor_idx, game_string in enumerate(string_values):
        # Split the game string into its components, handling empty sections
        components = game_string.rstrip(':').split(';')  # Remove trailing ':' if present
        
        # Always have player index and hand
        current_player_idx = int(components[0])
        hand_cards = components[1].split(',') if components[1] else []
        
        # Initialize other components as empty
        bids = []
        action_history_str = ""
        player_idx_history = []
        scores = []
        
        # Parse remaining components based on what's available
        if len(components) > 2 and components[2]:  # Bids
            bid_strings = components[2].split(',')
            bids = [bid_to_value[bid] for bid in bid_strings if bid in bid_to_value]
            
        if len(components) > 3 and components[3]:  # Action history as concatenated string
            action_history_str = components[3]
            
        if len(components) > 4 and components[4]:  # Player index history
            player_idx_history = [int(idx) for idx in components[4].split(',')]
            
        if len(components) > 5 and components[5]:  # Scores
            scores = [int(score) for score in components[5].split(',')]
        
        # Row 1: Player hand (binary encoding)
        for card in hand_cards:
            if card in card_to_idx:
                encoded_arrays[tensor_idx, 0, card_to_idx[card]] = 1
        
        # Row 2: Action history - Parse the concatenated string and set order values
        if action_history_str:
            # Parse the concatenated action history string into individual card names
            parsed_actions = []
            remaining = action_history_str
            
            while remaining:
                found = False
                for card_name in sorted(card_to_idx.keys(), key=len, reverse=True):  # Try longer names first
                    if remaining.startswith(card_name):
                        parsed_actions.append(card_name)
                        remaining = remaining[len(card_name):]
                        found = True
                        break
                if not found:
                    # If no card name was found at the start, skip one character
                    remaining = remaining[1:]
            
            # Now encode row 2 with the sequential play order (1st card=1, 2nd card=2, etc.)
            for i, card_name in enumerate(parsed_actions):
                card_idx = card_to_idx[card_name]
                # Set the value to the play order (i+1) - first card played gets 1, second gets 2, etc.
                encoded_arrays[tensor_idx, 1, card_idx] = i + 1
            
            # Row 3: Player index history - Link player indices to the cards they played
            if player_idx_history:
                for i, (card_name, player_idx) in enumerate(zip(parsed_actions, player_idx_history)):
                    if i < len(parsed_actions) and i < len(player_idx_history):
                        card_idx = card_to_idx[card_name]
                        encoded_arrays[tensor_idx, 2, card_idx] = player_idx + 1  # +1 to avoid 0
        
        # Row 4: Cards not in hand or play history (remaining cards)
        all_cards_seen = set()
        # Add cards from hand
        all_cards_seen.update(card_to_idx[card] for card in hand_cards if card in card_to_idx)
        # Add cards from parsed action history
        if action_history_str:
            all_cards_seen.update(card_to_idx[card] for card in parsed_actions if card in card_to_idx)
        
        # Mark remaining cards
        for i in range(13):
            if i not in all_cards_seen:
                encoded_arrays[tensor_idx, 3, i] = 1
        
        # Row 5: Special information
        encoded_arrays[tensor_idx, 4, 0] = current_player_idx + 1  # Current player index (+1)
        
        # Only set bids and scores if they exist
        if len(bids) >= 1:
            encoded_arrays[tensor_idx, 4, 1] = bids[0] + 10
        if len(bids) >= 2:
            encoded_arrays[tensor_idx, 4, 2] = bids[1] + 10
        if len(scores) >= 1:
            encoded_arrays[tensor_idx, 4, 3] = scores[0] + 100
        if len(scores) >= 2:
            encoded_arrays[tensor_idx, 4, 4] = scores[1] + 100
        
        if tensor_idx % update_interval == 0 or tensor_idx == total - 1:
            progress = (tensor_idx + 1) / total * 100
            print(f"\rEncoding: {progress:.2f}% complete", end='', flush=True)
    print()
    return encoded_arrays

def print_examples(strings, encoded_tensors, num_examples=10):
    indices = random.sample(range(len(strings)), min(num_examples, len(strings)))
    for i, idx in enumerate(indices):
        print(f"\nExample {i+1}/{num_examples}")
        print("Game String:", strings[idx])
        print("Encoded Tensor:")
        # Print each row of the 5x13 tensor
        for row_idx in range(5):
            row = encoded_tensors[idx, row_idx].tolist()
            print(f"Row {row_idx+1}: {row}")
        print("-" * 50) 
    

substrings = ['Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Jack', 'Queen', 'King', 'Ace','Bid0','Bid1','Bid2','Bid3','Bid4','Bid5']
file_path = 'StrategyTest.csv'
string_values, sets_of_values = loadDataFile(file_path)
encoded_states = encodeGameStrings(string_values, substrings)
#print example 10 random tensors alongside their respective strings
#print_examples(string_values, encoded_states)
