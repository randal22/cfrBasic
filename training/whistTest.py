import random
import torch
class XORModel(torch.nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.hidden1 = torch.nn.Linear(39, 64)
        self.hidden2 = torch.nn.Linear(64, 64)
        self.output = torch.nn.Linear(64, 3)

    def forward(self, x):
        x = x.view(-1, 3*13)
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.nn.functional.softmax(self.output(x), dim=1)
        return x


class WhistState:
    def __init__(self):
        self.shuffledDeck = random.sample(baseDeck, len(baseDeck))
        self.p1Hand = sorted(self.shuffledDeck[:3], key=lambda x: baseDeck.index(x))
        self.p2Hand = sorted(self.shuffledDeck[3:6], key=lambda x: baseDeck.index(x))
        self.gameString = ""
        self.turnCounter = 0
        self.scores =[0,0]
    def step(self,action):
        #add action to game string
        self.gameString = self.gameString + action
        current_player = self.turnCounter % 2
        #remove played card from player hand
        if current_player == 0:
            if action in self.p1Hand:
                self.p1Hand.remove(action)
        else:
            if action in self.p2Hand:
                self.p2Hand.remove(action)
        self.turnCounter += 1

    def to_matrix(self,state):
        encoded_tensor = torch.zeros(3, 13)
        post_counter = 0

        for index, sub in enumerate(substrings):
            if sub in state.lower():
                if ':' in state:
                    before_colon, after_colon = state.lower().split(':')
                    if sub in before_colon:
                        encoded_tensor[0, index] = 1
                    elif sub in after_colon:
                        count_after_colon = sum(1 for s in substrings if s in after_colon)
                    
                        if count_after_colon % 2 == 0:
                            encoded_tensor[1, index] = 1
                        else:
                            if count_after_colon == 1:
                                encoded_tensor[2, index] = 1
                            else:
                                post_counter += 1
                                if post_counter == 3:
                                    encoded_tensor[2, index] = 1
                                else:
                                    encoded_tensor[1, index] = 1
        return encoded_tensor
    @staticmethod
    def get_valid_action(model_output, hand_size):
        # Get the probabilities for each action
        probs = torch.nn.functional.softmax(model_output, dim=1)[0]
        # Only consider the first 'hand_size' probabilities
        valid_probs = probs[:hand_size]
        # Return the index of the highest probability among valid actions
        return valid_probs.argmax().item()
    @staticmethod
    def randomAction(hand):
        return random.randint(0, len(hand) - 1)

# Load the saved model
model = XORModel()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
baseDeck=["Two","Three","Four","Five","Six","Seven","Eight","Nine","Ten","Jack","Queen","King","Ace"]
substrings = ['two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'jack', 'queen', 'king', 'ace']
games=10000
game_results = []

for x in range(games):
    gameState = WhistState()
    gameOver = False

    while not gameOver:
        # Player 1's turn
        if len(gameState.p1Hand) == 1:
            p1_action = 0
        elif x < games // 2:  # First half of games: Player 1 uses model
            p1_input = '+'.join(gameState.p1Hand) + ':' + gameState.gameString
            p1_state = gameState.to_matrix(p1_input)
            model_output = model(p1_state.unsqueeze(0))
            p1_action = WhistState.get_valid_action(model_output, len(gameState.p1Hand))
        else:  # Second half: Player 1 uses random action
            p1_action = WhistState.randomAction(gameState.p1Hand)
        
        p1_card = gameState.p1Hand[p1_action]
        gameState.step(p1_card)

        # Player 2's turn
        if len(gameState.p2Hand) == 1:
            p2_action = 0
        elif x >= games // 2:  # Second half of games: Player 2 uses model
            p2_input = '+'.join(gameState.p2Hand) + ':' + gameState.gameString
            p2_state = gameState.to_matrix(p2_input)
            model_output = model(p2_state.unsqueeze(0))
            p2_action = WhistState.get_valid_action(model_output, len(gameState.p2Hand))
        else:  # First half: Player 2 uses random action
            p2_action = WhistState.randomAction(gameState.p2Hand)
        
        p2_card = gameState.p2Hand[p2_action]
        gameState.step(p2_card)

        # Compare plays and update scores
        if baseDeck.index(p1_card) > baseDeck.index(p2_card):
            gameState.scores[0] += 1
        elif baseDeck.index(p1_card) < baseDeck.index(p2_card):
            gameState.scores[1] += 1

        # Check if the game is over
        if max(gameState.scores) > 1:
            gameOver = True

    game_results.append((gameState.scores[0], gameState.scores[1]))
    #print(f"Game {x+1} finished. Scores: Player 1 - {gameState.scores[0]}, Player 2 - {gameState.scores[1]}")

# After all games, print summary
p1_model_wins = sum(1 for i in range(games // 2) if game_results[i][0] > game_results[i][1])
p2_model_wins = sum(1 for i in range(games // 2, games) if game_results[i][1] > game_results[i][0])

print(f"Player 1 (using model) won {p1_model_wins} out of {games // 2} games against an opponent playing randomly")
print(f"Player 2 (using model) won {p2_model_wins} out of {games // 2} games against an opponent playing randomly")
