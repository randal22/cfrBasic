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
        self.lastPlay=""
    def step(self,action):
        #add action to game string
        self.gameString = self.gameString + action
        current_player = self.turnCounter % 2
        self.lastPlay=action
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
    def greedyAction(hand,turn,prevPlay):
        if (len(hand)==1):
            return 0 #handsize 1 so play only card
        else:
            #check if even or odd turn
            if (turn%2==0):#even
                #lead highest card for best chance at winning immediate trick
                return len(hand)-1
            else:#odd
                #iterate through hand looking for a winning play against opponents last play
                target=baseDeck.index(prevPlay)
                
                for counter, card in enumerate(hand):
                    if baseDeck.index(card) > target:
                        return counter
                #winning play not found, play lowest card
                return 0

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
def modelVsRandom(model, games):
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

    # After all games, calculate summary
    p1_model_wins = sum(1 for i in range(games // 2) if game_results[i][0] > game_results[i][1])
    p2_model_wins = sum(1 for i in range(games // 2, games) if game_results[i][1] > game_results[i][0])

    return p1_model_wins, p2_model_wins, games // 2
def modelVsGreedy(model, games):
    game_results = []
    for x in range(games):
        gameState = WhistState()
        gameOver = False
        #print(f"Starting game {x+1}")
        while not gameOver:
            # Player 1's turn
            if len(gameState.p1Hand) == 1:
                p1_action = 0
            elif x < games // 2:  # First half of games: Player 1 uses model
                p1_input = '+'.join(gameState.p1Hand) + ':' + gameState.gameString
                p1_state = gameState.to_matrix(p1_input)
                model_output = model(p1_state.unsqueeze(0))
                p1_action = WhistState.get_valid_action(model_output, len(gameState.p1Hand))
            else:  # Second half: Player 1 uses greedy action
                #print ("(p1 greed) Hand ")
                #print(gameState.p1Hand)
                #print (gameState.lastPlay)
                p1_action = WhistState.greedyAction(gameState.p1Hand, gameState.turnCounter, gameState.lastPlay)
            
            p1_card = gameState.p1Hand[p1_action]
            #if (x>=games/2):
                #print ("Playing highest card ")
                #print(p1_card)#check if always leading high
            gameState.step(p1_card)

            # Player 2's turn
            if len(gameState.p2Hand) == 1:
                p2_action = 0
            elif x >= games // 2:  # Second half of games: Player 2 uses model
                p2_input = '+'.join(gameState.p2Hand) + ':' + gameState.gameString
                p2_state = gameState.to_matrix(p2_input)
                model_output = model(p2_state.unsqueeze(0))
                p2_action = WhistState.get_valid_action(model_output, len(gameState.p2Hand))
            else:  # First half: Player 2 uses greedy action
                #print ("(greedy p2 )Hand ")
                #print(gameState.p2Hand)
                #print ("Responding to "+gameState.lastPlay)
                p2_action = WhistState.greedyAction(gameState.p2Hand, gameState.turnCounter, gameState.lastPlay)
            
            p2_card = gameState.p2Hand[p2_action]
            #if (x<games/2):
                #print("greedy/lowest play= "+p2_card)#check if winning tricks asap/lowest
            gameState.step(p2_card)

            # Compare plays and update scores
            if baseDeck.index(p1_card) > baseDeck.index(p2_card):
                gameState.scores[0] += 1
            elif baseDeck.index(p1_card) < baseDeck.index(p2_card):
                gameState.scores[1] += 1

            # Check if the game is over
            if max(gameState.scores) > 1:
                gameOver = True
                #print(f"game {x+1} over")

        game_results.append((gameState.scores[0], gameState.scores[1]))

    # After all games, calculate summary
    p1_model_wins = sum(1 for i in range(games // 2) if game_results[i][0] > game_results[i][1])
    p2_model_wins = sum(1 for i in range(games // 2, games) if game_results[i][1] > game_results[i][0])

    return p1_model_wins, p2_model_wins, games // 2
def modelVsModel(model, games):
    game_results = []
    for x in range(games):
        gameState = WhistState()
        gameOver = False
        
        while not gameOver:
            # Player 1's turn
            if len(gameState.p1Hand) == 1:
                p1_action = 0
            else:
                p1_input = '+'.join(gameState.p1Hand) + ':' + gameState.gameString
                p1_state = gameState.to_matrix(p1_input)
                model_output = model(p1_state.unsqueeze(0))
                p1_action = WhistState.get_valid_action(model_output, len(gameState.p1Hand))
            
            p1_card = gameState.p1Hand[p1_action]
            gameState.step(p1_card)

            # Player 2's turn
            if len(gameState.p2Hand) == 1:
                p2_action = 0
            else:
                p2_input = '+'.join(gameState.p2Hand) + ':' + gameState.gameString
                p2_state = gameState.to_matrix(p2_input)
                model_output = model(p2_state.unsqueeze(0))
                p2_action = WhistState.get_valid_action(model_output, len(gameState.p2Hand))
            
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

    # After all games, calculate summary
    p1_wins = sum(1 for result in game_results if result[0] > result[1])
    p2_wins = games - p1_wins  # Since there are no draws, P2 wins all games P1 doesn't win

    return p1_wins, p2_wins, games



iterations=1000000
iterations=int(iterations)
p1_wins, p2_wins, total_games = modelVsRandom(model, iterations)
print(f"Player 1 (using model) won {p1_wins} out of {total_games} games against an opponent playing randomly")
print(f"Player 2 (using model) won {p2_wins} out of {total_games} games against an opponent playing randomly")
p1_wins2, p2_wins2, total_games2 = modelVsGreedy(model, iterations)
print(f"Player 1 (using model) won {p1_wins2} out of {total_games2} games against an opponent playing greedily")
print(f"Player 2 (using model) won {p2_wins2} out of {total_games2} games against an opponent playing greedily")
iterations=iterations//2
p1_wins3, p2_wins3, total_games3 = modelVsModel(model, iterations)
print(f"In model vs model games:")
print(f"Player 1 won {p1_wins3} out of {total_games3} games")
print(f"Player 2 won {p2_wins3} out of {total_games3} games")