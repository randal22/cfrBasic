using System;
using System.Collections.Generic;
using System.Diagnostics.Metrics;
using System.Linq;

// Much of this implementation follows the design of https://github.com/brianberns/Cfrm,
// applied to simplified-Whist instead of Kuhn Poker.
namespace Cfrm.SimplifiedWhist
{
    public class Whist
    {
        public enum Card
        {
            Two,
            Three,
            Four,
            Five,
            Six,
            Seven,
            Eight,
            Nine,
            Ten,
            Jack,
            Queen,
            King,
            Ace
        }
        //possible actions
        public enum Action
        {
            Two,
            Three,
            Four,
            Five,
            Six,
            Seven,
            Eight,
            Nine,
            Ten,
            Jack,
            Queen,
            King,
            Ace,
            Bid0,
            Bid1,
            Bid2,
            Bid3,
            Bid4,
            Bid5
        }
        public class WhistState
            : GameState<Action>
        {
            //default constructor
            public WhistState(Card[] cards)
                : this(cards, new Action[0])
            {

            }
            private WhistState(Card[] cards, Action[] actions)
            {
                //entire shuffled deck
                _cards = cards;
                _p1Hand = new Card[] { cards[0], cards[2], cards[4], cards[6], cards[8] };
                Array.Sort(_p1Hand);
                _p2Hand = new Card[] { cards[1], cards[3], cards[5], cards[7], cards[9] };
                Array.Sort(_p2Hand);
                //no actions at start
                _actions = actions;

            }
            private readonly Card[] _cards;
            private readonly Action[] _actions;
            public Card[] _p1Hand;
            public Card[] _p2Hand;


            //old ActionString from original implementation
            private string ActionString
            {
                get
                {
                    var chars = _actions.Select(action => action.ToString()[0]).ToArray();
                    return new string(chars);
                }
            }
            //turn checking
            //needs to returns 0 or 1 for p1/p2
            public override int CurrentPlayerIdx
            {
                get
                {

                    //check if still in bidding phase
                    if (_actions.Length == 0) //p1 to bid
                    {
                        return 0;
                    }
                    else if (_actions.Length == 1) //p2 to bid
                    {
                        return 1;
                    }
                    else if (_actions.Length == 2) //bidding over, p2 to start
                    {
                        return 1;
                    }
                    else if (_actions.Length == 3)
                    {
                        return 0;
                    }
                    else
                    {
                        //out of bidding phase and first turn


                        if (_actions.Length % 2 == 0) // winner of previous trick is new leader 
                        {
                            //determine who won last trick by comparing last two actions and who played them
                            Card[] cardsToComp = new Card[2];
                            cardsToComp[0] = (Card)_actions[_actions.Length - 2];
                            cardsToComp[1] = (Card)_actions[_actions.Length - 1];

                            if ((cardsToComp[0].CompareTo(cardsToComp[1])) > 0)
                            {
                                // find out who played the card at position 0

                                if (_p1Hand.Contains(cardsToComp[0]))
                                {
                                    return 0;
                                }
                                else
                                {
                                    return 1;
                                }
                            }
                            else
                            {
                                //draws are not possible here
                                // find out who played the card at position 1
                                if (_p1Hand.Contains(cardsToComp[1]))
                                {
                                    return 0;
                                }
                                else
                                {
                                    return 1;
                                }


                            }

                        }
                        else // mid turn - it is the player who did not have the last played card in their openeing hand to play
                        {
                            //get last played card
                            Card target = (Card)_actions[_actions.Length - 1];
                            if (_p1Hand.Contains(target))
                            {
                                return 1;
                            }
                            else
                            {
                                return 0;
                            }

                        }

                    }
                }
            }
            /*
            public override string Key //this may need reworked 
            {
                get
                {
                    string[] LegActStr = new string[LegalActions.Length];
                    //the key should be the current players hand minus their played cards + game history
                    string tempKey = "";
                    for (int i = 0; i < LegalActions.Length; i++)
                    {
                        LegActStr[i] = LegalActions[i].ToString();
                        tempKey += (LegActStr[i]);
                    }

                    tempKey += ":";
                    string[] ActStr = new string[_actions.Length];
                    for (int j = 0; j < _actions.Length; j++)
                    {
                        ActStr[j] = _actions[j].ToString();
                        tempKey += (ActStr[j]);
                    }
                    return tempKey;
                }
            }*/

            public override string Key
            {
                get
                {
                    var currentPlayerIdx = CurrentPlayerIdx;
                    string remainingCards;
                    string bids = "";
                    if (_actions.Length > 0)
                    {
                        if(_actions.Length == 1)
                        {
                            bids =string.Join(",", _actions[0]);
                        }
                        else
                        {

                            bids = string.Join(",", _actions[0], _actions[1]);

                        }
                    }
                    
                    string actionHistory = "";
                    if (playHistoryBoth!=null)
                    {
                        //set action history to actions minus bids
                        actionHistory=string.Join("", playHistoryBoth);
                    }
                    // Handle bidding phase (first two moves) separately
                    if (_actions.Length == 0)
                    {
                        //no history, no bids, no scores
                        //p1 to bid
                        //return playeridx,hand only
                        string strHand = string.Join(",", _p1Hand);
                        return $"{currentPlayerIdx};{strHand}:";
                    }
                    else if (_actions.Length == 1)
                    {
                        string strHand = string.Join(",", _p2Hand);
                        //only other player's bid in history, no scores
                        //Console.WriteLine($"edgecase saved{currentPlayerIdx};{strHand};{actionHistory}");
                        return $"{currentPlayerIdx};{strHand};{bids}:";
                    }
                    else if (_actions.Length == 2)
                    {
                        //p2 first turn
                        //idx;hand;bids;end
                        string strHand = string.Join(",", _p2Hand);
                        //only other player's bid,no history, no scores
                        return $"{currentPlayerIdx};{strHand};{bids}:";
                    }
                    else if (_actions.Length == 3)
                    {
                        //p1 first turn, responding, all except scores
                        //only entry in play idx history is known to be p1
                        string strHand = string.Join(",", _p1Hand);
                        //Console.WriteLine($"edgecase saved{currentPlayerIdx};{strHand};{actionHistory};1:");
                        return $"{currentPlayerIdx};{strHand};{bids};{actionHistory};1:";

                    }
                    else
                    {


                        // For gameplay phase, calculate remaining cards based on played history
                        var p1Played = playHistoryP1 ?? new Action[0];
                        var p2Played = playHistoryP2 ?? new Action[0];

                        if (currentPlayerIdx == 0)
                        {
                            var p1Remaining = _p1Hand.Where(c => !p1Played.Contains((Action)c)).OrderBy(c => c);
                            remainingCards = string.Join(",", p1Remaining);
                        }
                        else
                        {
                            var p2Remaining = _p2Hand.Where(c => !p2Played.Contains((Action)c)).OrderBy(c => c);
                            remainingCards = string.Join(",", p2Remaining);
                        }


                        int[] tricks = GetTricksWon();
                        string playIdxHistory = "";

                        playIdxHistory = string.Join(",", playHistoryPlayeridx);

                        if (playHistoryPlayeridx.Length != (_actions.Length - 2))
                        {
                            Console.WriteLine($"WARNING idx hist and action hist mismatch detected{actionHistory};{playIdxHistory}");
                        }
                        return $"{currentPlayerIdx};{remainingCards};{bids};{actionHistory};{playIdxHistory};{tricks[0]},{tricks[1]}:";
                    }
                }
            }


            //aquires the trick history as an array of actions, all of player 1's plays first then player 2s, split halfway


            private Action[] playHistoryP1
            {
                get
                {

                    //empty action history check
                    //skip first two actions bc bidding
                    if (_actions.Length < 2)
                    {
                        return null;
                    }
                    // Skip first two actions (bidding)
                    Action[] reducedActionHist = new Action[_actions.Length - 2];
                    for (int i = 2; i < _actions.Length; i++)
                    {
                        reducedActionHist[i - 2] = _actions[i];
                    }

                    // Initialize array to store P1's moves
                    List<Action> P1Hist = new List<Action>();

                    // For each action, check if it was from P1's starting hand
                    for (int i = 0; i < reducedActionHist.Length; i++)
                    {
                        if (_p1Hand.Contains((Card)reducedActionHist[i]))
                        {
                            P1Hist.Add(reducedActionHist[i]);
                        }
                    }

                    return P1Hist.ToArray();
                }
            }
            private Action[] playHistoryP2
            {
                get
                {
                    if (_actions.Length < 2)
                    {
                        return null;
                    }
                    // Skip first two actions (bidding)
                    Action[] reducedActionHist = new Action[_actions.Length - 2];
                    for (int i = 2; i < _actions.Length; i++)
                    {
                        reducedActionHist[i - 2] = _actions[i];
                    }

                    // Initialize array to store P2's moves
                    List<Action> P2Hist = new List<Action>();

                    // For each action, check if it was from P2's starting hand
                    for (int i = 0; i < reducedActionHist.Length; i++)
                    {
                        if (_p2Hand.Contains((Card)reducedActionHist[i]))
                        {
                            P2Hist.Add(reducedActionHist[i]);
                        }
                    }

                    return P2Hist.ToArray();
                }
            }


            private Action[] playHistoryBoth
            {
                get
                {
                    int len = _actions.Length - 2;

                    if (len < 1)
                    {
                        return null;
                    }
                    else
                    {
                        Action[] reducedActionHist = new Action[_actions.Length - 2];
                        for (int i = 2; i < _actions.Length; i++)
                        {
                            reducedActionHist[i - 2] = _actions[i];
                        }
                        return reducedActionHist;
                    }



                }
            }

            /*
            //old score calculation
            private int[] scores //needs reworked 
            {
                get
                {
                    int bidP1;
                    int bidP2;
                    if (trickHistory != null)
                    {
                        if (trickHistory.Length != 10)
                        {
                            //game ongoing
                            // score calc midgame to enable score tracking 


                            return null;
                        }
                        else
                        {
                            bidP1 = (int)_actions[0] - 13;
                            bidP2 = (int)_actions[1] - 13;
                        }
                        //by using the seperate play histories, we can deduce who won which tricks

                        int[] tempScores = new int[2];

                        for (int i = 0; i < trickHistory.Length / 2; i++)
                        {
                            int result = playHistoryP1[i].CompareTo(playHistoryP2[i]);
                            if (result > 0)
                            {
                                //p1 won the trick
                                tempScores[0]++;
                            }
                            else if (result < 0)
                            {//p2 won the trick
                                tempScores[1]++;
                            }//there is no drawing here


                        }
                        // now we have collected the scores, we need to check against the players bids, to see if they hit their bonus

                        if (tempScores[0] == bidP1)
                        {
                            tempScores[0] += 10;

                        }
                        if (tempScores[1] == bidP2)
                        {
                            tempScores[1] += 10;
                        }
                        return tempScores;
                        /*
                        int gameLength = _actions.Length;
                        if (gameLength % 2 == 1)
                        {
                            gameLength--;
                        }
                        double[] tempScores = new double[2];
                        for (int i = 0; i < gameLength / 2; i++)
                        {
                            int result = trickHistory[i].CompareTo(trickHistory[i + gameLength / 2]);
                            if (result == 1)
                            {
                                tempScores[0]++;
                            }
                            else if (result == -1)
                            {
                                tempScores[1]++;
                            }

                        }
                        return tempScores;




                    }
                    else
                    {
                        return null;//no score if no history
                    }
                }
            }
            */

            private int[] playHistoryPlayeridx
            {
                //enables deduction by ensuring known info is readable
                get
                {
                    if (playHistoryBoth == null)
                    {
                        return null;

                    }
                    else
                    {
                        //create an array with length playhistory both
                        int[] playIdHist = new int[playHistoryBoth.Length];
                        for (int i = 0; i < playHistoryBoth.Length; i++)
                        {
                            if (_p1Hand.Contains((Card)playHistoryBoth[i]))
                            {
                                playIdHist[i] = 0;
                            }
                            else if (_p2Hand.Contains((Card)playHistoryBoth[i]))
                            {
                                playIdHist[i] = 1;
                            }
                            else//failcase where card in reduced history did not start in either players hand
                            {
                                Console.WriteLine("card in reduced history did not start in either players hand");
                                break;
                            }
                        }
                        return playIdHist;
                    }
                }
            }

            private int[] GetTricksWon()
            {
                int[] tricks = new int[2] { 0, 0 };

                if (playHistoryBoth != null && playHistoryBoth.Length >= 2)
                {
                    int numTricks = playHistoryBoth.Length / 2;
                    for (int i = 0; i < numTricks; i++)
                    {
                        var p1Card = (Card)playHistoryBoth[2 * i];     // Player 1's card in trick i
                        var p2Card = (Card)playHistoryBoth[2 * i + 1]; // Player 2's card in trick i
                        if (p1Card.CompareTo(p2Card) > 0)
                            tricks[0]++;
                        else
                            tricks[1]++;
                    }
                }

                return tricks;
            }


            //used by cfr to determine if the game is over, needs reworked
            public override double[] TerminalValues
            {
                get
                {
                    if (_actions.Length == 12) // Game ends after 12 actions (2 bids + 10 plays)
                    {
                        int[] tricks = GetTricksWon(); // Raw tricks won
                        int bidP1 = (int)_actions[0] - 13; // Bid0=0, Bid1=1, etc.
                        int bidP2 = (int)_actions[1] - 13;

                        // Apply bonuses ONLY if tricks match bid at the end
                        if (tricks[0] == bidP1) tricks[0] += 10;
                        if (tricks[1] == bidP2) tricks[1] += 10;

                        // Determine winner
                        if (tricks[0] > tricks[1])
                            return new double[] { 1, -1 }; // P1 wins
                        else if (tricks[0] < tricks[1])
                            return new double[] { -1, 1 };  // P2 wins
                        else
                            return new double[] { 0, 0 };    // Tie
                    }
                    return null; // Game ongoing
                }
            }
            //potential issues with encoder as number of possible actions here is 6.
            private Action[] bids
            {
                get
                {
                    return new Action[] { Action.Bid0, Action.Bid1, Action.Bid2, Action.Bid3, Action.Bid4, Action.Bid5 }; //unrestricted bidding
                }
            }
            public override Action[] LegalActions
            {
                get
                {
                    if (_actions.Length < 2) //bidding
                    {
                        return bids;
                    }
                    else if (_actions.Length >= 2 && _actions.Length < 4)
                    {//first turn post bidding, all 5 cards in hand for both players

                        switch (this.CurrentPlayerIdx)
                        {
                            case 0:
                                Action[] plays = new Action[_p1Hand.Length];
                                for (int i = 0; i < _p1Hand.Length; i++)
                                {
                                    plays[i] = (Action)_p1Hand[i];
                                }
                                return plays;

                            case 1:
                                Action[] plays2 = new Action[_p2Hand.Length];
                                for (int i = 0; i < _p2Hand.Length; i++)
                                {
                                    plays2[i] = (Action)_p2Hand[i];
                                }
                                return plays2;
                            default:
                                return null;
                        }
                    }
                    else
                    {

                        //using trick history, give the original hand with the cards they've already played removed
                        switch (this.CurrentPlayerIdx)
                        {
                            case 0:
                                int len1 = _p1Hand.Length;
                                Action[] handAsActions1 = new Action[len1];
                                for (int i = 0; i < len1; i++)
                                {
                                    handAsActions1[i] = (Action)_p1Hand[i];
                                }

                                Action[] updatedActions = handAsActions1.Except(playHistoryP1).ToArray();
                                Array.Sort(updatedActions);
                                return updatedActions;
                            case 1:
                                int len2 = _p2Hand.Length;
                                Action[] handAsActions2 = new Action[len2];
                                for (int i = 0; i < len2; i++)
                                {
                                    handAsActions2[i] = (Action)_p2Hand[i];
                                }

                                Action[] updatedActions2 = handAsActions2.Except(playHistoryP2).ToArray();
                                Array.Sort(updatedActions2);
                                return updatedActions2;
                            default:
                                return null;
                        }
                    }
                }
            }
            /*public override bool checkTrivial
            {
                get
                {
                    //if odd length of _actions,check through the legal actions against the last action  in _actions
                    //this triviality only works for player2 as only they have enough information
                    if (_actions.Length % 2 != 0)
                    {
                        //target is the action you are responding to
                        Action target = _actions[_actions.Length - 1];
                        int[] results = new int[LegalActions.Length];
                        for (int i = 0; i < LegalActions.Length; i++)
                        {
                            results[i] = LegalActions[i].CompareTo(target);
                        }
                        if (results.Max() == results.Min())//if all results are equal then it is a trivial gamestate(where all options in hand all lose or all win so play lowest card to save higher cards for later
                        {
                            return true;
                        }
                        else
                        {
                            return false;
                        }
                    }
                    else
                    {
                        return false;
                    }
                }
            }*/

            //accurate game state tracking
            public override GameState<Action> AddAction(Action action)
            {
                var actions = _actions.Concat(Enumerable.Repeat(action, 1)).ToArray();
                return new WhistState(_cards, actions);
            }

            private int forcedLossCounter
            {
                get
                {
                    int Lcounter = 0;
                    string handDisplay = "";
                    switch (CurrentPlayerIdx)
                    {
                        case 0: // Player 1
                                // Find runs and calculate guaranteed losses from them
                            handDisplay = "P1 Hand: " + string.Join(",", _p1Hand);
                            for (int startIdx = 0; startIdx < _p1Hand.Length; startIdx++)
                            {
                                // Skip if this card has already been counted as part of a run
                                if (startIdx > 0 && _p1Hand[startIdx] == _p1Hand[startIdx - 1] + 1)
                                    continue;

                                // Find the length of the run starting at this index
                                int runLength = 1;
                                for (int j = startIdx; j < _p1Hand.Length - 1; j++)
                                {
                                    if (_p1Hand[j + 1] == _p1Hand[j] + 1)
                                        runLength++;
                                    else
                                        break;
                                }

                                if (runLength > 1) // Only consider actual runs
                                {
                                    Card lowestInRun = _p1Hand[startIdx];

                                    // Special handling for Two in a run
                                    if (lowestInRun == Card.Two)
                                    {
                                        // The Two itself guarantees one loss
                                        Lcounter++; // Count the Two loss

                                        // For runs starting with Two, we compare the run length (minus the Two) 
                                        // with cards below the second-lowest card in the run
                                        if (runLength > 1)
                                        {
                                            Card secondLowest = _p1Hand[startIdx + 1]; // Second lowest in run
                                            int cardsBelowSecondLowest = (int)secondLowest - 2; // Cards between Two and second lowest

                                            // Calculate additional forced losses from the rest of the run
                                            int additionalForcedLosses = Math.Max(0, (runLength - 1) - cardsBelowSecondLowest);
                                            Lcounter += additionalForcedLosses;
                                        }
                                    }
                                    else // Normal case - run doesn't include Two
                                    {
                                        int cardsBelowLowest = (int)lowestInRun - 1; // Cards from Two up to lowestInRun-1

                                        // Calculate forced losses from this run
                                        int forcedLossesFromRun = Math.Max(0, runLength - cardsBelowLowest);
                                        Lcounter += forcedLossesFromRun;
                                    }
                                }
                                else if (runLength == 1 && _p1Hand[startIdx] == Card.Two)
                                {
                                    // Handle the case where Two isn't part of a run
                                    Lcounter++;
                                }
                            }
                            return Lcounter;

                        case 1: // Player 2 
                                // Find runs and calculate guaranteed losses
                            handDisplay = "P2 Hand: " + string.Join(",", _p2Hand);
                            for (int startIdx = 0; startIdx < _p2Hand.Length; startIdx++)
                            {
                                // Skip if this card has already been counted as part of a run
                                if (startIdx > 0 && _p2Hand[startIdx] == _p2Hand[startIdx - 1] + 1)
                                    continue;

                                // Find the length of the run starting at this index
                                int runLength = 1;
                                for (int j = startIdx; j < _p2Hand.Length - 1; j++)
                                {
                                    if (_p2Hand[j + 1] == _p2Hand[j] + 1)
                                        runLength++;
                                    else
                                        break;
                                }

                                if (runLength > 1) // Only consider actual runs
                                {
                                    Card lowestInRun = _p2Hand[startIdx];

                                    // Special handling for Two in a run
                                    if (lowestInRun == Card.Two)
                                    {
                                        // The Two itself guarantees one loss
                                        Lcounter++; // Count the Two loss

                                        // For runs starting with Two, we compare the run length (minus the Two) 
                                        // with cards below the second-lowest card in the run
                                        if (runLength > 1)
                                        {
                                            Card secondLowest = _p2Hand[startIdx + 1]; // Second lowest in run
                                            int cardsBelowSecondLowest = (int)secondLowest - 2; // Cards between Two and second lowest

                                            // Calculate additional forced losses from the rest of the run
                                            int additionalForcedLosses = Math.Max(0, (runLength - 1) - cardsBelowSecondLowest);
                                            Lcounter += additionalForcedLosses;
                                        }
                                    }
                                    else // Normal case - run doesn't include Two
                                    {
                                        int cardsBelowLowest = (int)lowestInRun - 1; // Cards from Two up to lowestInRun-1

                                        // Calculate forced losses from this run
                                        int forcedLossesFromRun = Math.Max(0, runLength - cardsBelowLowest);
                                        Lcounter += forcedLossesFromRun;
                                    }
                                }
                                else if (runLength == 1 && _p2Hand[startIdx] == Card.Two)
                                {
                                    // Handle the case where Two isn't part of a run
                                    Lcounter++;
                                }
                            }
                            Console.WriteLine($"{handDisplay} | Forced Losses: {Lcounter}");
                            return Lcounter;

                        default:
                            return 0;
                    }
                }
            }
            private int forcedWinCounter
            {
                get
                {
                    int Wcounter = 0;
                    string handDisplay = "";
                    switch (CurrentPlayerIdx)
                    {
                        case 0: // Player 1
                                // Find runs and calculate guaranteed wins from them
                            handDisplay = "P1 Hand: " + string.Join(",", _p1Hand);
                            for (int startIdx = 0; startIdx < _p1Hand.Length; startIdx++)
                            {
                                // Skip if this card has already been counted as part of a run
                                if (startIdx > 0 && _p1Hand[startIdx] == _p1Hand[startIdx - 1] + 1)
                                    continue;

                                // Find the length of the run starting at this index
                                int runLength = 1;
                                for (int j = startIdx; j < _p1Hand.Length - 1; j++)
                                {
                                    if (_p1Hand[j + 1] == _p1Hand[j] + 1)
                                        runLength++;
                                    else
                                        break;
                                }

                                if (runLength > 1) // Only consider actual runs
                                {
                                    Card highestInRun = _p1Hand[startIdx + runLength - 1];

                                    // Special handling for Ace in a run
                                    if (highestInRun == Card.Ace)
                                    {
                                        // The Ace itself guarantees one win, the rest of the run may create additional forced wins
                                        Wcounter++; // Count the Ace win

                                        // For runs ending with Ace, we compare the run length (minus the Ace) 
                                        // with cards above the second-highest card in the run
                                        if (runLength > 1)
                                        {
                                            Card secondHighest = _p1Hand[startIdx + runLength - 2]; // Second highest in run
                                            int cardsAboveSecondHighest = 0;

                                            // Count cards between second highest and Ace
                                            for (Card c = secondHighest + 1; c < Card.Ace; c++)
                                            {
                                                if (!_p1Hand.Contains(c))
                                                    cardsAboveSecondHighest++;
                                            }

                                            // Calculate additional forced wins from the rest of the run
                                            int additionalForcedWins = Math.Max(0, (runLength - 1) - cardsAboveSecondHighest);
                                            Wcounter += additionalForcedWins;
                                        }
                                    }
                                    else // Normal case - run doesn't include Ace
                                    {
                                        int cardsAboveHighest = 0;

                                        // Count cards above the highest card in this run
                                        for (Card c = highestInRun + 1; c <= Card.Ace; c++)
                                        {
                                            if (!_p1Hand.Contains(c))
                                                cardsAboveHighest++;
                                        }

                                        // Calculate forced wins from this run
                                        int forcedWinsFromRun = Math.Max(0, runLength - cardsAboveHighest);
                                        Wcounter += forcedWinsFromRun;
                                    }
                                }
                                else if (runLength == 1 && _p1Hand[startIdx] == Card.Ace)
                                {
                                    // Handle the case where Ace isn't part of a run
                                    Wcounter++;
                                }
                            }
                            return Wcounter;

                        case 1: // Player 2 - similar logic with the same special handling for Ace
                            handDisplay = "P2 Hand: " + string.Join(",", _p2Hand);
                            for (int startIdx = 0; startIdx < _p2Hand.Length; startIdx++)
                            {
                                // Skip if this card has already been counted as part of a run
                                if (startIdx > 0 && _p2Hand[startIdx] == _p2Hand[startIdx - 1] + 1)
                                    continue;

                                // Find the length of the run starting at this index
                                int runLength = 1;
                                for (int j = startIdx; j < _p2Hand.Length - 1; j++)
                                {
                                    if (_p2Hand[j + 1] == _p2Hand[j] + 1)
                                        runLength++;
                                    else
                                        break;
                                }

                                if (runLength > 1) // Only consider actual runs
                                {
                                    Card highestInRun = _p2Hand[startIdx + runLength - 1];

                                    // Special handling for Ace in a run
                                    if (highestInRun == Card.Ace)
                                    {
                                        // The Ace itself guarantees one win, the rest of the run may create additional forced wins
                                        Wcounter++; // Count the Ace win

                                        // For runs ending with Ace, we compare the run length (minus the Ace) 
                                        // with cards above the second-highest card in the run
                                        if (runLength > 1)
                                        {
                                            Card secondHighest = _p2Hand[startIdx + runLength - 2]; // Second highest in run
                                            int cardsAboveSecondHighest = 0;

                                            // Count cards between second highest and Ace
                                            for (Card c = secondHighest + 1; c < Card.Ace; c++)
                                            {
                                                if (!_p2Hand.Contains(c))
                                                    cardsAboveSecondHighest++;
                                            }

                                            // Calculate additional forced wins from the rest of the run
                                            int additionalForcedWins = Math.Max(0, (runLength - 1) - cardsAboveSecondHighest);
                                            Wcounter += additionalForcedWins;
                                        }
                                    }
                                    else // Normal case - run doesn't include Ace
                                    {
                                        int cardsAboveHighest = 0;

                                        // Count cards above the highest card in this run
                                        for (Card c = highestInRun + 1; c <= Card.Ace; c++)
                                        {
                                            if (!_p2Hand.Contains(c))
                                                cardsAboveHighest++;
                                        }

                                        // Calculate forced wins from this run
                                        int forcedWinsFromRun = Math.Max(0, runLength - cardsAboveHighest);
                                        Wcounter += forcedWinsFromRun;
                                    }
                                }
                                else if (runLength == 1 && _p2Hand[startIdx] == Card.Ace)
                                {
                                    // Handle the case where Ace isn't part of a run
                                    Wcounter++;
                                }
                            }
                            Console.WriteLine($"{handDisplay} | Forced Wins: {Wcounter}");
                            return Wcounter;

                        default:
                            return 0;
                    }
                }
            }

            public override int[] FilterLegalActions
            {
                get
                {

                    switch (CurrentPlayerIdx)
                    {
                        case 0:

                            //check if bidding phase p1
                            if (_actions.Length == 0)
                            {

                                int[] filterArray = new int[LegalActions.Length];

                                // Count guaranteed wins/losses
                                int guaranteedWins = forcedWinCounter;
                                int guaranteedLosses = forcedLossCounter;

                                // Mark impossible bids
                                for (int i = 0; i < filterArray.Length; i++)
                                {
                                    int bid = i; // Bid0 = 0, Bid1 = 1, etc.

                                    // Filter bids that are too low given guaranteed wins
                                    if (bid < guaranteedWins)
                                    {
                                        filterArray[i] = 1;
                                    }

                                    // Filter bids that are too high given guaranteed losses
                                    if (bid > 5 - guaranteedLosses)
                                    {
                                        filterArray[i] = 1;
                                    }
                                }

                                return filterArray;
                            }
                            return null; // No filtering needed


                        case 1:
                            if (_actions.Length == 1)
                            {
                                int otherBid = (int)_actions[0] - 13;
                                int[] filterArray = new int[LegalActions.Length]; // Should be 6 for bids
                                int[] legalArray = new int[LegalActions.Length];
                                // First handle the impossible total bid constraint
                                for (int i = 0; i < 6; i++)
                                {
                                    if (i + otherBid == 5)
                                    {
                                        filterArray[i] = 1;
                                        legalArray[i] = 1;//copy used for reversion in rare cases
                                        break;
                                    }
                                }
                                // Then handle forced losses/wins if they exist
                                if (forcedLossCounter > 0)
                                {
                                    // Mark higher bids as illegal based on number of forced losses
                                    //this is a hack to reduce wasted compute on non-starter bids
                                    for (int i = 5; i > 5 - forcedLossCounter; i--)
                                    {
                                        filterArray[i] = 1;
                                    }
                                }
                                if (forcedWinCounter > 0)
                                {
                                    // Mark lower bids as illegal based on number of forced wins
                                    //this is a hack to reduce wasted compute on non-starter bids
                                    for (int i = 0; i < forcedWinCounter; i++)
                                    {
                                        filterArray[i] = 1;
                                    }
                                }
                                //check to make sure not all bids have been marked illegal as p2 still needs to play
                                bool hasMove = false;
                                for (int i = 0; i < filterArray.Length; i++)
                                {
                                    if (filterArray[i] == 0)
                                    {
                                        hasMove = true;
                                        break;
                                    }
                                }
                                if (hasMove == false)
                                {
                                    //rare position that optimising(+other bid) has marked all moves all illegal
                                    //return legal moves pre optimisation
                                    return legalArray;

                                }
                                return filterArray;
                            }
                            return null; // No filtering needed
                        default:
                            return null;//never reached


                    }


                    //return an array with the actions to be filtered marked
                    //if nothing to be removed then return an array of 0s with length equal to those actions









                    /*
                    if (_actions.Length % 2 != 0 && LegalActions.Length > 2)
                    {
                        //target is the action you are responding to
                        Action target = _actions[_actions.Length - 1];
                        int[] results = new int[LegalActions.Length];
                        for (int i = 0; i < LegalActions.Length; i++)
                        {
                            results[i] = LegalActions[i].CompareTo(target);
                        }
                        //custom logic for 3 card hand whist
                        if (results[0] == -1 && results[1] == -1 && results[2] == 1)//2 losing options therefore no point playing out both
                        {
                            //return 1,1 to indicate yes there is a obvious bad play, and to skip the middle card in the hand
                            return new int[] { 1, 1 };
                        }
                        else if (results[0] == -1 && results[1] == 1 && results[1] == 1)
                        {
                            //return 1,2 to indicate yes there is overkill, and to skip the highest card in the hand
                            return new int[] { 1, 2 };
                        }
                        else
                        {
                            return new int[] { 0, 0 };
                        }
                    }
                    else
                    {
                        return new int[] { 0, 0 };
                    }*/
                }
            }
        }
        /// Shuffles the given array in place.
        /// From http://rosettacode.org/wiki/Knuth_shuffle#C.23
        public static T[] Shuffle<T>(Random rng, T[] array)
        {
            for (int i = 0; i < array.Length; i++)
            {
                int j = rng.Next(i, array.Length); // Don't select from the entire array on subsequent loops
                T temp = array[i]; array[i] = array[j]; array[j] = temp;
            }
            return array;
        }
    }
}
