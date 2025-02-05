﻿using System;
using System.Collections.Generic;
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
            }
            //aquires the trick history as an array of actions, all of player 1's plays first then player 2s, split halfway


            private Action[] playHistoryP1
            {
                get
                {

                    //empty action history check
                    //skip first two actions bc bidding
                    Action[] reducedActionHist = new Action[_actions.Length - 2];
                    for (int i = 2; i < _actions.Length; i++)
                    {
                        reducedActionHist[i - 2] = _actions[i];
                    }
                    //look up redActHist for player1s starting hand
                    Action[] P1Hist = new Action[reducedActionHist.Length / 2];
                    int tempIndex = 0;
                    for (int i = 0; i < (reducedActionHist.Length); i++)
                    {
                        for (int j = 0; j < _p1Hand.Length; j++)
                        {
                            if (reducedActionHist[i] == (Action)_p1Hand[j])
                            {
                                P1Hist[tempIndex] = reducedActionHist[i];
                                tempIndex++;
                            }
                        }
                    }
                    return P1Hist;
                }
            }
            private Action[] playHistoryP2
            {
                get
                {
                    int gameLength = _actions.Length;
                    //empty action history check
                    //skip first two actions bc bidding
                    Action[] reducedActionHist = new Action[gameLength - 2];
                    for (int i = 2; i < gameLength; i++)
                    {
                        reducedActionHist[i - 2] = _actions[i];
                    }
                    //look up redActHist for player2s starting hand
                    Action[] P2Hist = new Action[reducedActionHist.Length / 2];
                    int tempIndex = 0;
                    for (int i = 0; i < (reducedActionHist.Length); i++)
                    {
                        for (int j = 0; j < _p2Hand.Length; j++)
                        {
                            if (reducedActionHist[i] == (Action)_p2Hand[j])
                            {
                                P2Hist[tempIndex] = reducedActionHist[i];
                                tempIndex++;
                            }
                        }
                    }
                    return P2Hist;
                }
            }


            private Action[] trickHistory
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


            //score calculation (best of 3)
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
                        return tempScores;*/




                    }
                    else
                    {
                        return null;//no score if no history
                    }
                }
            }
            //used by cfr to determine if the game is over, needs reworked
            public override double[] TerminalValues
            {
                get
                {
                    //check if game is over, if not return null
                    if (_actions.Length == 12)
                    {

                        if (scores[0] > scores[1])//p1 win
                        {
                            return new double[] { 1, -1 };
                        }
                        else if (scores[0] < scores[1])//p2 win
                        {
                            return new double[] { -1, 1 };
                        }
                        else if (scores[0] == scores[1])//tie
                        {
                            return new double[] { 0, 0 };
                        }
                        else
                        {
                            return null;
                        }
                    }
                    else
                    {
                        return null;
                    }
                }
            }
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
                        int otherBid = -1;

                        if (_actions.Length == 0)
                        {
                            return bids;
                        }
                        else
                        {
                            otherBid = (int)_actions[0] - 13;

                            // Allocate exact size
                            Action[] filteredBids = new Action[bids.Length - 1];
                            // Populate the filtered array
                            int index = 0;
                            for (int i = 0; i < 6; i++)
                            {
                                if (i + otherBid != 5)
                                {
                                    filteredBids[index] = bids[i];
                                    index++;
                                }
                            }

                            return filteredBids;
                        }
                    }
                    else if (_actions.Length > 2 && _actions.Length < 4)
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
            public override bool checkTrivial
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
            }
            //accurate game state tracking
            public override GameState<Action> AddAction(Action action)
            {
                var actions = _actions.Concat(Enumerable.Repeat(action, 1)).ToArray();
                return new WhistState(_cards, actions);
            }
            public override int[] FilterLegalActions
            {
                get
                {
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
                    }
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
