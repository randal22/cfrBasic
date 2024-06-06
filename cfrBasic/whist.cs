﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Metadata;
using static System.Runtime.InteropServices.JavaScript.JSType;
//using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Cfrm.Test
{





    public class Whist
    {
        public enum Card
        {
            //Ace_of_Hearts, Two_of_Hearts, Three_of_Hearts, Four_of_Hearts, Five_of_Hearts, Six_of_Hearts, Seven_of_Hearts, Eight_of_Hearts, Nine_of_Hearts, Ten_of_Hearts, Jack_of_Hearts, Queen_of_Hearts, King_of_Hearts,
            //Ace_of_Diamonds, Two_of_Diamonds, Three_of_Diamonds, Four_of_Diamonds, Five_of_Diamonds, Six_of_Diamonds, Seven_of_Diamonds, Eight_of_Diamonds, Nine_of_Diamonds, Ten_of_Diamonds, Jack_of_Diamonds, Queen_of_Diamonds, King_of_Diamonds,
            //Ace_of_Clubs, Two_of_Clubs, Three_of_Clubs, Four_of_Clubs, Five_of_Clubs, Six_of_Clubs, Seven_of_Clubs, Eight_of_Clubs, Nine_of_Clubs, Ten_of_Clubs, Jack_of_Clubs, Queen_of_Clubs, King_of_Clubs,
            //Ace_of_Spades, Two_of_Spades, Three_of_Spades, Four_of_Spades, Five_of_Spades, Six_of_Spades, Seven_of_Spades, Eight_of_Spades, Nine_of_Spades, Ten_of_Spades, Jack_of_Spades, Queen_of_Spades, King_of_Spades
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
            E,
            N,
            T,
            J,
            Q,
            K,
            A

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
                _p1Hand = new Card[] { cards[0], cards[2], cards[4] };
                Array.Sort(_p1Hand);
                _p2Hand = new Card[] { cards[1], cards[3], cards[5] };
                Array.Sort(_p2Hand);
                //no actions at start
                _actions = actions;

            }
            private readonly Card[] _cards;
            private readonly Action[] _actions;
            public Card[] _p1Hand;
            public Card[] _p2Hand;

            //action string is a shorthand version of the gamestate, which loses most of it's effectiveness given the 4 player noncyclical turn ordering, with more than 2 actions.
            private string ActionString
            {
                get
                {
                    var chars = _actions.Select(action => action.ToString()[0]).ToArray();

                    return new string(chars);
                }
            }

            //turn checking, needs changed
            public override int CurrentPlayerIdx => _actions.Length % 2;


            public override string Key => $"{_cards[this.CurrentPlayerIdx].ToString()[0]}{this.ActionString}";

            //aquires the trick history as an array of actions, all of player 1's plays first then player 2s, split halfway

            private Action[] trickHistory
            {
                get
                {
                    int gameLength = this.ActionString.Length;
                    //empty action string check

                    if (gameLength == 0 || gameLength == 1)
                    {
                        return null;
                    }

                    //get the odd/even action string values, and compute score, 

                    if (gameLength % 2 == 1)
                    {
                        gameLength--;
                    }

                    Action[] history = new Action[gameLength];
                    for (int i = 0; i < gameLength; i = i + 2)
                    {
                        history[i] = (Action)Enum.Parse(typeof(Action), (ActionString[i]).ToString());
                        history[i + 1] = (Action)Enum.Parse(typeof(Action), (ActionString[i + 1]).ToString());
                    }

                    return history;



                }
            }

            private double[] scores
            {
                get
                {
                    if (trickHistory != null)
                    {
                        int gameLength = this.ActionString.Length;
                        if (gameLength % 2 == 1)
                        {
                            gameLength--;
                        }
                        double[] tempScores = new double[2];
                        for (int i = 0; i < gameLength / 2; i++)
                        {
                            int result = trickHistory[i].CompareTo(trickHistory[i + gameLength / 2]);
                            tempScores[0] = tempScores[0] + 1 * result;
                            tempScores[1] = tempScores[1] - 1 * result;

                        }
                        return tempScores;
                    }
                    else
                    {
                        return null;
                    }



                }
            }

            //private static int roundCounter = 0;
            

            public override double[] TerminalValues
            {
                get
                {
                    //update scores


                    //check if game is over, if not return null
                    if (ActionString.Length == 6)
                    {
                        //Console.WriteLine(ActionString);
                        return scores;
                    }
                    else
                    {
                        //Console.WriteLine("game not won");
                        return null;
                    }


                }
            }
            //this is where the action rework needs to come in, this is a lambda funtion run each turn. 
            //at this stage of the game its just the cards in hand

            public override Action[] LegalActions
            {
                //these need to be sorted into size order to provide meaningful data
                get
                {


                    //check if first turn
                    if (trickHistory == null)
                    {
                        switch (this.CurrentPlayerIdx)
                        {
                            case 0:
                                //Console.WriteLine("p1 first turn");
                                return new Action[] { (Action)_p1Hand[0], (Action)_p1Hand[1], (Action)_p1Hand[2] };

                            case 1:
                                //Console.WriteLine("p2 first turn");
                                return new Action[] { (Action)_p2Hand[0], (Action)_p2Hand[1], (Action)_p2Hand[2] };
                            default:
                                return null;
                        }
                    }
                    else
                    {
                        int actionsLen = 0;
                        if (trickHistory.Length == 2)
                        {
                            actionsLen = 2;
                            //Console.WriteLine("2nd turn");
                        }
                        else if (trickHistory.Length == 4)
                        {
                            actionsLen = 1;
                            //Console.WriteLine("third turn");
                        }
                        Action[] updatedActions = new Action[actionsLen];
                        //using trick history, give the origianl hand with the cards they've already played removed
                        switch (this.CurrentPlayerIdx)
                        {
                            case 0:

                                //get hand as an array of actions
                                Action[] handAsActions1 = new Action[] { (Action)_p1Hand[0], (Action)_p1Hand[1], (Action)_p1Hand[2] };
                                //find and remove any that are also found in the trickHistory
                                updatedActions = handAsActions1.Except(trickHistory).ToArray();
                                Array.Sort(updatedActions);

                                return updatedActions;

                            case 1:

                                //get hand as an array of actions
                                Action[] handAsActions2 = new Action[] { (Action)_p2Hand[0], (Action)_p2Hand[1], (Action)_p2Hand[2] };
                                //find and remove any that are also found in the trickHistory
                                updatedActions = handAsActions2.Except(trickHistory).ToArray();
                                Array.Sort(updatedActions);

                                return updatedActions;
                            default:
                                return null;
                        }




                    }








                }
            }


            //accurate game state tracking
            public override GameState<Action> AddAction(Action action)
            {
                var actions = _actions.Concat(Enumerable.Repeat(action, 1)).ToArray();
                return new WhistState(_cards, actions);
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
