using System;
using System.Collections.Generic;
using System.Linq;
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
            
            Ten,
            Jack,
            Queen,
            King,
            Ace
        }
        //possible actions
        public enum Action
        {
            Check,
            Bet
            //F       ollowWHighCard,
            //FollowWLowCard,
            //Trump,
            //Throw,
            //LeadWHighCard,
            //LeadWLowCard
            //playHighestLegalCard,
            //playLowestLegalCard,

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
                //Assert.AreEqual(2, cards.Length);
                _cards = cards;
                _actions = actions;
            }
            private readonly Card[] _cards;
            private readonly Action[] _actions;

            //player trick counter
            //public int[] tricksWon = { 0, 0, 0, 0 };
            
            //action string is a shorthand version of the gamestate, which loses most of it's effectiveness given the 4 player noncyclical turn ordering, with more than 2 actions.
            private string ActionString
            {
                get
                {
                    var chars = _actions.Select(action =>action.ToString().ToLower()[0]).ToArray();
                    return new string(chars);
                }
            }
            //turn checking, needs changed
            public override int CurrentPlayerIdx =>_actions.Length % 2;

            //action string, used for gamestate tracking. less useful for whist, however all games end when action string is 13 characters long
            public override string Key =>$"{_cards[this.CurrentPlayerIdx].ToString()[0]}{this.ActionString}";

            public override double[] TerminalValues
            {
                get
                {

                    // if the action string is 52 characters long, then all tricks are played out as all actions have been taken (cards played). 


                    
                    int sign;
                    switch (this.ActionString)
                    {
                        case "cbc":   // player 1 wins ante only
                            return new double[] { -1, 1 };
                        case "bc":    // player 0 wins ante only
                            return new double[] { 1, -1 };
                        case "cc":    // no bets: high card wins ante only
                            sign = _cards[0].CompareTo(_cards[1]);
                            return new double[] { sign * 1, sign * -1 };
                        case "cbb":   // two bets: high card wins ante and bet
                            sign = _cards[1].CompareTo(_cards[0]);
                            return new double[] { sign * -2, sign * 2 };
                        case "bb":    // two bets: high card wins ante and bet
                            sign = _cards[0].CompareTo(_cards[1]);
                            return new double[] { sign * 2, sign * -2 };
                        default: return null;
                    }
                    
                    



                    
                }
            }

            public override Action[] LegalActions { get; } = new Action[] { Action.Check, Action.Bet };

            //accurate game state tracking
            public override GameState<Action> AddAction(Action action)
            {
                var actions = _actions
                    .Concat(Enumerable.Repeat(action, 1))
                    .ToArray();
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
