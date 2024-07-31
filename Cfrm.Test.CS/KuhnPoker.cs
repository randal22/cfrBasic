using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;


//reimplementation of the KuhnPoker class into c#, original code found at https://github.com/brianberns/Cfrm



namespace Cfrm.Test
{
    
    public class KuhnPoker
    {
        public enum Card
        {
            Jack,
            Queen,
            King
        }

        public enum Action
        {
            Check,
            Bet
        }

        public class KuhnPokerState
            : GameState<Action>
        {
            public KuhnPokerState(Card[] cards)
                : this(cards, new Action[0])
            {
            }

            private KuhnPokerState(Card[] cards, Action[] actions)
            {
                Assert.AreEqual(2, cards.Length);
                _cards = cards;
                _actions = actions;
            }
            private readonly Card[] _cards;
            private readonly Action[] _actions;

            private string ActionString
            {
                get
                {
                    var chars = _actions
                        .Select(action =>
                            action.ToString().ToLower()[0])
                        .ToArray();
                    return new string(chars);
                }
            }
            //identify the current player based on the number of actions (does not work for trick taking games)
            public override int CurrentPlayerIdx =>_actions.Length % 2;

            public override string Key =>$"{_cards[this.CurrentPlayerIdx].ToString()[0]}{this.ActionString}";

            //called every action to check if the game is over
            public override double[] TerminalValues
            {
                get
                {
                    //Console.WriteLine("is game over");
                    int sign;
                    switch (this.ActionString)
                    {
                        case "cbc":   // player 1 wins ante only
                            //Console.WriteLine("player 1 wins ante only");
                            return new double[] { -1, 1 };
                        case "bc":    // player 0 wins ante only
                            //Console.WriteLine("player 0 wins ante only");
                            return new double[] { 1, -1 };
                        case "cc":    // no bets: high card wins ante only
                            //Console.WriteLine("no bets: high card wins ante only");
                            sign = _cards[0].CompareTo(_cards[1]);
                            return new double[] { sign * 1, sign * -1 };
                        case "cbb":   // two bets: high card wins ante and bet
                            //Console.WriteLine("two bets: high card wins ante and bet");
                            sign = _cards[1].CompareTo(_cards[0]);
                            return new double[] { sign * -2, sign * 2 };
                        case "bb":    // two bets: high card wins ante and bet
                            //Console.WriteLine("two bets: high card wins ante and bet");
                            sign = _cards[0].CompareTo(_cards[1]);
                            return new double[] { sign * 2, sign * -2 };
                        default:
                            //Console.WriteLine("not over");
                            return null;
                    }
                }
            }

            public override Action[] LegalActions { get; } =new Action[] { Action.Check, Action.Bet };

            
            //advances game by taking the given action for the current player
            public override GameState<Action> AddAction(Action action)
            {
                var actions = _actions
                    .Concat(Enumerable.Repeat(action, 1))
                    .ToArray();
                //Console.WriteLine($"actions: {string.Join("", actions)}");
                return new KuhnPokerState(_cards, actions);
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


        //this method has been moved
        /*
        [TestMethod]
        public void Minimize()
        {
            
            var deck = new Card[] { Card.Jack, Card.Queen, Card.King };
            var rng = new Random(0);
            var numIterations = 100000;
            var delta = 0.03;

            var (expectedGameValues, strategyProfile) =
                CounterFactualRegret.Minimize(numIterations, 2, i =>
                {
                    var cards = Shuffle(rng, deck)[0..2];
                    return new KuhnPokerState(cards);
                });

            const string path = "Kuhn.strategy";
            strategyProfile.Save(path);
            strategyProfile = StrategyProfile.Load(path);

            // https://en.wikipedia.org/wiki/Kuhn_poker#Optimal_strategy
            var dict = strategyProfile.ToDict();
           // Assert.AreEqual(expectedGameValues[0], -1.0 / 18.0, delta);
            var alpha = dict["J"][1];
            //Assert.IsTrue(alpha >= 0.0);
            //Assert.IsTrue(alpha <= 1.0 / 3.0);
           // Assert.AreEqual(dict["Q"][0], 1.0, delta);
           // Assert.AreEqual(dict["Qcb"][1], alpha + 1.0 / 3.0, delta);
           // Assert.AreEqual(dict["K"][1], 3.0 * alpha, delta);
        }
        */
    }
}
