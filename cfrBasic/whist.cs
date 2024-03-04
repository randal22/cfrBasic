using System;
using System.Collections.Generic;
using System.Linq;
//using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Cfrm.Test
{
    // [TestClass]
    public class Whist
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

        public class WhistState
            : GameState<Action>
        {
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

            public override int CurrentPlayerIdx =>
                _actions.Length % 2;

            public override string Key =>
                $"{_cards[this.CurrentPlayerIdx].ToString()[0]}{this.ActionString}";

            public override double[] TerminalValues
            {
                get
                {
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

            public override Action[] LegalActions { get; } =
                new Action[] { Action.Check, Action.Bet };

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

        //[TestMethod]
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
                    return new WhistState(cards);
                });

            const string path = "Whist.strategy";
            strategyProfile.Save(path);
            strategyProfile = StrategyProfile.Load(path);

            // https://en.wikipedia.org/wiki/Kuhn_poker#Optimal_strategy
            var dict = strategyProfile.ToDict();
            //Assert.AreEqual(expectedGameValues[0], -1.0 / 18.0, delta);
            var alpha = dict["J"][1];
            //Assert.IsTrue(alpha >= 0.0);
           // Assert.IsTrue(alpha <= 1.0 / 3.0);
            //Assert.AreEqual(dict["Q"][0], 1.0, delta);
            //Assert.AreEqual(dict["Qcb"][1], alpha + 1.0 / 3.0, delta);
            //Assert.AreEqual(dict["K"][1], 3.0 * alpha, delta);
        }
    }
}



/*
//deck of cards, needs to be shuffled. 
// 4 suits, 13 cards each suit.


using System;
using System.Collections.Generic;
using System.Globalization;




public class Deck
{
    //string[] suits = { "♣", "♦", "♥", "♠" };
    public string[] cards = {
    "Ace ♥", "2 ♥", "3 ♥", "4 ♥", "5 ♥", "6 ♥", "7 ♥", "8 ♥", "9 ♥", "10 ♥", "Jack ♥", "Queen ♥", "King ♥",
    "Ace ♦", "2 ♦", "3 ♦", "4 ♦", "5 ♦", "6 ♦", "7 ♦", "8 ♦", "9 ♦", "10 ♦", "Jack ♦", "Queen ♦", "King ♦",
    "Ace ♣", "2 ♣", "3 ♣", "4 ♣", "5 ♣", "6 ♣", "7 ♣", "8 ♣", "9 ♣", "10 ♣", "Jack ♣", "Queen ♣", "King ♣",
    "Ace ♠", "2 ♠", "3 ♠", "4 ♠", "5 ♠", "6 ♠", "7 ♠", "8 ♠", "9 ♠", "10 ♠", "Jack ♠", "Queen ♠", "King ♠"};
    public int[] vals = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52 };


}
public class Hand
{

    public string[] cards = { "empty" };
    public int[] vals = { -1 };

}


class Whist
{



    private static Random rand = new Random();

    //Fisher-Yates shuffle
    static Deck Shuffle(Deck deck)
    {
        

        int remaining = 52;

        while (remaining > 1)
        {
            //random swaps till we're through the whole deck
            remaining--;
            int a = rand.Next(remaining + 1);
            //a is the target for the swap
            string tempStr = deck.cards[a];//store value for swap
            int tempInt = deck.vals[a];
            //str swap
            deck.cards[a] = deck.cards[remaining];
            deck.cards[remaining] = tempStr;
            //int swap
            deck.vals[a] = deck.vals[remaining];
            deck.vals[remaining] = tempInt;
            //swap complete, rinse and repeat
        }

        //return shuffled deck
        return deck;
    }

    static void PlayGameThru(Deck rawDeck, int runs)
    {
        //track hands
        Hand p1Hand = new Hand();
        Hand p2Hand = new Hand();
        Hand p3Hand = new Hand();
        Hand p4Hand = new Hand();
        
        for (int i = 0; i < runs; i++)
        {
            //start of each game deck is shuffled and dealt
            //shuffle
            rawDeck = Shuffle(rawDeck);


            //deal
            for (int j = 0; j < 13; j++)
            {
                p1Hand.vals[j] = rawDeck.vals[j];
                p1Hand.cards[j] = rawDeck.cards[j];

                p2Hand.vals[j] = rawDeck.vals[j+13];
                p2Hand.cards[j] = rawDeck.cards[+13];

                p3Hand.vals[j] = rawDeck.vals[j + 26];
                p3Hand.cards[j] = rawDeck.cards[j+26];

                p4Hand.vals[j] = rawDeck.vals[j + 39];
                p4Hand.cards[j] = rawDeck.cards[j + 39];
            }
            //playing the game through, players will take sequential turns until all hands are empty, and a winner for that round is declared.
            //cfr will be used for the turn taking, so the information accesible to the current player will need to be passed. 
            //as this is a cfr framework, information needs to be formatted carefully to allow for other card game implementations to also use cfr. 
            //list of player available information that cfr will need to take a turn: gamestate from player's pov (trump, local hand, expended cards (whole table), player scores?,current player), legal moves.  
            Cfr cfr = new Cfr();

        }
    }

    static void Main(string[] args)
    {

        Deck baseDeck = new Deck();


        //baseDeck = Shuffle(baseDeck);
        Console.WriteLine("Enter number of games to be played");
        int target = Convert.ToInt32(Console.ReadLine());

        PlayGameThru(baseDeck, target);
    }
}
*/
