using System;
using System.Collections.Generic;

//using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Cfrm.Test.CS
{
    using static Cfrm.Test.Whist;
    using static Whist.Card;

    class Program
    {
        static void Main(string[] args)
        {
            //there are 1235520 possible different distributions of the cards in the hands of the two players with this deck
            var deck = new Card[] { Card.Two, Card.Three, Card.Four, Card.Five, Card.Six, Card.Seven, Card.Eight, Card.Nine, Card.Ten, Card.Jack, Card.Queen, Card.King, Card.Ace };
            //var deck = new Card[] {  Card.Jack, Card.Queen, Card.King};
            var rng = new Random(Guid.NewGuid().GetHashCode());
            //var rng = new Random(0);
            var numIterations = 1000000;
            //var delta = 0.03;
            int progressInterval = 100000;
            var (expectedGameValues, strategyProfile) =
                CounterFactualRegret.Minimize(numIterations, 2, i =>
                {
                    var cards = Shuffle(rng, deck);
                    if (i % progressInterval == 0)
                    {
                        Console.WriteLine("Current iteration: " + i);
                    }
                    return new WhistState(cards);
                });

            const string path = "Whist.strategy";
            strategyProfile.Save(path);
            strategyProfile = StrategyProfile.Load(path);


            var dict = strategyProfile.ToDict();
            string cPath = "Strategy.csv";
            // print results
            Console.WriteLine("Expected game values:");
            Console.WriteLine(string.Join(", ", expectedGameValues));

            Console.WriteLine("Strategy profile:");
            // Open a stream for writing
            using (StreamWriter sw = new StreamWriter(cPath))
            {
                // Write header
                sw.WriteLine("Key,Values");

                // Write each key-value pair
                foreach (var kvp in dict)
                {
                    string valuesString = string.Join(",", kvp.Value.Select(d => d.ToString())); // Flatten the double array
                    sw.WriteLine($"{kvp.Key},{valuesString}");
                    //Console.WriteLine($"{kvp.Key}: {string.Join(", ", valuesString)}");
                }
            }



            //Console.WriteLine($"{key}: {string.Join(", ", value)}");
            //write to file


            Console.WriteLine("Strategy profile written to Strategy.csv");
        }
    }
}









