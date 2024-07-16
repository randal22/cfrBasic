using System;
using System.Collections.Generic;
using System.Diagnostics;
//reimplementation into c#(2024), and adapted to simplified whist from kuhn poker, original code from https://github.com/brianberns/Cfrm  
//additional saving logic added for different file format
namespace Cfrm.Test.CS
{
    using static Cfrm.Test.Whist;
    using static Whist.Card;

    class Program
    {
        static void Main(string[] args)
        {
            
            var deck = new Card[] { Card.Two, Card.Three, Card.Four, Card.Five, Card.Six, Card.Seven, Card.Eight, Card.Nine, Card.Ten, Card.Jack, Card.Queen, Card.King, Card.Ace };
            
            var rng = new Random(Guid.NewGuid().GetHashCode());
            
            var numIterations = 100000000;
            
            int progressInterval = 1000000;
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            var (expectedGameValues, strategyProfile) =
                CounterFactualRegret.Minimize(numIterations, 2, i =>
                {
                    var cards = Shuffle(rng, deck);
                    if (i % progressInterval == 0)
                    {
                        Console.WriteLine("Current iteration: " + i);
                        stopwatch.Stop();
                        TimeSpan elapsed = stopwatch.Elapsed;
                        Console.WriteLine($"Stage {i} took {elapsed.Minutes} minutes and {elapsed.Seconds}.{elapsed.Milliseconds:D3} seconds");
                        stopwatch.Reset();
                        stopwatch.Start();
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
                    
                }
            }

            //write to file

            Console.WriteLine("Strategy profile written to Strategy.csv");
        }
    }
}









