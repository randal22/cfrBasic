using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using Microsoft.FSharp.Collections;
using Microsoft.FSharp.Core;
//reimplementation into c#(2024), and adapted to simplified whist from kuhn poker, original code from https://github.com/brianberns/Cfrm  
//additional saving logic added for different file format
//added multicore functionality to drastically increase data generation speed
namespace Cfrm.Test.CS
{
    using static Cfrm.Test.Whist;
    using static Whist.Card;

    class Program
    {



        
        static void Main(string[] args)
        {
            bool multiThread = true;
            var numIterations = 1000000;
            int progressInterval = 10000;
            if (!multiThread)
            {
                var deck = new Card[] { Card.Two, Card.Three, Card.Four, Card.Five, Card.Six, Card.Seven, Card.Eight, Card.Nine, Card.Ten, Card.Jack, Card.Queen, Card.King, Card.Ace };

                var rng = new Random(Guid.NewGuid().GetHashCode());


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
                string cPath = "StrategyTest.csv";
                // print results
                Console.WriteLine("Expected game values:");
                Console.WriteLine(string.Join(", ", expectedGameValues));
                Console.WriteLine(expectedGameValues.GetType());
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

                Console.WriteLine("Strategy profile written to " + cPath);
            }
            else
            {


                int numCores = Environment.ProcessorCount;
                int baseIterationsPerCore = numIterations / numCores;
                int remainingIterations = numIterations % numCores;

                var tasks = new Task<(double[], StrategyProfile)>[numCores];

                for (int i = 0; i < numCores; i++)
                {
                    int coreIndex = i;
                    int iterationsForThisCore = baseIterationsPerCore + (i < remainingIterations ? 1 : 0);

                    tasks[i] = Task.Run(() =>
                    {
                        var rng = new Random(Guid.NewGuid().GetHashCode());
                        var deck = new Card[] { Card.Two, Card.Three, Card.Four, Card.Five, Card.Six, Card.Seven, Card.Eight, Card.Nine, Card.Ten, Card.Jack, Card.Queen, Card.King, Card.Ace };

                        Stopwatch stopwatch = new Stopwatch();
                        stopwatch.Start();

                        var (expectedGameValues, strategyProfile) =
                            CounterFactualRegret.Minimize(iterationsForThisCore, 2, iter =>
                            {
                                var cards = Shuffle(rng, deck);
                                if (iter % progressInterval == 0)
                                {
                                    Console.WriteLine($"Core {coreIndex}: Iteration {iter}/{iterationsForThisCore}");
                                    stopwatch.Stop();
                                    TimeSpan elapsed = stopwatch.Elapsed;
                                    Console.WriteLine($"Core {coreIndex}: Stage {iter} took {elapsed.Minutes} minutes and {elapsed.Seconds}.{elapsed.Milliseconds:D3} seconds");
                                    stopwatch.Reset();
                                    stopwatch.Start();
                                }
                                return new WhistState(cards);
                            });

                        return (expectedGameValues, strategyProfile);
                    });
                }

                Task.WaitAll(tasks);

                // Merge results from all threads
                const double epsilon = 1e-6; // Small value to account for floating-point precision

                bool IsDefaultStrategy(double[] strategy)
                {
                    if (strategy.Length <= 1) return false;
                    double firstValue = strategy[0];
                    return strategy.All(v => Math.Abs(v - firstValue) < epsilon);
                }

                // Merge results from all threads
                var finalExpectedGameValues = new DenseVector(2);
                var finalStrategyDictionary = new Dictionary<string, double[]>();

                foreach (var task in tasks)
                {
                    var (expectedGameValues, strategyProfile) = task.Result;

                    // Sum up expected game values
                    finalExpectedGameValues += new DenseVector(expectedGameValues);

                    // Merge strategy profiles
                    foreach (var (key, strategy) in strategyProfile.ToDict())
                    {
                        if (!finalStrategyDictionary.ContainsKey(key))
                        {
                            finalStrategyDictionary[key] = strategy;
                        }
                        else
                        {
                            var existingStrategy = finalStrategyDictionary[key];

                            if (IsDefaultStrategy(existingStrategy) && !IsDefaultStrategy(strategy))
                            {
                                finalStrategyDictionary[key] = strategy;
                            }
                            else if (!IsDefaultStrategy(existingStrategy) && IsDefaultStrategy(strategy))
                            {
                                // Keep the existing strategy (do nothing)
                            }
                            else if (!IsDefaultStrategy(existingStrategy) && !IsDefaultStrategy(strategy))
                            {
                                // Average the strategies
                                finalStrategyDictionary[key] = existingStrategy.Zip(strategy, (a, b) => (a + b) / 2).ToArray();
                            }
                            // If both are default strategies, keep the existing one (do nothing)
                        }
                    }
                }

                // Normalize the expected game values
                finalExpectedGameValues /= numCores;

                // Convert C# Dictionary to F# Map
                var finalStrategyMap = MapModule.OfSeq<string, double[]>(
                finalStrategyDictionary.Select(kvp =>
                new Tuple<string, double[]>(kvp.Key, kvp.Value)));

                // Create the final StrategyProfile object
                var finalStrategyProfileObject = new StrategyProfile(finalStrategyMap);

                // Save the merged strategy profile
                const string path = "WhistMulti.strategy";
                finalStrategyProfileObject.Save(path);

                // Output results
                Console.WriteLine("Expected game values:");
                Console.WriteLine(string.Join(", ", finalExpectedGameValues));

                // Save to CSV
                string cPath = "StrategyMultiTest.csv";
                using (StreamWriter sw = new StreamWriter(cPath))
                {
                    sw.WriteLine("Key,Values");
                    foreach (var kvp in finalStrategyDictionary)
                    {
                        string valuesString = string.Join(",", kvp.Value.Select(d => d.ToString()));
                        sw.WriteLine($"{kvp.Key},{valuesString}");
                    }
                }

                Console.WriteLine("Strategy profile written to " + cPath);



            }
        }
    }
}









