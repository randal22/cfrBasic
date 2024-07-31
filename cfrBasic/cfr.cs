using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading.Tasks;
using Cfrm.SimplifiedWhist;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using Microsoft.FSharp.Collections;
using Microsoft.FSharp.Core;
// Much of this implementation follows the design of https://github.com/brianberns/Cfrm,
// applied to simplified-Whist instead of Kuhn Poker.

//saving logic added for different file format
//added multicore functionality to drastically increase data generation speed
//merging of strategy files added to allow for multiple runs to be combined into one file
namespace Cfrm.SimplifiedWhist
{
    using static Whist;
    using static Whist.Card;

    class Program
    {
        static Dictionary<string, double[]> ReadExistingCSV(string filePath)
        {
            var existingData = new Dictionary<string, double[]>();
            if (File.Exists(filePath))
            {
                using (var reader = new StreamReader(filePath))
                {
                    reader.ReadLine(); // Skip header
                    string line;
                    while ((line = reader.ReadLine()) != null)
                    {
                        var parts = line.Split(',');
                        var key = parts[0];
                        var values = parts.Skip(1).Select(double.Parse).ToArray();
                        existingData[key] = values;
                    }
                }
            }
            return existingData;
        }
        static Dictionary<string, double[]> MergeStrategies(Dictionary<string, double[]> existingData, Dictionary<string, double[]> newData)
        {
            var mergedData = new Dictionary<string, double[]>(existingData);
            const double epsilon = 1e-6;

            bool IsDefaultStrategy(double[] strategy)
            {
                if (strategy.Length <= 1) return false;
                double firstValue = strategy[0];
                return strategy.All(v => Math.Abs(v - firstValue) < epsilon);
            }

            foreach (var (key, newStrategy) in newData)
            {
                if (!mergedData.TryGetValue(key, out var existingStrategy))
                {
                    mergedData[key] = newStrategy;
                }
                else
                {
                    if (IsDefaultStrategy(existingStrategy) && !IsDefaultStrategy(newStrategy))
                    {
                        mergedData[key] = newStrategy;
                    }
                    else if (!IsDefaultStrategy(existingStrategy) && IsDefaultStrategy(newStrategy))
                    {
                        // Keep the existing strategy (do nothing)
                    }
                    else if (!IsDefaultStrategy(existingStrategy) && !IsDefaultStrategy(newStrategy))
                    {
                        // Average the strategies
                        mergedData[key] = existingStrategy.Zip(newStrategy, (a, b) => (a + b) / 2).ToArray();
                    }
                    // If both are default strategies, keep the existing one (do nothing)
                }
            }

            return mergedData;
        }
        static void WriteDataToCSV(string filePath, Dictionary<string, double[]> data)
        {
            using (StreamWriter sw = new StreamWriter(filePath))
            {
                sw.WriteLine("Key,Values");
                foreach (var kvp in data)
                {
                    string valuesString = string.Join(",", kvp.Value.Select(d => d.ToString()));
                    sw.WriteLine($"{kvp.Key},{valuesString}");
                }
            }
        }

        static void Main(string[] args)
        {
            bool singleThread = false;
            bool fileMergingEnabled = false; //merging of strategy files is useful, but merging of strategies while mainting weighting is not implemented.
            var numIterations = 10000;
            int progressInterval = 1000;
            if (singleThread == false)
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

                //const string path = "Whist.strategy";
                //strategyProfile.Save(path);
                //strategyProfile = StrategyProfile.Load(path);

                string cPath = "StrategyTest.csv";

                var newData = new Dictionary<string, double[]>(strategyProfile.ToDict());

                bool existingFilePresent = File.Exists(cPath);
                if (existingFilePresent && fileMergingEnabled)
                {
                    var existingData = ReadExistingCSV(cPath);
                    var mergedData = MergeStrategies(existingData, newData);
                    Console.WriteLine("Merging strategy profile:");
                    WriteDataToCSV(cPath, mergedData);
                    //if merged, game values need to be updated accordingly
                }
                else
                {
                    Console.WriteLine("No file to merge with found(or merging is disabled), saving strategy profile:");
                    WriteDataToCSV(cPath, newData);
                }

                // print results
                Console.WriteLine("Expected game values:");
                Console.WriteLine(string.Join(", ", expectedGameValues));
                Console.WriteLine("Strategy profile saved:");
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
                                    Console.WriteLine($"Core {coreIndex}: Batch {iter}/{iterationsForThisCore}");
                                    stopwatch.Stop();
                                    TimeSpan elapsed = stopwatch.Elapsed;
                                    Console.WriteLine($"Core {coreIndex}: Batch {iter} took {elapsed.Minutes} minutes and {elapsed.Seconds}.{elapsed.Milliseconds:D3} seconds");
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
                //f# map strategy merging
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
                //const string path = "WhistMulti.strategy";
                //finalStrategyProfileObject.Save(path);

                // Output results
                Console.WriteLine("Expected game values from this batch:");
                Console.WriteLine(string.Join(", ", finalExpectedGameValues));

                // Save to CSV
                string cPath = "StrategyMultiTest.csv";
                bool existingFilePresent = File.Exists(cPath);
                //existing game values need to be merged
                if (existingFilePresent && fileMergingEnabled)
                {
                    var existingData = ReadExistingCSV(cPath);
                    var mergedData = MergeStrategies(existingData, finalStrategyDictionary);
                    Console.WriteLine("Merging strategy profile with existing file:");
                    WriteDataToCSV(cPath, mergedData);
                }
                else
                {
                    Console.WriteLine("No file to merge with found(or merging is disabled), saving strategy profile:");
                    WriteDataToCSV(cPath, finalStrategyDictionary);
                }
                Console.WriteLine("Strategy profile written to " + cPath);


            }
        }
    }
}









