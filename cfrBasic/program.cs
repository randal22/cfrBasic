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

namespace Cfrm.SimplifiedWhist
{
    using static Whist;
    using static Whist.Card;

    class Program
    {
        //function to check for defualt strategy, to avoid skewing data when merging
        /*
        private static bool IsDefaultStrategy(double[] strategy)
        {
            const double epsilon = 1e-6;
            if (strategy.Length <= 1) return false;
            double firstValue = strategy[0];
            return strategy.All(v => Math.Abs(v - firstValue) < epsilon);
        }*/
        private static bool IsDefaultStrategy(double[] strategy)
        {
            if (strategy.Length <= 1) return false;
            double avg = strategy.Average();
            double variance = strategy.Sum(v => Math.Pow(v - avg, 2));
            return variance < 1e-6;
        }
        //file reading helper function
        static Dictionary<string, double[]> ReadExistingCSV(string filePath)
        {
            var existingData = new Dictionary<string, double[]>();
            if (File.Exists(filePath))
            {
                using (var reader = new StreamReader(filePath))
                {
                    reader.ReadLine(); // Skip header
                    string line;
                    while ((line = reader.ReadLine()!) != null)
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
        //c# dictionary merging helper function

        static Dictionary<string, double[]> MergeStrategies(Dictionary<string, double[]> existingData, Dictionary<string, double[]> newData)
        {
            var mergedData = new Dictionary<string, double[]>(existingData);
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
        //c# file writing helper function
        static void WriteDataToCSV(string filePath, Dictionary<string, double[]> data)
        {
            using (StreamWriter sw = new StreamWriter(filePath))
            {
                sw.WriteLine("Key,Values");
                foreach (var kvp in data)
                {
                    string valuesString = string.Join(",", kvp.Value.Select(d => d.ToString()));
                    sw.WriteLine($"{kvp.Key}{valuesString}");
                }
            }
        }

        static void Main(string[] args)
        {
            bool multiThread = false; //toggle for multi-threading
            bool fileMergingEnabled = false; //toggle for merging of strategy files
            var numIterations = 10; //number of iterations to run
            int progressInterval = numIterations/10; //interval to print progress
            
            //single thread implementation
            if (multiThread == false)
            {
                Console.WriteLine("Single thread implementation starting" );
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

                string cPath = "StrategyTest.csv";

                var newData = new Dictionary<string, double[]>(strategyProfile.ToDict());

                bool existingFilePresent = File.Exists(cPath);
                if (existingFilePresent && fileMergingEnabled)//merge
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
                Console.WriteLine(string.Join(",", expectedGameValues));
                Console.WriteLine("Strategy profile saved:");
            }
            //multi-thread implementation
            else
            {
                Console.WriteLine("Multi-thread implementation starting");
                int numCores = Environment.ProcessorCount;
                int baseIterationsPerCore = numIterations / numCores;
                int remainingIterations = numIterations % numCores;
                var tasks = new Task<(double[], StrategyProfile)>[numCores];
                if (progressInterval>numIterations/ numCores)
                {
                    Console.WriteLine("progress interval too large for split workload, reducing interval to 1%:");
                    progressInterval = numIterations/100;
                }

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
                                    Console.WriteLine($"Core {coreIndex}: Progress {iter}/{iterationsForThisCore}");
                                    stopwatch.Stop();
                                    TimeSpan elapsed = stopwatch.Elapsed;
                                    Console.WriteLine($"Core {coreIndex}: Batch took {elapsed.Minutes} minutes and {elapsed.Seconds}.{elapsed.Milliseconds:D3} seconds");
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
                var finalExpectedGameValues = new DenseVector(2);
                var finalStrategyDictionary = new Dictionary<string, double[]>();

                foreach (var task in tasks)
                {
                    var (expectedGameValues, strategyProfile) = task.Result;

                    // Sum up expected game values
                    finalExpectedGameValues += new DenseVector(expectedGameValues);
                    var newData = new Dictionary<string, double[]>(strategyProfile.ToDict());
                    // Merge strategy profiles using the MergeStrategies helper function
                    finalStrategyDictionary = MergeStrategies(finalStrategyDictionary, newData);
                }

                // Normalise the expected game values
                finalExpectedGameValues /= numCores;

                // Create the final Strategy map
                var finalStrategyMap = MapModule.OfSeq<string, double[]>(
                finalStrategyDictionary.Select(kvp =>
                new Tuple<string, double[]>(kvp.Key, kvp.Value)));

                // Create the final StrategyProfile object
                var finalStrategyProfileObject = new StrategyProfile(finalStrategyMap);
                // Output results
                Console.WriteLine("Expected game values from this batch:");
                Console.WriteLine(string.Join("", finalExpectedGameValues));

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
