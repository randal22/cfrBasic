using System;

//using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Cfrm.Test.CS
{
    using static Cfrm.Test.Whist;
    using static Whist.Card;
    
    class Program
    {
        static void Main(string[] args)
        {
            var deck = Enum.GetNames(typeof(Card)).Cast<Card>().ToArray();
            var rng = new Random(0);
            var numIterations = 100000;
            //var delta = 0.03;

            var (expectedGameValues, strategyProfile) =
                CounterFactualRegret.Minimize(numIterations, 2, i =>
                {
                    var cards = Shuffle(rng, deck)[0..2];
                    return new WhistState(cards);
                });

            const string path = "Whist.strategy";
            strategyProfile.Save(path);
            strategyProfile = StrategyProfile.Load(path);

            
            var dict = strategyProfile.ToDict();
            //Assert.AreEqual(expectedGameValues[0], -1.0 / 18.0, delta);
            //var alpha = dict["J"][1];
            //Assert.IsTrue(alpha >= 0.0);
            //Assert.IsTrue(alpha <= 1.0 / 3.0);
            //Assert.AreEqual(dict["Q"][0], 1.0, delta);
            //Assert.AreEqual(dict["Qcb"][1], alpha + 1.0 / 3.0, delta);
            //Assert.AreEqual(dict["K"][1], 3.0 * alpha, delta);

            // print results
            Console.WriteLine("Expected game values:");
            Console.WriteLine(string.Join(", ", expectedGameValues));
            Console.WriteLine("Strategy profile:");
            foreach (var (key, value) in dict)
            {
                Console.WriteLine($"{key}: {string.Join(", ", value)}");
            }
        }
    }
}

   
// to do list 

// 1. define the game state and rules
//1.1 gamestate = current player, trump, local hand, expended cards (whole table), player scores?, legal moves.
//1.2 rules = legal moves, tricks/points 
// 2. define game tree
// 2.1 4 game trees ? as each player has different information?
// 3. information sets, decision points, and strategies
// 3.1 as game progresses, known information other players' hands will improve as all cards played are known to the the whole table. a player trumping aslo gives a large amount of info for example
// 3.2 decision points & strategies - based on legal moves and gamestate
// 4. utitity functions based on winning tricks (points) 
//4.1 evaluate taking lead early vs late? taking tricks asap is not always optimal play 
// 5. regrets values and strategies - based on regrets or directly on distribution over actions 
//5.1 regrets - based on the difference between the value of the current strategy and the value of the best strategy at the time of the decision
//5.2 strategies - based on probablity of other player actions






