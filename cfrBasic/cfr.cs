using System;


    class Cfr
    {
    //what info needs to be defined 
    int players;
    //what deck is being used?
    Deck standardDeck = new Deck();








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






