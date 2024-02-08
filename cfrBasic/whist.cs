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


