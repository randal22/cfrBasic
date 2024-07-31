namespace Cfrm
// Much of this file follows the design of https://github.com/brianberns/Cfrm,
//additonal functionality for wider classic card game triviality detection and filtering of weakly dominated strategies has been added

/// Immutable representation of a game state.
[<AbstractClass>]
type GameState<'action>() =

    /// Current player's 0-based index.
    abstract member CurrentPlayerIdx : int

    /// Unique key for this game state.
    abstract member Key : string

    /// Per-player payoffs iff this is a terminal game state;
    /// None otherwise. F# subclasses should override this.
    abstract member TerminalValuesOpt : Option<float[]>

    /// Per-player payoffs if this is a terminal game state;
    /// null otherwise. C# subclasses should override this.
    abstract member TerminalValues : float[]

    /// Legal actions available in this game state.
    abstract member LegalActions : 'action[]
    
    //new functionality
    /// check for trivial actions
    abstract member checkTrivial : bool
    
    abstract member FilterLegalActions :  int[]
    //end of new functionality

    /// Moves to the next game state by taking the given action.
    abstract member AddAction : 'action -> GameState<'action>

    /// Default implementation.
    default this.TerminalValues =
        this.TerminalValuesOpt |> Option.toObj

    /// Default implementation.
    default this.TerminalValuesOpt =
        this.TerminalValues |> Option.ofObj
