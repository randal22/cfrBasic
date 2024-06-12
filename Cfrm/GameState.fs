﻿namespace Cfrm

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
    /// check for trivial actions
    abstract member checkTrivial : bool
    
    abstract member FilterLegalActions :  int[]
    /// the trivial action, index of legal actions
    // member TrivialAction : int

    /// Moves to the next game state by taking the given action.
    abstract member AddAction : 'action -> GameState<'action>

    /// Default implementation.
    default this.TerminalValues =
        this.TerminalValuesOpt |> Option.toObj

    /// Default implementation.
    default this.TerminalValuesOpt =
        this.TerminalValues |> Option.ofObj
