namespace Cfrm
// Much of this file follows the design of https://github.com/brianberns/Cfrm,
//additions made to triviality detection and logic for running cfr on filterted set of legal actions rather than all legal actions 

open System.IO
open MathNet.Numerics.LinearAlgebra

/// Represents the information accumulated during a batch of
/// iterations.
type CfrBatch<'gameState, 'action when 'gameState :> GameState<'action>> =
    {
        /// Callback to start a new game.
        GetInitialState : int (*iteration #*) -> 'gameState

        /// Per-player utilities.
        Utilities : Vector<float>

        /// Total number of iterations run so far.
        NumIterations : int

        /// Information sets.
        InfoSetMap : InfoSetMap

        /// Strategy profile.
        StrategyProfile : StrategyProfile
    }

    /// Per-player expected game values.
    member batch.ExpectedGameValues =
        batch.Utilities / float batch.NumIterations

module CfrBatch =

    /// Initializes batch input.
    let create numPlayers getInitialState =
        {
            GetInitialState = getInitialState
            Utilities = DenseVector.zero numPlayers
            NumIterations = 0
            InfoSetMap = Map.empty
            StrategyProfile = StrategyProfile(Map.empty)
        }

    /// Saves a vector.
    let private saveVector (vector : Vector<float>) (wtr : BinaryWriter) =
        wtr.Write(uint8 vector.Count)
        for value in vector do
            wtr.Write(value)

    /// Saves the given batch to a file.
    let save path batch =

        use stream = new FileStream(path, FileMode.Create)
        use wtr = new BinaryWriter(stream)

            // expected game values
        wtr |> saveVector batch.Utilities
        wtr.Write(batch.NumIterations)

            // info set map
        wtr.Write(batch.InfoSetMap.Count)
        for (KeyValue(key, infoSet)) in batch.InfoSetMap do
            wtr.Write(key)
            wtr |> saveVector infoSet.RegretSum
            wtr |> saveVector infoSet.StrategySum

    /// Loads a vector
    let private loadVector (rdr : BinaryReader) =
        let n = rdr.ReadByte() |> int
        [|
            for _ = 1 to n do
                rdr.ReadDouble()
        |] |> DenseVector.ofArray

    /// Loads a batch from a file.
    let load path getInitialState =

        use stream = new FileStream(path, FileMode.Open)
        use rdr = new BinaryReader(stream)

            // expected game values
        let utilities = loadVector rdr
        let numIterations = rdr.ReadInt32()

            // info set map
        let infoSetMap =
            let nInfoSets = rdr.ReadInt32()
            (Map.empty, seq { 1 .. nInfoSets })
                ||> Seq.fold (fun acc _ ->
                    let key = rdr.ReadString()
                    let infoSet =
                        {
                            RegretSum = loadVector rdr
                            StrategySum = loadVector rdr
                        }
                    acc |> Map.add key infoSet)

        if stream.Length <> stream.Position then
            failwith "Corrupt batch"
        {
            GetInitialState = getInitialState
            Utilities = utilities
            NumIterations = numIterations
            InfoSetMap = infoSetMap
            StrategyProfile =
                InfoSetMap.toStrategyProfile infoSetMap
        }

module CounterFactualRegret =

    /// Computes counterfactual reach probability.
    let private getCounterFactualReachProb (probs : Vector<_>) iPlayer =
        let prod vector = vector |> Vector.fold (*) 1.0
        (prod probs.[0 .. iPlayer-1]) * (prod probs.[iPlayer+1 ..])

    /// Main CFR loop.
    let rec private loop infoSetMap reachProbs (gameState : GameState<_>) =

        match gameState.TerminalValuesOpt with

                // game is still in progress
            | None ->

                // can prune node? (see https://github.com/deepmind/open_spiel/issues/102)
                if reachProbs |> Vector.forall ((=) 0.0) then
                    DenseVector.zero reachProbs.Count, infoSetMap

                    // evaluate node
                else

                    let legalActions = gameState.LegalActions
                    match legalActions.Length with
                    | 0 -> failwith "No legal actions"
                    | 1 ->   // trivial case
                        let nextState = gameState.AddAction(legalActions.[0])
                        loop infoSetMap reachProbs nextState
                    | _ -> 
                        let filter = gameState.FilterLegalActions
                        match filter with
                        | null -> cfrCore infoSetMap reachProbs gameState legalActions
                        | filterArray ->
                        // Count legal (unfiltered) actions
                            let legalCount = filterArray |> Array.sumBy (fun x -> if x = 0 then 1 else 0)
                            match legalCount with
                            | 0 -> cfrCore infoSetMap reachProbs gameState legalActions  // Use backup legal moves
                            | 1 -> // Only one legal action after filtering - treat as trivial case
                                let actionIndex = Array.findIndex ((=) 0) filterArray
                                let nextState = gameState.AddAction(legalActions.[actionIndex])
                                loop infoSetMap reachProbs nextState
                            | _ -> cfrCoreFiltered infoSetMap reachProbs gameState filterArray legalActions
                   

                // game is over
            | Some values ->
                DenseVector.ofArray values, infoSetMap

    /// Core CFR algorithm.
    and private cfrCore infoSetMap (reachProbs : Vector<_>) gameState legalActions =

            // obtain info set for this game state
        let key = gameState.Key
        let infoSet, infoSetMap =
            infoSetMap
                |> InfoSetMap.getInfoSet key legalActions.Length

            // update strategy for this player in this info set
        let iCurPlayer = gameState.CurrentPlayerIdx
        let strategy, infoSet =
            infoSet |> InfoSet.getStrategy reachProbs.[iCurPlayer]

            // recurse for each legal action
        let counterFactualValues, infoSetMaps =
            legalActions
                |> Seq.indexed
                |> Seq.scan (fun (_, accMap) (iAction, action) ->
                    let nextState = gameState.AddAction(action)
                    let reachProbs =
                        reachProbs
                            |> Vector.mapi (fun iPlayer reach ->
                                if iPlayer = iCurPlayer then
                                    reach * strategy.[iAction]
                                else
                                    reach)
                    loop accMap reachProbs nextState)
                        (DenseVector.ofArray Array.empty, infoSetMap)
                |> Seq.toArray
                |> Array.unzip
        assert(counterFactualValues.Length = legalActions.Length + 1)
        let counterFactualValues =
            counterFactualValues.[1..] |> DenseMatrix.ofRowSeq
        let infoSetMap = infoSetMaps |> Array.last

            // value of current game state is counterfactual values weighted
            // by action probabilities
        let result = strategy * counterFactualValues
        assert(result.Count = reachProbs.Count)

            // accumulate regret
        let infoSet =
            let cfReachProb =
                getCounterFactualReachProb reachProbs iCurPlayer
            let regrets =
                cfReachProb *
                    (counterFactualValues.[0.., iCurPlayer] - result.[iCurPlayer])
            infoSet |> InfoSet.accumulateRegret regrets

        let infoSetMap = infoSetMap |> Map.add key infoSet
        result, infoSetMap
    /// Core CFR algorithm for filtered actions. Added function
    and private cfrCoreFiltered infoSetMap (reachProbs : Vector<_>) gameState filterArray legalActions =

        // obtain info set for this game state
        let key = gameState.Key
        let infoSet, infoSetMap =
            infoSetMap
            |> InfoSetMap.getInfoSet key legalActions.Length

            // update strategy for this player in this info set
        let iCurPlayer = gameState.CurrentPlayerIdx
        let strategy, infoSet =
            infoSet |> InfoSet.getStrategy reachProbs.[iCurPlayer]
        
        // === Filtering Logic ===
        let total = strategy |> Vector.sum
        if total > 0.0 then
            for i = 0 to strategy.Count - 1 do
                strategy.[i] <- strategy.[i] / total
        
        // Identify filtered and unfiltered actions
        let remainingIndices = 
            [0 .. strategy.Count - 1] 
            |> List.filter (fun i -> filterArray.[i] = 0)
        
        let numRemaining = List.length remainingIndices

        if numRemaining < strategy.Count then
        // Calculate total probability mass of unfiltered actions
            let unfilteredMass = 
                remainingIndices 
                |> List.sumBy (fun i -> strategy.[i])
        
            // Zero out filtered actions
            for i = 0 to strategy.Count - 1 do
                if filterArray.[i] = 1 then
                    strategy.[i] <- 0.0
        
            // If we have any unfiltered actions, redistribute their probabilities
            if numRemaining > 0 then
                // Scale up remaining probabilities to sum to 1.0
                let scaleFactor = if unfilteredMass > 0.0 then 1.0 / unfilteredMass else 1.0 / float numRemaining
                for i in remainingIndices do
                    if unfilteredMass > 0.0 then
                        strategy.[i] <- strategy.[i] * scaleFactor
                    else
                        // If all remaining actions had 0 probability, give them equal probability
                        strategy.[i] <- 1.0 / float numRemaining
                

        // Verify sum is 1.0 (within floating point precision)
        let sum = strategy |> Vector.sum
        assert(abs(1.0 - sum) < 1e-10)

        // === Continue with CFR recursion ===
        let counterFactualValues, infoSetMaps =
            legalActions
            |> Seq.indexed
            |> Seq.scan (fun (_, accMap) (iAction, action) ->
                let nextState = gameState.AddAction(action)
                let reachProbs =
                    reachProbs
                    |> Vector.mapi (fun iPlayer reach ->
                        if iPlayer = iCurPlayer then
                            reach * strategy.[iAction]
                        else
                            reach)
                loop accMap reachProbs nextState)
                (DenseVector.ofArray Array.empty, infoSetMap)
            |> Seq.toArray
            |> Array.unzip

        assert(counterFactualValues.Length = legalActions.Length + 1)
        let counterFactualValues =
            counterFactualValues.[1..] |> DenseMatrix.ofRowSeq
        let infoSetMap = infoSetMaps |> Array.last

        // Compute the value of the current game state weighted by action probabilities
        let result = strategy * counterFactualValues
        assert(result.Count = reachProbs.Count)

        // Accumulate regret
        let infoSet =
            let cfReachProb = getCounterFactualReachProb reachProbs iCurPlayer
            let regrets =
                cfReachProb *
                (counterFactualValues.[0.., iCurPlayer] - result.[iCurPlayer])
            infoSet |> InfoSet.accumulateRegret regrets

        let infoSetMap = infoSetMap |> Map.add key infoSet
        result, infoSetMap
        
        (*
        /// Core CFR algorithm for filtered actions. Added function
    and private cfrCoreFiltered infoSetMap (reachProbs : Vector<_>) gameState filteredIndex legalActions =

             // obtain info set for this game state
        let key = gameState.Key
        let infoSet, infoSetMap =
            infoSetMap
                |> InfoSetMap.getInfoSet key legalActions.Length

            // update strategy for this player in this info set
        let iCurPlayer = gameState.CurrentPlayerIdx
        let strategy, infoSet =
            infoSet |> InfoSet.getStrategy reachProbs.[iCurPlayer]
        let difference = strategy.[filteredIndex]
        strategy.[filteredIndex] <- 0.0 // Set the playrate of the filtered action to 0

        // Calculate total probability of remaining actions
        let totalProb = strategy |> Vector.sum 
        let remainingActions = strategy.Count - 1

        // Calculate equal probability for remaining actions
        let equalProb =  difference / float remainingActions

        // Distribute the remaining probability equally among the other actions
        for i = 0 to strategy.Count - 1 do
            if i <> filteredIndex then
                strategy.[i] <- strategy.[i] + equalProb   
  

        let counterFactualValues, infoSetMaps =
            legalActions
                |> Seq.indexed
                |> Seq.scan (fun (_, accMap) (iAction, action) ->
                    let nextState = gameState.AddAction(action)
                    let reachProbs =
                        reachProbs
                            |> Vector.mapi (fun iPlayer reach ->
                                if iPlayer = iCurPlayer then
                                    reach * strategy.[iAction]
                                else
                                    reach)
                    loop accMap reachProbs nextState)
                        (DenseVector.ofArray Array.empty, infoSetMap)
                |> Seq.toArray
                |> Array.unzip
        assert(counterFactualValues.Length = legalActions.Length + 1)
        let counterFactualValues =
            counterFactualValues.[1..] |> DenseMatrix.ofRowSeq
        let infoSetMap = infoSetMaps |> Array.last

            // value of current game state is counterfactual values weighted
            // by action probabilities
        let result = strategy * counterFactualValues
        assert(result.Count = reachProbs.Count)

            // accumulate regret
        let infoSet =
            let cfReachProb =
                getCounterFactualReachProb reachProbs iCurPlayer
            let regrets =
                cfReachProb *
                    (counterFactualValues.[0.., iCurPlayer] - result.[iCurPlayer])
            infoSet |> InfoSet.accumulateRegret regrets

        let infoSetMap = infoSetMap |> Map.add key infoSet
        result, infoSetMap
        *)
    /// Runs a CFR minimization batch.
    let minimizeBatch numIterations batch =

            // accumulate utilties
        let numPlayers = batch.Utilities.Count
        let utilities, infoSetMap =
            let iterations = seq { 1 .. numIterations }
            ((batch.Utilities, batch.InfoSetMap), iterations)
                ||> Seq.fold (fun (accUtils, accMap) iterNum ->
                    let utils, accMap =
                        batch.GetInitialState(iterNum)
                            |> loop accMap (DenseVector.create numPlayers 1.0)
                    accUtils + utils, accMap)

            // compute equilibrium strategies and expected game values
        let numIterations = batch.NumIterations + numIterations
        {
            batch with
                Utilities = utilities
                InfoSetMap = infoSetMap
                NumIterations = numIterations
                StrategyProfile =
                    InfoSetMap.toStrategyProfile infoSetMap
        }

    /// Runs CFR minimization for the given number of iterations.
    let minimize numIterations numPlayers getInitialState =
        let batch =
            CfrBatch.create numPlayers getInitialState
                |> minimizeBatch numIterations   // run a single batch
        batch.ExpectedGameValues.ToArray(), batch.StrategyProfile

/// C# support.
[<AbstractClass; Sealed>]
type CounterFactualRegret private () =

    /// Runs CFR minimization for the given number of iterations.
    static member Minimize(numIterations, numPlayers, getInitialState) =
        let getInitialStateF =
            FuncConvert.FromFunc<int, GameState<_>>(getInitialState)
        CounterFactualRegret.minimize
            numIterations
            numPlayers
            getInitialStateF
