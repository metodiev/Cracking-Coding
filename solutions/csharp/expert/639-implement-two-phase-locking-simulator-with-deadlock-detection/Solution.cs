// Task 639 - Implement two-phase locking simulator with deadlock detection. (Tier: Expert, CTCI Ch 14 - Databases)
//
// A real (single-threaded) lock manager: transactions are explicit schedules of
// LockShared/LockUnlock/Read/Write/Increment/Commit steps and the simulator runs one step per
// transaction per round, so interleavings are declared by the plans and are fully deterministic.
// Reads need a shared or exclusive lock, writes/increments need an exclusive lock, and locks are
// granted through a per-resource FIFO wait queue: a request that cannot be granted is queued, and a
// later *compatible* request still queues behind the earlier one (no writer starvation, no
// barging). When no transaction can move, the wait-for graph (queued txn -> current conflicting
// holders, or -> the txn ahead of it in the queue) is searched for a cycle; a cycle is reported as a
// deadlock and its highest-numbered member is the victim: it is rolled back with its undo log
// (values it overwrote are restored), its locks are released and the queues are drained.
//
// Simplified: one step per transaction per round (no threads, no lock timeouts, no lock escalation),
// undo is in-memory pre-image restoration rather than a WAL, and only a single deadlock cycle per
// stall is resolved. Verified: the classic two-transaction cycle, a three-transaction cycle, an
// S->X upgrade cycle, plain blocking, FIFO fairness, two-phase-locking plan validation, and - the
// real correctness property - the committed history is conflict-serialisable while the final state
// equals a serial replay of exactly the committed transactions.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

internal enum LockMode
{
    Shared,
    Exclusive,
}

internal enum StepKind
{
    LockShared,
    LockExclusive,
    Unlock,
    Read,
    Write,
    Increment,
    Commit,
}

internal sealed class Step
{
    public readonly StepKind Kind;
    public readonly string Resource;
    public readonly int Value;

    private Step(StepKind kind, string resource, int value)
    {
        Kind = kind;
        Resource = resource;
        Value = value;
    }

    public static Step LockShared(string resource) => new Step(StepKind.LockShared, resource, 0);
    public static Step LockExclusive(string resource) => new Step(StepKind.LockExclusive, resource, 0);
    public static Step Unlock(string resource) => new Step(StepKind.Unlock, resource, 0);
    public static Step Read(string resource) => new Step(StepKind.Read, resource, 0);
    public static Step Write(string resource, int value) => new Step(StepKind.Write, resource, value);
    public static Step Increment(string resource, int delta) => new Step(StepKind.Increment, resource, delta);

    public static Step Commit() => new Step(StepKind.Commit, "", 0);

    public bool IsLock => Kind == StepKind.LockShared || Kind == StepKind.LockExclusive;
    public bool IsWrite => Kind == StepKind.Write || Kind == StepKind.Increment;
}

internal sealed class Plan
{
    public readonly string Name;
    public readonly List<Step> Steps;

    public Plan(string name, params Step[] steps)
    {
        Name = name;
        Steps = new List<Step>(steps);
    }
}

internal sealed class HistEntry
{
    public readonly int Txn;
    public readonly string Resource;
    public readonly bool IsWrite;

    public HistEntry(int txn, string resource, bool isWrite)
    {
        Txn = txn;
        Resource = resource;
        IsWrite = isWrite;
    }
}

internal sealed class GrantRecord
{
    public readonly int Txn;
    public readonly string Resource;
    public readonly LockMode Mode;

    public GrantRecord(int txn, string resource, LockMode mode)
    {
        Txn = txn;
        Resource = resource;
        Mode = mode;
    }
}

internal sealed class LockSimulation
{
    public readonly List<string> Trace = new List<string>();
    public readonly List<int> CommittedOrder = new List<int>();
    public readonly List<int> AbortedOrder = new List<int>();
    public readonly List<List<int>> Deadlocks = new List<List<int>>();
    public readonly List<HistEntry> History = new List<HistEntry>();
    public readonly List<GrantRecord> Grants = new List<GrantRecord>();
    public readonly Dictionary<string, int> FinalState = new Dictionary<string, int>(StringComparer.Ordinal);
    public readonly Dictionary<int, Dictionary<string, int>> Reads = new Dictionary<int, Dictionary<string, int>>();

    public int WaitEvents;
}

internal sealed class LockSimulator
{
    private sealed class Pending
    {
        public int Txn;
        public LockMode Mode;
        public string Resource = "";
    }

    private sealed class TxnState
    {
        public int Index;
        public Plan Plan = null!;
        public int ProgramCounter;
        public bool Blocked;
        public bool Done;
        public bool Committed;
        public Pending? Wait;
        public readonly Dictionary<string, LockMode> Held = new Dictionary<string, LockMode>(StringComparer.Ordinal);
        public readonly List<KeyValuePair<string, int>> UndoLog = new List<KeyValuePair<string, int>>();
        public readonly Dictionary<string, int> ReadValues = new Dictionary<string, int>(StringComparer.Ordinal);
    }

    private readonly List<Plan> _plans;
    private readonly List<TxnState> _state = new List<TxnState>();
    private readonly Dictionary<string, int> _store = new Dictionary<string, int>(StringComparer.Ordinal);
    private readonly Dictionary<string, List<int>> _shared = new Dictionary<string, List<int>>(StringComparer.Ordinal);
    private readonly Dictionary<string, int> _exclusive = new Dictionary<string, int>(StringComparer.Ordinal);
    private readonly Dictionary<string, List<Pending>> _queues = new Dictionary<string, List<Pending>>(StringComparer.Ordinal);
    private readonly LockSimulation _result = new LockSimulation();

    public LockSimulator(List<Plan> plans, Dictionary<string, int> initial)
    {
        _plans = plans;
        foreach (KeyValuePair<string, int> seed in initial) _store[seed.Key] = seed.Value;
        for (int i = 0; i < plans.Count; i++) _state.Add(new TxnState { Index = i, Plan = plans[i] });
    }

    // Basic two-phase locking discipline: locks are only taken while nothing has been released, every
    // read holds S or X, every write holds X, unlocks target a held lock and nothing follows COMMIT.
    public static bool Validate(Plan plan)
    {
        var held = new Dictionary<string, LockMode>(StringComparer.Ordinal);
        bool released = false, committed = false;
        foreach (Step step in plan.Steps)
        {
            if (committed) return false;
            switch (step.Kind)
            {
                case StepKind.LockShared:
                case StepKind.LockExclusive:
                    if (released) return false;
                    LockMode mode = step.Kind == StepKind.LockShared ? LockMode.Shared : LockMode.Exclusive;
                    if (held.TryGetValue(step.Resource, out LockMode existing))
                    {
                        if (existing == LockMode.Exclusive || mode == LockMode.Shared) return false;
                    }
                    held[step.Resource] = mode;
                    break;
                case StepKind.Unlock:
                    if (!held.Remove(step.Resource)) return false;
                    released = true;
                    break;
                case StepKind.Read:
                    if (!held.ContainsKey(step.Resource)) return false;
                    break;
                case StepKind.Write:
                case StepKind.Increment:
                    if (!held.TryGetValue(step.Resource, out LockMode lockMode) || lockMode != LockMode.Exclusive) return false;
                    break;
                case StepKind.Commit:
                    committed = true;
                    break;
            }
        }
        return true;
    }

    public LockSimulation Run()
    {
        foreach (Plan plan in _plans)
        {
            if (!Validate(plan)) throw new Exception("plan " + plan.Name + " violates two-phase locking");
        }

        int guard = 0;
        while (true)
        {
            if (++guard > 20000) throw new Exception("simulator failed to terminate");
            bool progress = false;
            foreach (TxnState txn in _state)
            {
                if (txn.Done || txn.Blocked) continue;
                if (ExecuteOneStep(txn)) progress = true;
            }
            if (progress) continue;

            if (!ResolveDeadlock())
            {
                if (_state.Any(t => !t.Done && t.Blocked)) throw new Exception("deadlock resolution stalled");
                break;
            }
        }

        foreach (TxnState txn in _state)
        {
            if (!txn.Done) throw new Exception("transaction " + txn.Plan.Name + " did not finish");
        }
        foreach (KeyValuePair<string, int> value in _store) _result.FinalState[value.Key] = value.Value;
        for (int i = 0; i < _state.Count; i++) _result.Reads[i] = _state[i].ReadValues;
        return _result;
    }

    private bool ExecuteOneStep(TxnState txn)
    {
        if (txn.ProgramCounter >= txn.Plan.Steps.Count)
        {
            CommitTxn(txn);
            return true;
        }

        Step step = txn.Plan.Steps[txn.ProgramCounter];
        switch (step.Kind)
        {
            case StepKind.LockShared:
                return RequestLock(txn, step.Resource, LockMode.Shared);
            case StepKind.LockExclusive:
                return RequestLock(txn, step.Resource, LockMode.Exclusive);
            case StepKind.Unlock:
                if (!txn.Held.Remove(step.Resource)) throw new Exception(txn.Plan.Name + " unlocks " + step.Resource + " without holding it");
                ReleaseLock(txn, step.Resource);
                txn.ProgramCounter++;
                _result.Trace.Add(txn.Plan.Name + " unlock " + step.Resource);
                return true;
            case StepKind.Read:
            {
                LockMode mode = RequireLock(txn, step.Resource);
                int value = Value(step.Resource);
                txn.ReadValues[step.Resource] = value;
                _result.History.Add(new HistEntry(txn.Index, step.Resource, false));
                txn.ProgramCounter++;
                _result.Trace.Add(txn.Plan.Name + " read " + step.Resource + "=" + value + " (" + mode + ")");
                return true;
            }
            case StepKind.Write:
            case StepKind.Increment:
            {
                RequireExclusive(txn, step.Resource);
                int old = Value(step.Resource);
                int next = step.Kind == StepKind.Write ? step.Value : old + step.Value;
                Undo(txn, step.Resource, old);
                _store[step.Resource] = next;
                _result.History.Add(new HistEntry(txn.Index, step.Resource, true));
                txn.ProgramCounter++;
                _result.Trace.Add(txn.Plan.Name + " write " + step.Resource + "=" + next + " (undo " + old + ")");
                return true;
            }
            case StepKind.Commit:
                CommitTxn(txn);
                return true;
            default:
                throw new Exception("unhandled step " + step.Kind);
        }
    }

    private int Value(string resource) => _store.TryGetValue(resource, out int value) ? value : 0;

    private LockMode RequireLock(TxnState txn, string resource)
    {
        if (!txn.Held.TryGetValue(resource, out LockMode mode))
        {
            throw new Exception(txn.Plan.Name + " reads " + resource + " without a lock");
        }
        return mode;
    }

    private void RequireExclusive(TxnState txn, string resource)
    {
        if (!txn.Held.TryGetValue(resource, out LockMode mode) || mode != LockMode.Exclusive)
        {
            throw new Exception(txn.Plan.Name + " writes " + resource + " without an exclusive lock");
        }
    }

    private void Undo(TxnState txn, string resource, int old)
    {
        txn.UndoLog.Add(new KeyValuePair<string, int>(resource, old));
    }

    private bool RequestLock(TxnState txn, string resource, LockMode mode)
    {
        List<Pending> queue = Queue(resource);
        bool head = queue.Count > 0 && queue[0].Txn == txn.Index;
        if (head)
        {
            if (!CanGrant(txn, resource, mode)) return false;
            queue.RemoveAt(0);
            Grant(txn, resource, mode);
            txn.Blocked = false;
            txn.Wait = null;
            txn.ProgramCounter++;
            return true;
        }
        if (queue.Count == 0 && CanGrant(txn, resource, mode))
        {
            Grant(txn, resource, mode);
            txn.ProgramCounter++;
            return true;
        }

        if (txn.Wait is null)
        {
            var pending = new Pending { Txn = txn.Index, Mode = mode, Resource = resource };
            queue.Add(pending);
            txn.Wait = pending;
            txn.Blocked = true;
            _result.WaitEvents++;
            _result.Trace.Add(txn.Plan.Name + " wait " + mode + "(" + resource + ") behind [" + string.Join(",", Holders(resource)) + "]");
        }
        return false;
    }

    private bool CanGrant(TxnState txn, string resource, LockMode mode)
    {
        if (_exclusive.TryGetValue(resource, out int owner) && owner != txn.Index) return false;
        if (mode == LockMode.Exclusive)
        {
            foreach (int holder in Shared(resource))
            {
                if (holder != txn.Index) return false;
            }
        }
        return true;
    }

    private void Grant(TxnState txn, string resource, LockMode mode)
    {
        txn.Held[resource] = mode;
        if (mode == LockMode.Exclusive)
        {
            Shared(resource).Remove(txn.Index);
            _exclusive[resource] = txn.Index;
        }
        else
        {
            Shared(resource).Add(txn.Index);
        }
        _result.Grants.Add(new GrantRecord(txn.Index, resource, mode));
        _result.Trace.Add(txn.Plan.Name + " grant " + mode + "(" + resource + ")");
    }

    private void ReleaseLock(TxnState txn, string resource)
    {
        if (_exclusive.TryGetValue(resource, out int owner) && owner == txn.Index) _exclusive.Remove(resource);
        Shared(resource).Remove(txn.Index);
        DrainQueue(resource);
    }

    private void DrainQueue(string resource)
    {
        List<Pending> queue = Queue(resource);
        while (queue.Count > 0)
        {
            Pending head = queue[0];
            TxnState waiter = _state[head.Txn];
            if (waiter.Done)
            {
                queue.RemoveAt(0);
                continue;
            }
            if (!CanGrant(waiter, resource, head.Mode)) break;
            queue.RemoveAt(0);
            waiter.Blocked = false;
            waiter.Wait = null;
            Grant(waiter, resource, head.Mode);
            waiter.ProgramCounter++;
        }
    }

    private void CommitTxn(TxnState txn)
    {
        foreach (string resource in txn.Held.Keys.ToList())
        {
            txn.Held.Remove(resource);
            ReleaseLock(txn, resource);
        }
        txn.Done = true;
        txn.Committed = true;
        txn.Blocked = false;
        _result.CommittedOrder.Add(txn.Index);
        _result.Trace.Add(txn.Plan.Name + " COMMIT");
    }

    private void AbortTxn(TxnState txn)
    {
        txn.Done = true;
        txn.Committed = false;
        txn.Blocked = false;
        txn.Wait = null;
        txn.ProgramCounter = txn.Plan.Steps.Count;
        foreach (List<Pending> queue in _queues.Values)
        {
            queue.RemoveAll(pending => pending.Txn == txn.Index);
        }
        for (int i = txn.UndoLog.Count - 1; i >= 0; i--)
        {
            KeyValuePair<string, int> entry = txn.UndoLog[i];
            _store[entry.Key] = entry.Value;
            _result.Trace.Add(txn.Plan.Name + " undo " + entry.Key + "=" + entry.Value);
        }
        foreach (string resource in txn.Held.Keys.ToList())
        {
            txn.Held.Remove(resource);
            ReleaseLock(txn, resource);
        }
        _result.AbortedOrder.Add(txn.Index);
        _result.Trace.Add(txn.Plan.Name + " ABORT (deadlock victim)");
    }

    private List<int> Holders(string resource)
    {
        var holders = new List<int>();
        if (_exclusive.TryGetValue(resource, out int owner)) holders.Add(owner);
        foreach (int shared in Shared(resource))
        {
            if (!holders.Contains(shared)) holders.Add(shared);
        }
        holders.Sort();
        return holders;
    }

    private List<int> Blockers(TxnState txn)
    {
        var blockers = new List<int>();
        Pending wait = txn.Wait!;
        List<Pending> queue = Queue(wait.Resource);
        int position = queue.FindIndex(pending => pending.Txn == txn.Index);
        if (position > 0)
        {
            blockers.Add(queue[0].Txn);
        }
        else
        {
            foreach (int holder in Holders(wait.Resource))
            {
                if (holder != txn.Index) blockers.Add(holder);
            }
        }
        blockers.Sort();
        return blockers;
    }

    // Returns the first cycle found in the wait-for graph, or null when the graph is acyclic.
    private List<int>? FindCycle()
    {
        var graph = new Dictionary<int, List<int>>();
        foreach (TxnState txn in _state)
        {
            if (!txn.Done && txn.Blocked) graph[txn.Index] = Blockers(txn);
        }
        var state = new Dictionary<int, int>();
        var stack = new List<int>();
        foreach (int start in graph.Keys.OrderBy(k => k))
        {
            List<int>? cycle = Dfs(start, graph, state, stack);
            if (cycle != null) return cycle;
        }
        return null;
    }

    private List<int>? Dfs(int node, Dictionary<int, List<int>> graph, Dictionary<int, int> state, List<int> stack)
    {
        state[node] = 1;
        stack.Add(node);
        if (graph.TryGetValue(node, out List<int> targets))
        {
            foreach (int next in targets)
            {
                if (!graph.ContainsKey(next)) continue;
                if (!state.TryGetValue(next, out int colour) || colour == 0)
                {
                    List<int>? cycle = Dfs(next, graph, state, stack);
                    if (cycle != null) return cycle;
                }
                else if (colour == 1)
                {
                    int at = stack.IndexOf(next);
                    return stack.GetRange(at, stack.Count - at);
                }
            }
        }
        stack.RemoveAt(stack.Count - 1);
        state[node] = 2;
        return null;
    }

    private bool ResolveDeadlock()
    {
        List<int>? cycle = FindCycle();
        if (cycle == null) return false;
        _result.Deadlocks.Add(new List<int>(cycle.OrderBy(i => i)));
        int victim = cycle.Max();
        TxnState txn = _state[victim];
        _result.Trace.Add("deadlock detected involving [" + string.Join(",", cycle.OrderBy(i => i)) + "] -> victim " + txn.Plan.Name);
        AbortTxn(txn);
        foreach (string resource in QueueKeys()) DrainQueue(resource);
        return true;
    }

    private IEnumerable<string> QueueKeys() => _queues.Keys.ToList();

    private List<int> Shared(string resource)
    {
        if (!_shared.TryGetValue(resource, out List<int> holders))
        {
            holders = new List<int>();
            _shared[resource] = holders;
        }
        return holders;
    }

    private List<Pending> Queue(string resource)
    {
        if (!_queues.TryGetValue(resource, out List<Pending> queue))
        {
            queue = new List<Pending>();
            _queues[resource] = queue;
        }
        return queue;
    }
}

public static class Solution
{
    private static readonly Dictionary<string, int> Empty = new Dictionary<string, int>(StringComparer.Ordinal);

    private static Dictionary<string, int> Seed(params (string Key, int Value)[] pairs)
    {
        var initial = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach ((string key, int value) in pairs) initial[key] = value;
        return initial;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static Plan[] ClassicTwoWayCycle() => new[]
    {
        new Plan("T1", Step.LockExclusive("A"), Step.Read("A"), Step.Increment("A", 1),
            Step.LockExclusive("B"), Step.Read("B"), Step.Increment("B", 1), Step.Commit()),
        new Plan("T2", Step.LockExclusive("B"), Step.Read("B"), Step.Increment("B", 10),
            Step.LockExclusive("A"), Step.Read("A"), Step.Increment("A", 10), Step.Commit()),
    };

    private static void TwoWayDeadlock()
    {
        var sim = new LockSimulator(new List<Plan>(ClassicTwoWayCycle()), Seed(("A", 1), ("B", 1)));
        LockSimulation result = sim.Run();

        Check(result.Deadlocks.Count == 1, "two-way deadlock: exactly one cycle detected");
        Check(result.Deadlocks[0].Count == 2, "two-way deadlock: cycle has two members");
        Check(result.AbortedOrder.Count == 1 && result.AbortedOrder[0] == 1, "two-way deadlock: highest txn id is the victim");
        Check(result.CommittedOrder.Count == 1 && result.CommittedOrder[0] == 0, "two-way deadlock: survivor commits");
        Check(result.FinalState["A"] == 2 && result.FinalState["B"] == 2, "two-way deadlock: victim's updates were undone");
        Check(result.WaitEvents == 2, "two-way deadlock: both transactions waited once");
        Check(IsConflictSerializable(result, false), "two-way deadlock: committed history is conflict-serialisable");
        Check(SerialReplayMatches(ClassicTwoWayCycle(), result, Seed(("A", 1), ("B", 1))), "two-way deadlock: state equals serial replay of the survivor");
        Console.WriteLine("two-way deadlock: cycle [0,1] -> victim T2 rolled back, A=2 B=2");
    }

    private static Plan[] ThreeWayCycle() => new[]
    {
        new Plan("T1", Step.LockExclusive("A"), Step.Increment("A", 1), Step.LockExclusive("B"),
            Step.Increment("B", 1), Step.Commit()),
        new Plan("T2", Step.LockExclusive("B"), Step.Increment("B", 10), Step.LockExclusive("C"),
            Step.Increment("C", 10), Step.Commit()),
        new Plan("T3", Step.LockExclusive("C"), Step.Increment("C", 100), Step.LockExclusive("A"),
            Step.Increment("A", 100), Step.Commit()),
    };

    private static void ThreeWayDeadlock()
    {
        var sim = new LockSimulator(new List<Plan>(ThreeWayCycle()), Seed(("A", 0), ("B", 0), ("C", 0)));
        LockSimulation result = sim.Run();

        Check(result.Deadlocks.Count == 1, "three-way deadlock: exactly one cycle detected");
        Check(result.Deadlocks[0].Count == 3, "three-way deadlock: cycle has three members");
        Check(result.AbortedOrder.Count == 1 && result.AbortedOrder[0] == 2, "three-way deadlock: victim is T3");
        Check(result.CommittedOrder.Count == 2, "three-way deadlock: the other two commit");
        Check(result.FinalState["A"] == 1 && result.FinalState["B"] == 11 && result.FinalState["C"] == 10,
            "three-way deadlock: two increments survive, victim's 100 was undone");
        Check(IsConflictSerializable(result, false), "three-way deadlock: committed history is conflict-serialisable");
        Check(SerialReplayMatches(ThreeWayCycle(), result, Seed(("A", 0), ("B", 0), ("C", 0))), "three-way deadlock: state matches a serial replay");
        Console.WriteLine("three-way deadlock: cycle [0,1,2] -> victim T3 rolled back, A=1 B=11 C=10");
    }

    private static void PlainBlocking()
    {
        var plans = new[]
        {
            new Plan("T1", Step.LockExclusive("r"), Step.Read("r"), Step.Increment("r", 1), Step.Commit()),
            new Plan("T2", Step.LockExclusive("r"), Step.Read("r"), Step.Increment("r", 1), Step.Commit()),
        };
        var sim = new LockSimulator(new List<Plan>(plans), Seed(("r", 0)));
        LockSimulation result = sim.Run();

        Check(result.Deadlocks.Count == 0, "plain blocking: no false deadlock");
        Check(result.WaitEvents == 1, "plain blocking: T2 waited once");
        Check(result.CommittedOrder.Count == 2, "plain blocking: both commit after the waiter is released");
        Check(result.FinalState["r"] == 2, "plain blocking: no increment lost");
        Check(result.Reads[1]["r"] == 1, "plain blocking: waiter read the value committed by T1");
    }

    private static void SharedLocksAreCompatible()
    {
        var plans = new[]
        {
            new Plan("R1", Step.LockShared("A"), Step.Read("A"), Step.Commit()),
            new Plan("R2", Step.LockShared("A"), Step.Read("A"), Step.Commit()),
        };
        var sim = new LockSimulator(new List<Plan>(plans), Seed(("A", 7)));
        LockSimulation result = sim.Run();

        Check(result.WaitEvents == 0, "shared locks: readers never wait on each other");
        Check(result.CommittedOrder.Count == 2, "shared locks: both readers commit");
        Check(result.Reads[0]["A"] == 7 && result.Reads[1]["A"] == 7, "shared locks: both read the same value");
    }

    // A shared holder keeps the resource, an exclusive waiter queues, and a later *compatible* shared
    // request must queue behind it rather than barge in.
    private static void FifoQueuePreventsStarvation()
    {
        var plans = new[]
        {
            new Plan("Holder", Step.LockShared("A"), Step.Read("A"), Step.LockShared("B"), Step.Read("B"), Step.Commit()),
            new Plan("Writer", Step.LockExclusive("A"), Step.Increment("A", 1), Step.Commit()),
            new Plan("Reader", Step.LockShared("A"), Step.Read("A"), Step.Commit()),
        };
        var sim = new LockSimulator(new List<Plan>(plans), Seed(("A", 0), ("B", 0)));
        LockSimulation result = sim.Run();

        int writerGrant = result.Grants.FindIndex(g => g.Txn == 1 && g.Resource == "A" && g.Mode == LockMode.Exclusive);
        int readerGrant = result.Grants.FindIndex(g => g.Txn == 2 && g.Resource == "A" && g.Mode == LockMode.Shared);
        Check(writerGrant >= 0 && readerGrant >= 0, "fairness: both waiters are eventually granted");
        Check(writerGrant < readerGrant, "fairness: the queued writer is granted before the later compatible reader");
        Check(result.WaitEvents == 2, "fairness: the reader queued even though shared locks were compatible");
        Check(result.Deadlocks.Count == 0, "fairness: no deadlock");
        Check(result.FinalState["A"] == 1, "fairness: writer's update landed");
    }

    private static void UpgradeDeadlock()
    {
        var plans = new[]
        {
            new Plan("T1", Step.LockShared("A"), Step.Read("A"), Step.LockExclusive("A"), Step.Increment("A", 1), Step.Commit()),
            new Plan("T2", Step.LockShared("A"), Step.Read("A"), Step.LockExclusive("A"), Step.Increment("A", 1), Step.Commit()),
        };
        var sim = new LockSimulator(new List<Plan>(plans), Seed(("A", 0)));
        LockSimulation result = sim.Run();

        Check(result.Deadlocks.Count == 1, "upgrade: shared-to-exclusive conversions deadlock");
        Check(result.AbortedOrder.Count == 1 && result.AbortedOrder[0] == 1, "upgrade: T2 is the victim");
        Check(result.CommittedOrder.Count == 1 && result.CommittedOrder[0] == 0, "upgrade: T1 upgrades and commits");
        Check(result.FinalState["A"] == 1, "upgrade: exactly one increment applied");
        Check(SerialReplayMatches(plans, result, Seed(("A", 0))), "upgrade: state matches a serial replay");
    }

    private static void PlanValidation()
    {
        Check(LockSimulator.Validate(new Plan("ok", Step.LockExclusive("A"), Step.Increment("A", 1), Step.Commit())),
            "validation: strict two-phase plan accepted");
        Check(LockSimulator.Validate(new Plan("basic", Step.LockExclusive("A"), Step.Increment("A", 1), Step.Unlock("A"), Step.Commit())),
            "validation: basic two-phase plan accepted");
        Check(!LockSimulator.Validate(new Plan("lock-after-unlock", Step.LockExclusive("A"), Step.Unlock("A"), Step.LockExclusive("B"), Step.Commit())),
            "validation: acquiring a lock after an unlock is rejected");
        Check(!LockSimulator.Validate(new Plan("write-under-shared", Step.LockShared("A"), Step.Write("A", 5), Step.Commit())),
            "validation: writing under a shared lock is rejected");
        Check(!LockSimulator.Validate(new Plan("read-unlocked", Step.Read("A"), Step.Commit())),
            "validation: reading without a lock is rejected");
        Check(!LockSimulator.Validate(new Plan("unlock-not-held", Step.Unlock("A"), Step.Commit())),
            "validation: unlocking an unheld resource is rejected");
        Check(!LockSimulator.Validate(new Plan("after-commit", Step.LockExclusive("A"), Step.Commit(), Step.Increment("A", 1))),
            "validation: steps after COMMIT are rejected");

        bool threw = false;
        try
        {
            new LockSimulator(new List<Plan> { new Plan("bad", Step.LockShared("A"), Step.Write("A", 1), Step.Commit()) }, Empty).Run();
        }
        catch (Exception) { threw = true; }
        Check(threw, "validation: the simulator refuses to run an illegal plan");
    }

    // Precedence graph over the committed transactions' read/write operations.
    private static bool IsConflictSerializable(LockSimulation result, bool verbose)
    {
        var committed = new HashSet<int>(result.CommittedOrder);
        var edges = new Dictionary<int, HashSet<int>>();
        foreach (HistEntry from in result.History)
        {
            if (!committed.Contains(from.Txn)) continue;
            foreach (HistEntry to in result.History)
            {
                if (!committed.Contains(to.Txn) || from.Txn == to.Txn) continue;
                if (!string.Equals(from.Resource, to.Resource, StringComparison.Ordinal)) continue;
                if (!from.IsWrite && !to.IsWrite) continue;
                int fromIndex = result.History.IndexOf(from);
                int toIndex = result.History.IndexOf(to);
                if (fromIndex >= toIndex) continue;
                if (!edges.TryGetValue(from.Txn, out HashSet<int> targets))
                {
                    targets = new HashSet<int>();
                    edges[from.Txn] = targets;
                }
                targets.Add(to.Txn);
            }
        }

        var indegree = new Dictionary<int, int>();
        foreach (int txn in committed) indegree[txn] = 0;
        foreach (KeyValuePair<int, HashSet<int>> edge in edges)
        {
            foreach (int target in edge.Value) indegree[target]++;
        }
        var ready = new Queue<int>(indegree.Where(kv => kv.Value == 0).Select(kv => kv.Key).OrderBy(k => k));
        int visited = 0;
        while (ready.Count > 0)
        {
            int txn = ready.Dequeue();
            visited++;
            if (!edges.TryGetValue(txn, out HashSet<int> targets)) continue;
            foreach (int target in targets.OrderBy(t => t))
            {
                if (--indegree[target] == 0) ready.Enqueue(target);
            }
        }
        if (verbose) Console.WriteLine("precedence graph acyclic: " + (visited == committed.Count));
        return visited == committed.Count;
    }

    // Replays the committed transactions of a simulation one after another on a fresh store and asks
    // whether some order reproduces the simulated final state (increments are order independent here).
    private static bool SerialReplayMatches(Plan[] plans, LockSimulation result, Dictionary<string, int> initial)
    {
        var committed = result.CommittedOrder.ToList();
        foreach (int[] permutation in Permutations(committed.Count))
        {
            var store = new Dictionary<string, int>(initial, StringComparer.Ordinal);
            foreach (int index in permutation)
            {
                foreach (Step step in plans[committed[index]].Steps)
                {
                    switch (step.Kind)
                    {
                        case StepKind.Write:
                            store[step.Resource] = step.Value;
                            break;
                        case StepKind.Increment:
                            store.TryGetValue(step.Resource, out int current);
                            store[step.Resource] = current + step.Value;
                            break;
                    }
                }
            }
            bool same = store.Count == result.FinalState.Count;
            if (same)
            {
                foreach (KeyValuePair<string, int> value in store)
                {
                    if (!result.FinalState.TryGetValue(value.Key, out int seen) || seen != value.Value) same = false;
                }
            }
            if (same) return true;
        }
        return false;
    }

    private static List<int[]> Permutations(int n)
    {
        var result = new List<int[]>();
        BuildPermutations(0, n, new int[n], new bool[n], result);
        return result;
    }

    private static void BuildPermutations(int depth, int n, int[] current, bool[] used, List<int[]> result)
    {
        if (depth == n)
        {
            result.Add((int[])current.Clone());
            return;
        }
        for (int i = 0; i < n; i++)
        {
            if (used[i]) continue;
            used[i] = true;
            current[depth] = i;
            BuildPermutations(depth + 1, n, current, used, result);
            used[i] = false;
        }
    }

    public static void Main()
    {
        TwoWayDeadlock();
        ThreeWayDeadlock();
        PlainBlocking();
        SharedLocksAreCompatible();
        FifoQueuePreventsStarvation();
        UpgradeDeadlock();
        PlanValidation();
        Console.WriteLine("PASS");
    }
}
