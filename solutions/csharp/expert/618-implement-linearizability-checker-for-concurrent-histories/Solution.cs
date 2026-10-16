// Task 618 - Implement linearizability checker for concurrent histories. (Tier: Expert, CTCI Ch 15 — Threads and Locks)
//
// A linearizability checker that searches the space of legal sequential executions. A history is a
// list of operations with global invoke/response event indices; an operation A must precede B when
// A's response event is before B's invoke event (real-time order), and per-process order is inherited
// from the event order. The checker performs a backtracking search over *linearization orders*: at
// every step the legal candidates are the not-yet-placed operations that no remaining operation is
// forced to precede, and each candidate is applied to a sequential model of the object; if the model
// returns the recorded result the search continues, otherwise that branch is pruned. An operation
// whose response event is missing (still pending when the history was cut) may return any result.
// The search returns the first complete linearization as a witness, or reports that none exists.
//
// Two models are provided - a read/write register and a FIFO queue - because the two classic
// non-linearizability counterexamples are different shapes (a stale read after a completed write vs.
// a dequeue returning a value that was never enqueued in time).
//
// Verified: hand-written linearizable and non-linearizable histories for both objects give the
// expected verdict; the witness order is replayed independently to confirm it respects real-time and
// per-process order and reproduces every recorded result; the checker's verdict agrees with a
// brute-force oracle that enumerates every permutation respecting the partial order on small random
// histories; and the degenerate cases (empty history, one pending operation, all operations pending,
// a history of writes only) are handled.

using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;          // deterministic stand-in for Random
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }
}

internal sealed class HistoryOp
{
    internal int Id;
    internal int Process;
    internal int Kind;              // 0 = read/dequeue, 1 = write/enqueue
    internal int Arg;               // value written / enqueued
    internal int Result;            // observed result (unused when Response < 0)
    internal int Invoke;            // global event index of the invocation
    internal int Response;          // global event index of the response, or -1 when pending
}

/** Sequential model of the object under test; Copy() snapshots it for the backtracking search. */
internal abstract class Model
{
    internal abstract Model Copy();
    internal abstract bool Apply(int kind, int arg, out int result);
}

internal sealed class RegisterModel : Model
{
    private int value;

    internal RegisterModel() { }
    private RegisterModel(int value) { this.value = value; }

    internal override Model Copy() => new RegisterModel(value);

    internal override bool Apply(int kind, int arg, out int result)
    {
        if (kind == 1)
        {
            value = arg;
            result = 0;
            return true;
        }
        result = value;
        return true;
    }
}

internal sealed class QueueModel : Model
{
    private readonly List<int> items;

    internal QueueModel() { items = new List<int>(); }
    private QueueModel(List<int> items) { this.items = items; }

    internal override Model Copy() => new QueueModel(new List<int>(items));

    internal override bool Apply(int kind, int arg, out int result)
    {
        if (kind == 1)
        {
            items.Add(arg);
            result = 0;
            return true;
        }
        if (items.Count == 0)
        {
            result = 0;
            return false;                                  // a dequeue from an empty queue cannot explain it
        }
        result = items[0];
        items.RemoveAt(0);
        return true;
    }
}

internal sealed class LinearizabilityChecker
{
    private readonly List<HistoryOp> history;
    private readonly Model prototype;
    private readonly List<int> witness = new List<int>();
    private long steps;

    internal LinearizabilityChecker(List<HistoryOp> history, Model prototype)
    {
        this.history = history;
        this.prototype = prototype;
    }

    internal long Steps => steps;
    internal List<int> Witness => witness;

    internal bool IsLinearizable()
    {
        witness.Clear();
        steps = 0;
        return Search(prototype.Copy(), new bool[history.Count]);
    }

    private bool Search(Model model, bool[] placed)
    {
        steps++;
        if (witness.Count == history.Count) return true;
        var remaining = new List<int>();
        for (int i = 0; i < history.Count; i++)
        {
            if (!placed[i]) remaining.Add(i);
        }
        foreach (int candidate in remaining)
        {
            HistoryOp op = history[candidate];
            bool blocked = false;
            foreach (int other in remaining)
            {
                if (other == candidate) continue;
                HistoryOp rival = history[other];
                // Forced precedence: everything that finished before this operation started, and every
                // earlier operation of the same process, must be linearized first.
                bool finishesFirst = rival.Response >= 0 && op.Invoke >= 0 && rival.Response < op.Invoke;
                bool sameProcessEarlier = rival.Process == op.Process && rival.Invoke < op.Invoke;
                if (finishesFirst || sameProcessEarlier)
                {
                    blocked = true;
                    break;
                }
            }
            if (blocked) continue;
            Model next = model.Copy();
            if (!next.Apply(op.Kind, op.Arg, out int result)) continue;
            bool mustMatch = op.Response >= 0 && op.Kind == 0;
            if (mustMatch && result != op.Result) continue;
            if (op.Response >= 0 && op.Kind == 1 && op.Result != 0) continue;
            placed[candidate] = true;
            witness.Add(candidate);
            if (Search(next, placed)) return true;
            witness.RemoveAt(witness.Count - 1);
            placed[candidate] = false;
        }
        return false;
    }

    /** Independent oracle: enumerate every permutation respecting the partial order. */
    internal static bool BruteForce(List<HistoryOp> history, Model prototype)
    {
        var order = new List<int>();
        var placed = new bool[history.Count];
        return Permute(history, prototype.Copy(), placed, order, 0);
    }

    private static bool Permute(List<HistoryOp> history, Model model, bool[] placed, List<int> order, int depth)
    {
        if (depth == history.Count) return true;
        for (int i = 0; i < history.Count; i++)
        {
            if (placed[i]) continue;
            HistoryOp op = history[i];
            bool blocked = false;
            for (int j = 0; j < history.Count && !blocked; j++)
            {
                if (placed[j] || j == i) continue;
                HistoryOp rival = history[j];
                if (rival.Response >= 0 && rival.Response < op.Invoke) blocked = true;
                if (rival.Process == op.Process && rival.Invoke < op.Invoke) blocked = true;
            }
            if (blocked) continue;
            Model next = model.Copy();
            if (!next.Apply(op.Kind, op.Arg, out int result)) continue;
            if (op.Response >= 0 && ((op.Kind == 0 && result != op.Result) || (op.Kind == 1 && op.Result != 0)))
            {
                continue;
            }
            placed[i] = true;
            order.Add(i);
            if (Permute(history, next, placed, order, depth + 1)) return true;
            order.RemoveAt(order.Count - 1);
            placed[i] = false;
        }
        return false;
    }
}

public static class Solution
{
    private static ulong state = 0x618618618UL;

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static HistoryOp Op(int id, int process, int kind, int arg, int result, int invoke, int response)
    {
        return new HistoryOp
        {
            Id = id,
            Process = process,
            Kind = kind,
            Arg = arg,
            Result = result,
            Invoke = invoke,
            Response = response,
        };
    }

    /** Replay a witness order to confirm it is legal and reproduces every recorded result. */
    private static bool WitnessValid(List<HistoryOp> history, Model prototype, List<int> witness)
    {
        if (witness.Count != history.Count) return false;
        Model model = prototype.Copy();
        int placedCount = 0;
        for (int i = 0; i < witness.Count; i++)
        {
            HistoryOp op = history[witness[i]];
            for (int j = 0; j < i; j++)
            {
                HistoryOp earlier = history[witness[j]];
                // The later operation may not have finished before the earlier one started.
                if (op.Response >= 0 && op.Response < earlier.Invoke) return false;
                if (earlier.Process == op.Process && earlier.Invoke > op.Invoke) return false;
            }
            if (!model.Apply(op.Kind, op.Arg, out int result)) return false;
            if (op.Response >= 0 && op.Kind == 0 && result != op.Result) return false;
            placedCount++;
        }
        return placedCount == history.Count;
    }

    /** Build a concurrent history that is linearizable by construction, then sometimes corrupt it.
     *  The schedule interleaves invokes and responses (one outstanding operation per process) and the
     *  results are derived by linearizing in response order, which always respects real-time order. */
    private static List<HistoryOp> RandomHistory(int ops, int processes, Model prototype)
    {
        var history = new List<HistoryOp>();
        for (int i = 0; i < ops; i++)
        {
            int kind = (int)(SplitMix64.Next(ref state) % 2);
            int arg = 1 + (int)(SplitMix64.Next(ref state) % 3);
            history.Add(Op(i, i % processes, kind, arg, 0, -1, -1));
        }
        var invoked = new bool[ops];
        var responded = new bool[ops];
        var cursor = new int[processes];
        var busy = new bool[processes];
        for (int p = 0; p < processes; p++) cursor[p] = p;
        for (int step = 0; step < 2 * ops; step++)
        {
            ulong roll = SplitMix64.Next(ref state);
            var invokable = new List<int>();
            for (int p = 0; p < processes; p++)
            {
                if (cursor[p] < ops && !busy[p]) invokable.Add(cursor[p]);
            }
            var respondable = new List<int>();
            for (int i = 0; i < ops; i++)
            {
                if (invoked[i] && !responded[i]) respondable.Add(i);
            }
            if (invokable.Count > 0 && (respondable.Count == 0 || roll % 3 == 0))
            {
                int pick = invokable[(int)(roll % (ulong)invokable.Count)];
                history[pick].Invoke = step;
                invoked[pick] = true;
                busy[pick % processes] = true;
                cursor[pick % processes] += processes;
            }
            else
            {
                int pick = respondable[(int)(roll % (ulong)respondable.Count)];
                history[pick].Response = step;
                responded[pick] = true;
                busy[pick % processes] = false;
            }
        }
        var byResponse = new List<int>();
        for (int i = 0; i < ops; i++)
        {
            Check(responded[i], "generated history completed");
            byResponse.Add(i);
        }
        byResponse.Sort((a, b) => history[a].Response.CompareTo(history[b].Response));
        Model model = prototype.Copy();
        foreach (int index in byResponse)
        {
            model.Apply(history[index].Kind, history[index].Arg, out int result);
            history[index].Result = result;
        }
        if (SplitMix64.Next(ref state) % 3 == 0)
        {
            int victim = (int)(SplitMix64.Next(ref state) % (ulong)ops);
            history[victim].Result = SplitMix64.Next(ref state) % 2 == 0 ? history[victim].Result + 1 : 0;
        }
        return history;
    }

    public static void Main()
    {
        var registerPrototype = new RegisterModel();
        var queuePrototype = new QueueModel();

        // Register, linearizable: the write completes before the read starts.
        var linear = new List<HistoryOp>
        {
            Op(0, 0, 1, 7, 0, 0, 1),
            Op(1, 1, 0, 0, 7, 2, 3),
        };
        var checker = new LinearizabilityChecker(linear, registerPrototype);
        Check(checker.IsLinearizable(), "register: write then read");
        Check(WitnessValid(linear, registerPrototype, checker.Witness), "register witness replays");

        // Register, non-linearizable: a stale read that starts after the write completed.
        var stale = new List<HistoryOp>
        {
            Op(0, 0, 1, 1, 0, 0, 1),
            Op(1, 1, 0, 0, 0, 2, 3),
        };
        Check(!new LinearizabilityChecker(stale, registerPrototype).IsLinearizable(),
            "register: stale read rejected");

        // Register, concurrent: two overlapping reads/writes admit a linearization.
        var overlapping = new List<HistoryOp>
        {
            Op(0, 0, 1, 5, 0, 0, 4),
            Op(1, 1, 0, 0, 0, 1, 5),
            Op(2, 1, 0, 0, 5, 6, 7),
        };
        var overlappingChecker = new LinearizabilityChecker(overlapping, registerPrototype);
        Check(overlappingChecker.IsLinearizable(), "register: overlapping admits an order");
        Check(WitnessValid(overlapping, registerPrototype, overlappingChecker.Witness),
            "overlapping witness replays");

        // Register, non-linearizable: two reads disagree with the only possible serial order.
        var disagree = new List<HistoryOp>
        {
            Op(0, 0, 1, 3, 0, 0, 1),
            Op(1, 0, 0, 0, 3, 2, 3),
            Op(2, 1, 0, 0, 0, 2, 5),
        };
        Check(!new LinearizabilityChecker(disagree, registerPrototype).IsLinearizable(),
            "register: read 3 then read 0 while a write completed before both is rejected");

        // Queue, linearizable: concurrent enqueues, dequeues in the admitted order.
        var queueHistory = new List<HistoryOp>
        {
            Op(0, 0, 1, 1, 0, 0, 3),
            Op(1, 1, 1, 2, 0, 1, 4),
            Op(2, 0, 0, 0, 1, 5, 6),
            Op(3, 1, 0, 0, 2, 7, 8),
        };
        var queueChecker = new LinearizabilityChecker(queueHistory, queuePrototype);
        Check(queueChecker.IsLinearizable(), "queue: FIFO order matches enqueue order");
        Check(WitnessValid(queueHistory, queuePrototype, queueChecker.Witness), "queue witness replays");

        // Queue, non-linearizable: dequeue returns a value that was never enqueued.
        var impossible = new List<HistoryOp>
        {
            Op(0, 0, 1, 1, 0, 0, 1),
            Op(1, 1, 0, 0, 2, 2, 3),
        };
        Check(!new LinearizabilityChecker(impossible, queuePrototype).IsLinearizable(),
            "queue: dequeue of a missing value rejected");

        // Queue: two concurrent enqueues may be linearized in either order, so both dequeue orders
        // are legal - this is a positive case, not the negative one it looks like.
        var swapped = new List<HistoryOp>
        {
            Op(0, 0, 1, 1, 0, 0, 1),
            Op(1, 1, 1, 2, 0, 0, 1),
            Op(2, 0, 0, 0, 2, 2, 3),
            Op(3, 1, 0, 0, 1, 2, 3),
        };
        var swappedChecker = new LinearizabilityChecker(swapped, queuePrototype);
        Check(swappedChecker.IsLinearizable(), "queue: concurrent enqueues may swap order");
        Check(WitnessValid(swapped, queuePrototype, swappedChecker.Witness), "swapped witness replays");

        // Queue, non-linearizable: the enqueues are ordered, so the dequeue must return the first one.
        var fifoViolation = new List<HistoryOp>
        {
            Op(0, 0, 1, 1, 0, 0, 1),
            Op(1, 0, 1, 2, 0, 2, 3),
            Op(2, 0, 0, 0, 2, 4, 5),
        };
        Check(!new LinearizabilityChecker(fifoViolation, queuePrototype).IsLinearizable(),
            "queue: dequeue returning the second value rejected");

        // Pending operations: a read that never returned may take any value, a write still counts.
        var pendingHistory = new List<HistoryOp>
        {
            Op(0, 0, 1, 9, 0, 0, 1),
            Op(1, 1, 0, 0, 0, 2, -1),
        };
        var pendingChecker = new LinearizabilityChecker(pendingHistory, registerPrototype);
        Check(pendingChecker.IsLinearizable(), "pending read is unconstrained");
        Check(WitnessValid(pendingHistory, registerPrototype, pendingChecker.Witness),
            "pending witness replays");

        var pendingWrite = new List<HistoryOp>
        {
            Op(0, 0, 1, 4, 0, 0, -1),
            Op(1, 1, 0, 0, 4, 1, 2),
        };
        Check(new LinearizabilityChecker(pendingWrite, registerPrototype).IsLinearizable(),
            "pending write still takes effect");

        // Degenerate histories.
        Check(new LinearizabilityChecker(new List<HistoryOp>(), registerPrototype).IsLinearizable(),
            "empty history is linearizable");
        Check(new LinearizabilityChecker(new List<HistoryOp>(), queuePrototype).IsLinearizable(),
            "empty queue history is linearizable");
        var writesOnly = new List<HistoryOp>
        {
            Op(0, 0, 1, 1, 0, 0, 1),
            Op(1, 1, 1, 2, 0, 2, 3),
        };
        Check(new LinearizabilityChecker(writesOnly, registerPrototype).IsLinearizable(),
            "writes only are always linearizable");
        var allPending = new List<HistoryOp>
        {
            Op(0, 0, 0, 0, 0, 0, -1),
            Op(1, 1, 1, 1, 0, 1, -1),
            Op(2, 0, 0, 0, 0, 2, -1),
        };
        Check(new LinearizabilityChecker(allPending, registerPrototype).IsLinearizable(),
            "all pending reads see the initial value");

        // Random small histories: the pruned search must agree with the brute-force oracle.
        int trials = 0;
        int linearizable = 0;
        for (int trial = 0; trial < 40; trial++)
        {
            ulong roll = SplitMix64.Next(ref state);
            Model prototype = roll % 4 == 0 ? queuePrototype : registerPrototype;
            List<HistoryOp> history = RandomHistory(5 + (int)(roll % 2), 2 + (int)((roll >> 4) % 2), prototype);
            var fast = new LinearizabilityChecker(history, prototype);
            bool fastVerdict = fast.IsLinearizable();
            bool slowVerdict = LinearizabilityChecker.BruteForce(history, prototype);
            Check(fastVerdict == slowVerdict, "search agrees with brute force on trial " + trial);
            if (fastVerdict)
            {
                Check(WitnessValid(history, prototype, fast.Witness), "random witness replays " + trial);
                linearizable++;
            }
            trials++;
        }
        Check(trials == 40 && linearizable > 0 && linearizable < trials,
            "the random histories covered both verdicts: " + linearizable + "/" + trials);

        // A longer interleaved register history with a genuine race: still decidable.
        var longHistory = new List<HistoryOp>();
        int clock = 0;
        for (int i = 0; i < 9; i++)
        {
            int kind = i % 3 == 0 ? 1 : 0;
            longHistory.Add(Op(i, i % 3, kind, i + 1, kind == 0 ? i + 1 : 0, clock, clock + 1 + i % 2));
            clock += 1 + i % 2;
        }
        var longChecker = new LinearizabilityChecker(longHistory, registerPrototype);
        bool longVerdict = longChecker.IsLinearizable();
        Check(longVerdict == LinearizabilityChecker.BruteForce(longHistory, registerPrototype),
            "long history agrees with the oracle");
        Check(!longVerdict || WitnessValid(longHistory, registerPrototype, longChecker.Witness),
            "long witness replays");

        Console.WriteLine("  randomTrials=" + trials + " linearizable=" + linearizable
            + " searchSteps=" + new LinearizabilityChecker(queueHistory, queuePrototype).Steps);
        Console.WriteLine("  registerHistory=2 ops queueHistory=4 ops longHistory=" + longHistory.Count
            + " ops verdict=" + longVerdict);
        Console.WriteLine("PASS");
    }
}
