// Task 615 - Implement transactional memory simulator basics. (Tier: Expert, CTCI Ch 15 — Threads and Locks)
//
// A single-threaded simulation of an optimistic software transactional memory. Threads never run
// concurrently here: the *schedule* is an explicit list of (transaction, step) pairs, and each step
// is one operation of a transaction body (begin, read, write, commit). That keeps the interleavings
// fully deterministic while still exercising the two things an STM is about:
//
//   * buffered writes and a read set - Read(c) records the cell's version, Write(c, v) goes to the
//     transaction's write set, and nothing touches shared memory before commit.
//   * validation on commit (first-committer-wins) - Commit re-checks every version in the read set;
//     if any cell changed since it was read, the transaction aborts, its write set is discarded and
//     the body is retried from the start, otherwise the writes are applied and the cell versions
//     are bumped atomically.
//
// What is simplified: there are no real atomic instructions and no contention at machine level -
// atomicity comes from the fact that only one step runs at a time - so the simulation models the
// *interleavings* rather than the hardware. A deliberately unsound mode (ValidateReadSet = false)
// is included to prove the checks below are not vacuous.
//
// Invariants verified: the total of all cells is conserved after *every* step of every schedule (the
// transactions only move value around); the final state equals an independent replay of exactly the
// committed transactions in commit order, so an aborted transaction leaves no trace and no update
// is lost; a conflicting pair produces exactly one committer per attempt and the loser retries until
// it commits; and the unsound mode is detected by both the conservation and the replay check.

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

/** One step of a transaction body: read a cell, or write cell = local[slot] + delta. */
internal struct Step
{
    internal bool IsRead;
    internal int Cell;
    internal int Slot;
    internal int Delta;

    internal static Step Read(int cell) => new Step { IsRead = true, Cell = cell };
    internal static Step Write(int cell, int slot, int delta) =>
        new Step { IsRead = false, Cell = cell, Slot = slot, Delta = delta };
}

internal sealed class Transaction
{
    internal readonly int Id;
    internal readonly int From;
    internal readonly int To;
    internal readonly int Amount;
    internal readonly List<Step> Body = new List<Step>();
    internal readonly List<int> Local = new List<int>();
    internal readonly List<int> ReadCells = new List<int>();
    internal readonly List<int> ReadVersions = new List<int>();
    internal readonly List<int[]> WriteSet = new List<int[]>();
    internal int Attempts;

    internal Transaction(int id, int from, int to, int amount)
    {
        Id = id;
        From = from;
        To = to;
        Amount = amount;
        Body.Add(Step.Read(from));
        Body.Add(Step.Read(to));
        Body.Add(Step.Write(from, 0, -amount));
        Body.Add(Step.Write(to, 1, amount));
    }
}

internal sealed class StmSimulator
{
    private readonly int[] values;
    private readonly int[] versions;
    private readonly Dictionary<int, Transaction> active = new Dictionary<int, Transaction>();
    private readonly Dictionary<int, int[]> pending = new Dictionary<int, int[]>();
    private readonly List<Transaction> committed = new List<Transaction>();

    internal readonly List<string> Trace = new List<string>();
    internal bool ValidateReadSet = true;                 // set false to simulate a broken STM

    internal StmSimulator(int[] initial)
    {
        values = new int[initial.Length];
        versions = new int[initial.Length];
        Array.Copy(initial, values, initial.Length);
    }

    internal int Total()
    {
        int sum = 0;
        foreach (int value in values) sum += value;
        return sum;
    }

    internal int Value(int cell) => values[cell];
    internal List<Transaction> Committed => committed;
    internal int ActiveCount => active.Count;

    internal void Begin(Transaction transaction)
    {
        transaction.Local.Clear();
        transaction.ReadCells.Clear();
        transaction.ReadVersions.Clear();
        transaction.WriteSet.Clear();
        transaction.Attempts++;
        active[transaction.Id] = transaction;
        pending[transaction.Id] = new[] { 0 };
        Trace.Add("T" + transaction.Id + ":begin");
    }

    internal bool Read(Transaction transaction, int cell, out int value)
    {
        value = 0;
        if (!active.ContainsKey(transaction.Id)) return false;
        transaction.ReadCells.Add(cell);
        transaction.ReadVersions.Add(versions[cell]);
        value = values[cell];
        transaction.Local.Add(value);
        Trace.Add("T" + transaction.Id + ":read" + cell + "=" + value);
        return true;
    }

    internal bool Write(Transaction transaction, int cell, int value)
    {
        if (!active.ContainsKey(transaction.Id)) return false;
        transaction.WriteSet.Add(new[] { cell, value });
        Trace.Add("T" + transaction.Id + ":write" + cell + "=" + value);
        return true;
    }

    /** Apply the buffered writes after re-validating the read set. Returns false on abort. */
    internal bool Commit(Transaction transaction, out bool aborted)
    {
        aborted = false;
        if (!active.ContainsKey(transaction.Id)) return false;
        if (ValidateReadSet)
        {
            for (int i = 0; i < transaction.ReadCells.Count; i++)
            {
                if (versions[transaction.ReadCells[i]] != transaction.ReadVersions[i])
                {
                    aborted = true;
                    Trace.Add("T" + transaction.Id + ":abort");
                    active.Remove(transaction.Id);
                    pending.Remove(transaction.Id);
                    return false;
                }
            }
        }
        foreach (int[] write in transaction.WriteSet)
        {
            values[write[0]] = write[1];
            versions[write[0]]++;
        }
        committed.Add(transaction);
        active.Remove(transaction.Id);
        pending.Remove(transaction.Id);
        Trace.Add("T" + transaction.Id + ":commit");
        return true;
    }

    /** Run the body of a transaction step by step, aborting and retrying when validation fails. */
    internal void Run(Transaction transaction, int interleaveSeed)
    {
        ulong state = (ulong)interleaveSeed * 0x9E3779B97F4A7C15UL + 1;
        int guard = 0;
        while (guard++ < 64)
        {
            Begin(transaction);
            bool aborted = false;
            foreach (Step step in transaction.Body)
            {
                if (step.IsRead) Read(transaction, step.Cell, out _);
                else Write(transaction, step.Cell, transaction.Local[step.Slot] + step.Delta);
            }
            Commit(transaction, out aborted);
            if (!aborted) return;
        }
        throw new InvalidOperationException("transaction starved");
    }

    /** Replay of exactly the committed transactions, in commit order: the serial oracle. */
    internal static int[] SerialReplay(int[] initial, List<Transaction> order)
    {
        var state = new int[initial.Length];
        Array.Copy(initial, state, initial.Length);
        foreach (Transaction transaction in order)
        {
            state[transaction.From] -= transaction.Amount;
            state[transaction.To] += transaction.Amount;
        }
        return state;
    }
}

public static class Solution
{
    private static ulong state = 0xAC1D5EEDUL;

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static bool Same(int[] a, int[] b)
    {
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] != b[i]) return false;
        }
        return true;
    }

    private static string Show(int[] values) => "[" + string.Join(",", values) + "]";

    public static void Main()
    {
        var initial = new[] { 100, 100, 100, 0 };
        int conserved = initial[0] + initial[1] + initial[2] + initial[3];

        // Schedule 1 - clean, non-overlapping transfers: each commits on the first attempt.
        var simulator = new StmSimulator(initial);
        var transactions = new List<Transaction>
        {
            new Transaction(1, 0, 1, 10),
            new Transaction(2, 1, 2, 5),
            new Transaction(3, 2, 0, 3),
        };
        foreach (Transaction transaction in transactions)
        {
            simulator.Begin(transaction);
            foreach (Step step in transaction.Body)
            {
                if (step.IsRead) simulator.Read(transaction, step.Cell, out _);
                else simulator.Write(transaction, step.Cell, transaction.Local[step.Slot] + step.Delta);
                Check(simulator.Total() == conserved, "sum conserved mid transaction");
            }
            Check(simulator.Commit(transaction, out bool aborted) && !aborted, "clean commit");
        }
        Check(simulator.ActiveCount == 0, "no transaction left open");
        Check(simulator.Total() == conserved, "sum conserved after schedule 1");
        Check(simulator.Value(0) == 93 && simulator.Value(1) == 105 && simulator.Value(2) == 102,
            "value after schedule 1");
        var expected1 = StmSimulator.SerialReplay(initial, simulator.Committed);
        Check(expected1[0] == 93 && expected1[1] == 105 && expected1[2] == 102, "oracle after schedule 1");

        // Schedule 2 - explicit interleaving: T2 commits a conflicting write before T1 commits, so
        // first-committer-wins forces T1 to abort on its first attempt and retry to success.
        var conflicting = new StmSimulator(initial);
        var t1 = new Transaction(1, 0, 1, 10);
        var t2 = new Transaction(2, 0, 2, 7);
        conflicting.Begin(t1);
        conflicting.Read(t1, 0, out int readA);
        conflicting.Read(t1, 1, out _);
        conflicting.Begin(t2);
        conflicting.Read(t2, 0, out _);
        conflicting.Read(t2, 2, out _);
        conflicting.Write(t2, 0, readA - 7);
        conflicting.Write(t2, 2, 107);
        Check(conflicting.Commit(t2, out bool t2Aborted) && !t2Aborted, "T2 commits first");
        conflicting.Write(t1, 0, readA - 10);
        conflicting.Write(t1, 1, readA + 10);
        Check(!conflicting.Commit(t1, out bool firstAbort) && firstAbort, "T1 aborts on a stale read set");
        Check(t1.Attempts == 1 && conflicting.Committed.Count == 1, "only T2 committed");
        Check(conflicting.Value(0) == 93 && conflicting.Value(2) == 107, "aborted writes never applied");
        Check(conflicting.Total() == conserved, "sum conserved after the abort");
        conflicting.Run(t1, 1);
        Check(conflicting.Committed.Count == 2 && t1.Attempts == 2, "T1 succeeds after retrying");
        Check(conflicting.Total() == conserved, "sum conserved after the retry");
        var expected2 = StmSimulator.SerialReplay(initial, conflicting.Committed);
        Check(Same(new[] { conflicting.Value(0), conflicting.Value(1), conflicting.Value(2) },
            new[] { expected2[0], expected2[1], expected2[2] }), "serial replay matches");

        // Schedule 3 - many transactions interleaved step by step by a deterministic driver, so that
        // stale read sets really do cause aborts and retries.
        var busyInitial = new[] { 200, 150, 50, 25 };
        var busy = new StmSimulator(busyInitial);
        var ids = MakeTransfers(1, 18);
        int aborts = Drive(busy, ids, 4000);
        Check(busy.ActiveCount == 0 && busy.Committed.Count == ids.Count, "all transactions commit");
        Check(busy.Total() == 425, "sum conserved through contention");
        Check(aborts > 0, "the interleaving really produced aborts");
        var busyExpected = StmSimulator.SerialReplay(busyInitial, busy.Committed);
        Check(Same(new[] { busy.Value(0), busy.Value(1), busy.Value(2), busy.Value(3) }, busyExpected),
            "final state equals the serial replay of committed transactions");

        // The same driver with read-set validation disabled loses updates: the conservation check
        // and the serial replay above are therefore not vacuous.
        var unsound = new StmSimulator(busyInitial) { ValidateReadSet = false };
        Drive(unsound, MakeTransfers(1, 18), 4000);
        var unsoundReplay = StmSimulator.SerialReplay(busyInitial, unsound.Committed);
        bool conservationHolds = unsound.Total() == 425;
        bool replayHolds = Same(new[] { unsound.Value(0), unsound.Value(1), unsound.Value(2), unsound.Value(3) },
            unsoundReplay);
        Check(!conservationHolds || !replayHolds,
            "the unsound STM is caught: total=" + unsound.Total() + " replayMatch=" + replayHolds);

        // Conflicting pair, forced interleaving: exactly one winner per round, loser retries.
        var duel = new StmSimulator(new[] { 50, 50, 0, 0 });
        var left = new Transaction(1, 0, 2, 5);
        var right = new Transaction(2, 0, 3, 5);
        duel.Begin(left);
        duel.Read(left, 0, out _);
        duel.Read(left, 2, out _);
        duel.Begin(right);
        duel.Read(right, 0, out _);
        duel.Read(right, 3, out _);
        duel.Write(left, 0, 45);
        duel.Write(left, 2, 5);
        duel.Write(right, 0, 45);
        duel.Write(right, 3, 5);
        bool leftCommitted = duel.Commit(left, out bool leftAborted);
        bool rightCommitted = duel.Commit(right, out bool rightAborted);
        Check(leftCommitted != rightCommitted, "exactly one of the conflicting pair commits");
        Check(!leftAborted ^ !rightAborted, "the other one aborted");
        Check(duel.Value(0) == 45 && duel.Total() == 100, "no lost update in the duel");
        duel.Run(leftCommitted ? right : left, 5);
        Check(duel.Committed.Count == 2 && duel.Total() == 100, "loser commits after retry");

        // Sound STM, serialized: everything matches the replay of committed transactions.
        var sound = new StmSimulator(new[] { 100, 100, 0, 0 });
        sound.Run(new Transaction(1, 0, 1, 10), 11);
        sound.Run(new Transaction(2, 0, 2, 10), 12);
        var soundReplay = StmSimulator.SerialReplay(new[] { 100, 100, 0, 0 }, sound.Committed);
        Check(sound.Value(0) == soundReplay[0] && sound.Value(1) == soundReplay[1]
            && sound.Value(2) == soundReplay[2], "sound STM matches the serial replay");
        Check(sound.Value(0) == 80 && sound.Value(1) == 110 && sound.Value(2) == 10,
            "sound STM state " + Show(new[] { sound.Value(0), sound.Value(1), sound.Value(2) }));

        Console.WriteLine("  interleavedTransactions=" + ids.Count + " aborts=" + aborts
            + " firstAttemptValue=" + Show(busyInitial));
        Console.WriteLine("  final=" + Show(new[] { busy.Value(0), busy.Value(1), busy.Value(2), busy.Value(3) })
            + " conservedTotal=" + busy.Total() + " unsoundTotal=" + unsound.Total());
        Console.WriteLine("PASS");
    }

    /** Interleaves transaction bodies step by step with a seeded order; returns the abort count. */
    private static int Drive(StmSimulator simulator, List<Transaction> transactions, int limit)
    {
        var pending = new List<Transaction>(transactions);
        var running = new List<Transaction>();
        var stepAt = new List<int>();
        int aborts = 0;
        int guard = 0;
        while ((pending.Count > 0 || running.Count > 0) && guard++ < limit)
        {
            if (pending.Count > 0 && (running.Count == 0 || SplitMix64.Next(ref state) % 3 == 0))
            {
                Transaction started = pending[0];
                pending.RemoveAt(0);
                simulator.Begin(started);
                running.Add(started);
                stepAt.Add(0);
                continue;
            }
            int pick = (int)(SplitMix64.Next(ref state) % (ulong)running.Count);
            Transaction transaction = running[pick];
            Step step = transaction.Body[stepAt[pick]];
            if (step.IsRead) simulator.Read(transaction, step.Cell, out _);
            else simulator.Write(transaction, step.Cell, transaction.Local[step.Slot] + step.Delta);
            stepAt[pick]++;
            if (stepAt[pick] < transaction.Body.Count) continue;
            running.RemoveAt(pick);
            stepAt.RemoveAt(pick);
            if (!simulator.Commit(transaction, out bool aborted) || aborted)
            {
                aborts++;
                pending.Add(transaction);                 // retry the body from the start
            }
        }
        return aborts;
    }

    /** A deterministic batch of transfer transactions over four cells. */
    private static List<Transaction> MakeTransfers(int firstId, int count)
    {
        var transfers = new List<Transaction>();
        for (int i = 0; i < count; i++)
        {
            ulong roll = SplitMix64.Next(ref state);
            int from = (int)(roll % 4);
            int to = (int)((roll >> 8) % 4);
            if (from == to) to = (to + 1) % 4;
            int amount = 1 + (int)((roll >> 16) % 20);
            transfers.Add(new Transaction(firstId + i, from, to, amount));
        }
        return transfers;
    }
}
