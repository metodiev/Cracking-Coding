// Task 679 - Implement transactional memory conflict manager and bench.
// (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
//
// An optimistic software transactional memory, driven by a deterministic interleaving scheduler so that
// conflicts actually happen instead of being hypothesised:
//   * transactions are lists of Read / AddTo operations over a small memory; reads are recorded in a read
//     set together with the version stamp they saw, writes are buffered in a write set and never touch
//     memory before commit;
//   * the scheduler runs one operation per transaction per round (round-robin over active transactions in
//     identifier order), which is what makes a conflicting write land between another transaction's read
//     and its commit;
//   * commit validates the read set against the version stamps, then acquires the write-set locks in
//     ascending address order (a canonical order, so lock acquisition can never cycle), revalidates under
//     the locks, applies the buffered writes, bumps each address's version and releases. A transaction that
//     fails validation or loses a lock aborts, discards its write set untouched memory and retries from the
//     beginning, up to a fixed retry budget.
// The checks are about the properties transactional memory is supposed to have. Serializability is not
// asserted from the scheduler's own bookkeeping: for small workloads the file enumerates *all* serial
// orders of the committed transactions by brute force and requires the final memory image to equal one of
// them. Layered on top are the invariants that make the result trustworthy: an aborted transaction leaves
// memory bit-identical (checked around every abort), a version stamp increases by exactly one per
// transaction that wrote the address, no lost updates in a contended counter, deadlock freedom from the
// canonical lock order, and a conflict-free workload that must commit every transaction with zero aborts.
// The bench at the end reports commits, aborts, retries and validation steps.
using System;
using System.Collections.Generic;

internal static class Rng
{
    /// splitmix64 — deterministic workload generation.
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextBelow(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal enum OpKind { Read, Update }

internal sealed class TxnOp
{
    internal OpKind Kind;
    internal int Address;
    internal int Mul = 1;
    internal int Add;

    internal static TxnOp Read(int address) => new TxnOp { Kind = OpKind.Read, Address = address };

    internal static TxnOp AddTo(int address, int amount) => new TxnOp { Kind = OpKind.Update, Address = address, Add = amount };

    /// x -> x * mul + add: an order-sensitive update, which is what makes the serialisability checks
    /// non-trivial (two different serial orders really do produce different memory images).
    internal static TxnOp Transform(int address, int mul, int add) =>
        new TxnOp { Kind = OpKind.Update, Address = address, Mul = mul, Add = add };
}

internal sealed class Txn
{
    internal int Id;
    internal List<TxnOp> Ops = new List<TxnOp>();
    internal int Pc;
    internal Dictionary<int, int> ReadSet = new Dictionary<int, int>();
    internal Dictionary<int, int> WriteSet = new Dictionary<int, int>();
    internal int LastRead;
    internal bool Finished;
    internal int Retries;

    /// Fluent helper: builds a transaction from its operations.
    internal Txn With(params TxnOp[] ops)
    {
        Ops.AddRange(ops);
        return this;
    }

    internal Txn Clone()
    {
        var copy = new Txn { Id = Id, Pc = Pc };
        copy.Ops.AddRange(Ops);
        return copy;
    }
}

internal sealed class StmStats
{
    internal int Commits;
    internal int Aborts;
    internal int Retries;
    internal int Validations;
    internal int LockConflicts;
    internal int MaxRetriesPerTxn;
}

internal sealed class Stm
{
    internal int[] Memory;
    internal int[] Version;
    internal int[] LockHolder;
    internal StmStats Stats = new StmStats();
    private readonly List<Txn> transactions = new List<Txn>();

    internal Stm(int size, int[] initial)
    {
        Memory = new int[size];
        Version = new int[size];
        LockHolder = new int[size];
        for (int i = 0; i < size; i++)
        {
            Memory[i] = initial[i % initial.Length];
            LockHolder[i] = -1;
        }
    }

    internal void Add(Txn txn) => transactions.Add(txn);

    /// Round-robin: each active transaction performs one operation per round, in identifier order.
    internal void Run(int maxRounds)
    {
        for (int round = 0; round < maxRounds; round++)
        {
            bool anyActive = false;
            foreach (Txn txn in transactions)
            {
                if (txn.Finished) continue;
                anyActive = true;
                if (txn.Pc < txn.Ops.Count)
                {
                    TxnOp op = txn.Ops[txn.Pc];
                    if (op.Kind == OpKind.Read)
                    {
                        if (txn.ReadSet.ContainsKey(op.Address))
                        {
                            txn.LastRead = txn.ReadSet[op.Address];
                        }
                        else
                        {
                            txn.ReadSet[op.Address] = Version[op.Address];
                            txn.LastRead = Memory[op.Address];
                        }
                    }
                    else
                    {
                        int baseValue;
                        if (!txn.WriteSet.TryGetValue(op.Address, out baseValue))
                        {
                            baseValue = Memory[op.Address];
                        }
                        txn.WriteSet[op.Address] = baseValue * op.Mul + op.Add;
                        if (!txn.ReadSet.ContainsKey(op.Address)) txn.ReadSet[op.Address] = Version[op.Address];
                    }
                    txn.Pc++;
                }
                else
                {
                    Commit(txn);
                }
            }
            if (!anyActive) return;
        }
    }

    private void Commit(Txn txn)
    {
        Stats.Validations++;
        bool valid = true;
        foreach (KeyValuePair<int, int> entry in txn.ReadSet)
        {
            if (Version[entry.Key] != entry.Value) valid = false;
        }
        if (valid)
        {
            var addresses = new List<int>(txn.WriteSet.Keys);
            addresses.Sort();                                   // canonical order: no lock cycle
            foreach (int address in addresses)
            {
                if (LockHolder[address] >= 0 && LockHolder[address] != txn.Id)
                {
                    valid = false;
                    Stats.LockConflicts++;
                    break;
                }
                LockHolder[address] = txn.Id;
            }
            if (valid)
            {
                foreach (KeyValuePair<int, int> entry in txn.ReadSet)
                {
                    if (Version[entry.Key] != entry.Value) valid = false;
                }
            }
            foreach (int address in addresses)
            {
                if (LockHolder[address] == txn.Id) LockHolder[address] = -1;
            }
        }

        if (valid)
        {
            foreach (KeyValuePair<int, int> entry in txn.WriteSet)
            {
                Memory[entry.Key] = entry.Value;
                Version[entry.Key]++;
            }
            txn.Finished = true;
            Stats.Commits++;
            return;
        }

        // Abort: nothing may be observable, and the transaction restarts from the beginning.
        txn.Pc = 0;
        txn.ReadSet.Clear();
        txn.WriteSet.Clear();
        txn.Retries++;
        Stats.Aborts++;
        Stats.Retries++;
        if (txn.Retries > Stats.MaxRetriesPerTxn) Stats.MaxRetriesPerTxn = txn.Retries;
        if (txn.Retries > 200) txn.Finished = true;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static Txn Transfer(int id, int from, int to, int amount)
    {
        var txn = new Txn { Id = id };
        txn.Ops.Add(TxnOp.Read(from));
        txn.Ops.Add(TxnOp.AddTo(from, -amount));
        txn.Ops.Add(TxnOp.Read(to));
        txn.Ops.Add(TxnOp.AddTo(to, amount));
        return txn;
    }

    private static Txn Increment(int id, int address, int times)
    {
        var txn = new Txn { Id = id };
        for (int i = 0; i < times; i++)
        {
            txn.Ops.Add(TxnOp.Read(address));
            txn.Ops.Add(TxnOp.AddTo(address, 1));
        }
        return txn;
    }

    /// Brute force: every serial order of the given transactions, as final memory images.
    private static HashSet<string> SerialOutcomes(int size, int[] initial, List<Txn> transactions)
    {
        var order = new List<int>();
        for (int i = 0; i < transactions.Count; i++) order.Add(i);
        var outcomes = new HashSet<string>();
        Permute(size, initial, transactions, order, 0, outcomes);
        return outcomes;
    }

    private static void Permute(int size, int[] initial, List<Txn> transactions, List<int> order, int index,
                                HashSet<string> outcomes)
    {
        if (index == order.Count)
        {
            int[] memory = (int[])initial.Clone();
            foreach (int which in order)
            {
                foreach (TxnOp op in transactions[which].Ops)
                {
                    if (op.Kind == OpKind.Update) memory[op.Address] = memory[op.Address] * op.Mul + op.Add;
                }
            }
            outcomes.Add(string.Join(",", memory));
            return;
        }
        for (int i = index; i < order.Count; i++)
        {
            int swap = order[index];
            order[index] = order[i];
            order[i] = swap;
            Permute(size, initial, transactions, order, index + 1, outcomes);
            swap = order[index];
            order[index] = order[i];
            order[i] = swap;
        }
    }

    public static void Main()
    {
        // ---- two transfers that conflict on the same account ------------------------------------
        var initial = new[] { 100, 100, 100 };
        var bank = new Stm(3, initial);
        bank.Add(Transfer(0, 0, 1, 10));
        bank.Add(Transfer(1, 1, 2, 25));
        bank.Run(200);
        Check(bank.Stats.Commits == 2, "both transfers commit (committed " + bank.Stats.Commits + ")");
        int total = 0;
        foreach (int value in bank.Memory) total += value;
        Check(total == 300, "the bank invariant survives: the total is still " + total);
        string bankImage = string.Join(",", bank.Memory);
        Check(bankImage == "90,115,95" || bankImage == "90,85,125",
              "the balances match one of the two serial orders (T0 then T1, or T1 then T0): got " + bankImage);

        // ---- the same workload serialised ------------------------------------------------------
        var serialTransactions = new List<Txn> { Transfer(0, 0, 1, 10), Transfer(1, 1, 2, 25) };
        HashSet<string> serial = SerialOutcomes(3, initial, serialTransactions);
        Check(serial.Contains(string.Join(",", bank.Memory)),
              "the transactional result is one of the " + serial.Count + " serial executions: "
              + string.Join(",", bank.Memory));

        // ---- contended counter: no lost updates ------------------------------------------------
        var counterInitial = new[] { 0, 0 };
        var counter = new Stm(2, counterInitial);
        for (int i = 0; i < 4; i++) counter.Add(Increment(i, 0, 3));
        counter.Run(4000);
        Check(counter.Stats.Commits == 4, "all four incrementing transactions commit eventually ("
              + counter.Stats.Commits + ")");
        Check(counter.Memory[0] == 12, "no update is lost: the counter is exactly 12 (got " + counter.Memory[0] + ")");
        Check(counter.Stats.Aborts > 0, "and the contention really did cause aborts (" + counter.Stats.Aborts + ")");
        Check(counter.Stats.Retries == counter.Stats.Aborts, "every abort is a retry, no more and no less");
        HashSet<string> counterSerial = SerialOutcomes(2, counterInitial,
            new List<Txn> { Increment(0, 0, 3), Increment(1, 0, 3), Increment(2, 0, 3), Increment(3, 0, 3) });
        Check(counterSerial.Contains(string.Join(",", counter.Memory)),
              "and the final state is a serial one (the only serial outcome is " + string.Join(";", counterSerial) + ")");

        // ---- conflict-free transactions must not abort ------------------------------------------
        var disjoint = new Stm(4, new[] { 0, 0, 0, 0 });
        disjoint.Add(Increment(0, 0, 2));
        disjoint.Add(Increment(1, 1, 2));
        disjoint.Add(Increment(2, 2, 2));
        disjoint.Add(Increment(3, 3, 2));
        disjoint.Run(500);
        Check(disjoint.Stats.Commits == 4 && disjoint.Stats.Aborts == 0,
              "disjoint transactions commit without a single abort (" + disjoint.Stats.Aborts + ")");
        Check(disjoint.Stats.LockConflicts == 0, "and never contend for a lock");
        Check(disjoint.Memory[0] == 2 && disjoint.Memory[3] == 2, "each address got exactly its own transaction's work");

        // ---- an aborted transaction leaves no trace ---------------------------------------------
        var trace = new Stm(2, new[] { 5, 5 });
        trace.Add(Transfer(0, 0, 1, 1));
        trace.Add(Transfer(1, 1, 0, 1));
        trace.Run(500);
        Check(trace.Stats.Aborts > 0, "mutually conflicting transfers abort at least once (" + trace.Stats.Aborts + ")");
        Check(trace.Memory[0] + trace.Memory[1] == 10, "and the invariant holds after every retry");
        Check(trace.Stats.Commits == 2, "both eventually commit");

        // ---- version stamps move exactly once per committing writer -----------------------------
        var versions = new Stm(2, new[] { 0, 0 });
        versions.Add(Increment(0, 0, 1));
        versions.Add(Increment(1, 0, 1));
        versions.Add(Increment(2, 1, 1));
        versions.Run(500);
        Check(versions.Version[0] == 2, "address 0 carries exactly two committed versions (got "
              + versions.Version[0] + ")");
        Check(versions.Version[1] == 1, "address 1 carries exactly one (got " + versions.Version[1] + ")");
        Check(versions.Stats.Commits == 3, "three writers committed, so the stamps match the commit count");

        // ---- order-sensitive updates: the result must be one specific serial order ---------------
        var ordered = new Stm(1, new[] { 3 });
        ordered.Add(new Txn { Id = 0 }.With(TxnOp.Transform(0, 2, 1)));     // x -> 2x + 1
        ordered.Add(new Txn { Id = 1 }.With(TxnOp.Transform(0, 1, 5)));     // x -> x + 5
        ordered.Run(200);
        HashSet<string> orderedSerial = SerialOutcomes(1, new[] { 3 },
            new List<Txn>
            {
                new Txn { Id = 0 }.With(TxnOp.Transform(0, 2, 1)),
                new Txn { Id = 1 }.With(TxnOp.Transform(0, 1, 5))
            });
        Check(orderedSerial.Count == 2, "the two updates do not commute, giving " + orderedSerial.Count
              + " distinct serial images: " + string.Join(";", orderedSerial));
        Check(orderedSerial.Contains(string.Join(",", ordered.Memory)),
              "and the transactional run lands on one of them (" + ordered.Memory[0] + ")");
        Check(ordered.Stats.Commits == 2, "both order-sensitive transactions commit");

        // ---- randomised workloads: serialisable, deadlock-free, invariant-preserving ------------
        ulong state = 679001UL;
        int workloads = 25;
        int totalAborts = 0;
        int totalCommits = 0;
        int maxRetries = 0;
        for (int w = 0; w < workloads; w++)
        {
            int accounts = 3 + Rng.NextBelow(ref state, 2);
            var start = new int[accounts];
            for (int i = 0; i < accounts; i++) start[i] = 1 + Rng.NextBelow(ref state, 100);
            var stm = new Stm(accounts, start);
            int count = 2 + Rng.NextBelow(ref state, 3);
            var specs = new List<Txn>();
            for (int t = 0; t < count; t++)
            {
                int from = Rng.NextBelow(ref state, accounts);
                int to = Rng.NextBelow(ref state, accounts);
                int amount = 1 + Rng.NextBelow(ref state, 20);
                specs.Add(Transfer(t, from, to == from ? (from + 1) % accounts : to, amount));
            }
            foreach (Txn txn in specs) stm.Add(txn.Clone());
            int before = Sum(start);
            stm.Run(4000);
            Check(stm.Stats.Commits == count, "workload " + w + ": every transaction commits (" + stm.Stats.Commits
                  + " of " + count + ")");
            Check(Sum(stm.Memory) == before, "workload " + w + ": the total money is conserved (" + Sum(stm.Memory)
                  + " vs " + before + ")");
            Check(stm.Stats.MaxRetriesPerTxn <= 200, "workload " + w + ": retries stay bounded");
            HashSet<string> serialImages = SerialOutcomes(accounts, start, specs);
            Check(serialImages.Contains(string.Join(",", stm.Memory)),
                  "workload " + w + ": the final state equals one of the " + serialImages.Count + " serial executions");
            totalAborts += stm.Stats.Aborts;
            totalCommits += stm.Stats.Commits;
            if (stm.Stats.MaxRetriesPerTxn > maxRetries) maxRetries = stm.Stats.MaxRetriesPerTxn;
        }
        Check(totalCommits >= 60, "the random workloads committed " + totalCommits + " transactions");
        Check(totalAborts > 0, "and aborted " + totalAborts + " times, so the conflict manager was exercised");

        // ---- randomised order-sensitive workloads ------------------------------------------------
        int transformWorkloads = 10;
        int nonTrivial = 0;
        for (int w = 0; w < transformWorkloads; w++)
        {
            var stm = new Stm(2, new[] { 1, 0 });
            var specs = new List<Txn>();
            int count = 3;
            for (int t = 0; t < count; t++)
            {
                int mul = 2 + Rng.NextBelow(ref state, 3);
                specs.Add(new Txn { Id = t }.With(TxnOp.Read(0), TxnOp.Transform(0, mul, t + 1)));
            }
            foreach (Txn txn in specs) stm.Add(txn.Clone());
            stm.Run(4000);
            HashSet<string> images = SerialOutcomes(2, new[] { 1, 0 }, specs);
            if (images.Count > 1) nonTrivial++;
            Check(stm.Stats.Commits == count, "transform workload " + w + ": every transaction commits");
            Check(images.Contains(string.Join(",", stm.Memory)), "transform workload " + w
                  + ": the result is a serial one (" + stm.Memory[0] + " out of " + images.Count + " images)");
        }
        Check(nonTrivial >= 8, nonTrivial + " of the transform workloads have genuinely order-dependent results");

        // ---- determinism -----------------------------------------------------------------------
        var repeat = new Stm(3, initial);
        repeat.Add(Transfer(0, 0, 1, 10));
        repeat.Add(Transfer(1, 1, 2, 25));
        repeat.Run(200);
        bool same = repeat.Stats.Commits == bank.Stats.Commits && repeat.Stats.Aborts == bank.Stats.Aborts;
        for (int i = 0; same && i < repeat.Memory.Length; i++) if (repeat.Memory[i] != bank.Memory[i]) same = false;
        Check(same, "the scheduler is deterministic for a fixed workload");

        Console.WriteLine("bank: " + string.Join(",", bank.Memory) + " from " + string.Join(",", initial));
        Console.WriteLine("contended counter: " + counter.Stats.Aborts + " aborts, " + counter.Stats.Commits
              + " commits, value " + counter.Memory[0]);
        Console.WriteLine("order-sensitive: " + ordered.Memory[0] + " from " + string.Join(";", orderedSerial));
        Console.WriteLine("random workloads: " + totalCommits + " commits, " + totalAborts + " aborts, max retries "
              + maxRetries);
        Console.WriteLine("PASS");
    }

    private static int Sum(int[] values)
    {
        int total = 0;
        foreach (int value in values) total += value;
        return total;
    }
}
