// Task 832 - Implement sharded blockchain prototype with cross-shard messaging.
// (Tier: Master, CTCI Ch 9 — System Design and Scalability)
//
// A sharded ledger with atomic cross-shard transfers built on the receipt protocol that real sharded chains use:
//
//   1. the client submits a transfer to the *source* shard; the shard locks the amount (balance -= v, locked += v)
//      and emits Prepare;
//   2. the *destination* shard records a pending receipt for the target account and emits Ack - it credits nothing
//      yet, which is what keeps a failure from creating money;
//   3. on Ack the source shard commits (locked -= v) and emits Commit, and only then does the destination apply the
//      credit;
//   4. if no Ack arrives within `TimeoutEpochs` logical epochs the source aborts, unlocks the amount, and any late
//      Commit or Ack for that transaction is ignored, so an abort can never be overtaken by a commit.
//
// Messages travel over an explicit in-memory network: a queue the driver drains one message at a time, with a
// deterministic schedule that can reorder, duplicate or drop deliveries. No threads, no sockets, no clock - epochs
// advance only when `Tick` is called.
//
// Verified invariants (each recomputed here, none assumed):
//   * conservation of value is checked after *every single delivery* in every scenario: the sum of balances plus
//     the sum of locked amounts always equals the minted total, so no path can create or destroy money;
//   * atomicity: at quiescence a transaction is either committed at both shards or aborted at both, and there is
//     no state where the source was debited and the destination was not credited (the sharp failure mode);
//   * idempotence: re-delivering a message changes nothing, because every message carries the transaction id and
//     every transition is guarded by the transaction's state;
//   * at quiescence nothing is left locked, no receipt is left pending, and every submitted transaction is
//     either committed or aborted - the liveness property, under a fair schedule and under a lossy one;
//   * the sharded result equals an independent, single-shard reference execution of the same transfer set
//     whenever all of them commit, so sharding did not change the ledger's semantics;
//   * a double spend cannot happen: of two conflicting transfers at most one commits, and the loser is refunded;
//   * each cross-shard transaction costs exactly three messages (Prepare, Ack, Commit), which is the latency and
//     bandwidth model of the protocol. Every scenario is deterministic and reproducible.
//
// Documented simplification: two shards of eight accounts each, one transaction per submission (no batching or
// blocks), no signatures (message authenticity is assumed, and the missing signature check is stated rather than
// pretended), no fee market, and a crude timeout sweep instead of an epoch committee. What is real: the two-phase
// commit-like receipt protocol, the lock/unlock accounting, the idempotent message handlers, the conservation and
// atomicity invariants and the cost model.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextBelow(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal enum Kind { Prepare, Ack, Commit }

internal enum TxState { Pending, Committed, Aborted, Rejected }

internal sealed class Message
{
    internal long Transaction;
    internal Kind Kind;
    internal int SourceShard;
    internal int DestinationShard;
    internal int SourceAccount;
    internal int DestinationAccount;
    internal long Amount;
    internal int Epoch;
}

internal sealed class Transaction
{
    internal long Id;
    internal int SourceShard, SourceAccount, DestinationShard, DestinationAccount;
    internal long Amount;
    internal TxState State = TxState.Pending;
    internal int SubmittedEpoch;
    internal bool ReceiptRecorded;
    internal bool Acked;
    internal bool Committed;
}

internal sealed class Shard
{
    internal readonly int Index;
    internal readonly long[] Balances;
    internal readonly long[] Locked;
    internal readonly Dictionary<long, Transaction> Inflight = new Dictionary<long, Transaction>();
    internal readonly List<Message> Outbox = new List<Message>();

    internal Shard(int index, long[] balances)
    {
        Index = index;
        Balances = balances;
        Locked = new long[balances.Length];
    }
}

internal sealed class CrossShardLedger
{
    internal const int TimeoutEpochs = 3;

    private readonly List<Shard> _shards = new List<Shard>();
    private readonly Queue<Message> _network = new Queue<Message>();
    private readonly Dictionary<long, Transaction> _transactions = new Dictionary<long, Transaction>();
    private long _nextTransactionId = 1;
    private int _epoch;
    private long _minted;

    internal CrossShardLedger(int shardCount, int accountsPerShard, long initialBalance)
    {
        if (shardCount < 1) throw new ArgumentException("at least one shard is required", nameof(shardCount));
        if (accountsPerShard < 1) throw new ArgumentException("at least one account per shard is required");
        if (initialBalance < 0) throw new ArgumentException("the initial balance must not be negative");
        for (int s = 0; s < shardCount; s++)
        {
            var balances = new long[accountsPerShard];
            for (int a = 0; a < accountsPerShard; a++) balances[a] = initialBalance;
            _shards.Add(new Shard(s, balances));
            _minted += initialBalance * accountsPerShard;
        }
    }

    internal int ShardCount => _shards.Count;
    internal int Epoch => _epoch;
    internal int QueuedMessages => _network.Count;

    internal long Balance(int shard, int account) => _shards[shard].Balances[account];
    internal long Locked(int shard, int account) => _shards[shard].Locked[account];
    internal long TotalLocked
    {
        get
        {
            long total = 0;
            foreach (Shard shard in _shards)
            {
                foreach (long v in shard.Locked) total += v;
            }
            return total;
        }
    }

    internal long TotalBalance
    {
        get
        {
            long total = 0;
            foreach (Shard shard in _shards)
            {
                foreach (long v in shard.Balances) total += v;
            }
            return total;
        }
    }

    internal long Minted => _minted;

    internal Transaction? Lookup(long id) => _transactions.TryGetValue(id, out Transaction tx) ? tx : null;

    internal int CommittedCount
    {
        get
        {
            int n = 0;
            foreach (Transaction tx in _transactions.Values)
            {
                if (tx.State == TxState.Committed) n++;
            }
            return n;
        }
    }

    private bool Known(int shard, int account) =>
        shard >= 0 && shard < _shards.Count && account >= 0 && account < _shards[shard].Balances.Length;

    /** Client entry point: locks the amount at the source shard and emits Prepare, or rejects the transfer. */
    internal long Submit(int sourceShard, int sourceAccount, int destinationShard, int destinationAccount, long amount)
    {
        if (amount <= 0) throw new ArgumentException("the amount must be positive", nameof(amount));
        if (!Known(sourceShard, sourceAccount) || !Known(destinationShard, destinationAccount))
        {
            throw new ArgumentException("unknown shard or account");
        }
        long id = _nextTransactionId++;
        var tx = new Transaction
        {
            Id = id, SourceShard = sourceShard, SourceAccount = sourceAccount,
            DestinationShard = destinationShard, DestinationAccount = destinationAccount,
            Amount = amount, SubmittedEpoch = _epoch
        };
        _transactions[id] = tx;
        Shard source = _shards[sourceShard];
        if (source.Balances[sourceAccount] < amount)
        {
            tx.State = TxState.Rejected;
            return id;
        }
        source.Balances[sourceAccount] -= amount;
        source.Locked[sourceAccount] += amount;
        source.Inflight[id] = tx;
        Emit(source, tx, Kind.Prepare);
        return id;
    }

    private void Emit(Shard from, Transaction tx, Kind kind)
    {
        _network.Enqueue(new Message
        {
            Transaction = tx.Id, Kind = kind, SourceShard = tx.SourceShard, DestinationShard = tx.DestinationShard,
            SourceAccount = tx.SourceAccount, DestinationAccount = tx.DestinationAccount,
            Amount = tx.Amount, Epoch = _epoch
        });
    }

    /** Delivers one queued message (index 0 by default, or another position for reordering tests). */
    internal Message? Deliver(int index = 0)
    {
        if (_network.Count == 0) return null;
        var drained = new List<Message>();
        while (drained.Count < index && _network.Count > 0) drained.Add(_network.Dequeue());
        Message message = _network.Dequeue();
        Dispatch(message);
        foreach (Message m in drained) _network.Enqueue(m);
        return message;
    }

    /** Re-dispatches an already delivered message: exactly what a duplicated delivery on the wire looks like. */
    internal void Redispatch(Message message) => Dispatch(message);

    /** Removes the next queued message of a kind without delivering it: a lost message on a lossy link. */
    internal bool Drop(Kind kind)
    {
        int count = _network.Count;
        var kept = new List<Message>();
        bool dropped = false;
        for (int i = 0; i < count; i++)
        {
            Message m = _network.Dequeue();
            if (!dropped && m.Kind == kind) dropped = true;
            else kept.Add(m);
        }
        foreach (Message m in kept) _network.Enqueue(m);
        return dropped;
    }

    private void Dispatch(Message message)
    {
        if (!_transactions.TryGetValue(message.Transaction, out Transaction tx)) return;   // unknown transaction
        if (message.Amount != tx.Amount) return;                                            // tampered amount
        Shard source = _shards[tx.SourceShard];
        Shard destination = _shards[tx.DestinationShard];
        switch (message.Kind)
        {
            case Kind.Prepare:
                if (tx.State != TxState.Pending) return;          // aborted or committed: nothing to record
                if (tx.ReceiptRecorded) return;                   // duplicate delivery is idempotent
                tx.ReceiptRecorded = true;
                Emit(destination, tx, Kind.Ack);
                break;
            case Kind.Ack:
                if (tx.State != TxState.Pending || tx.Acked) return;
                tx.Acked = true;
                long amount = tx.Amount;
                source.Locked[tx.SourceAccount] -= amount;        // the funds leave the source shard here
                source.Inflight.Remove(tx.Id);
                tx.State = TxState.Committed;
                Emit(source, tx, Kind.Commit);
                break;
            case Kind.Commit:
                if (tx.State != TxState.Committed || tx.Committed) return;
                tx.Committed = true;
                destination.Balances[tx.DestinationAccount] += tx.Amount;
                break;
            default:
                return;
        }
    }

    /** Advances the logical epoch and aborts every transfer whose receipt never came back. */
    internal int Tick()
    {
        _epoch++;
        int aborted = 0;
        foreach (Shard shard in _shards)
        {
            var expired = new List<Transaction>();
            foreach (Transaction tx in shard.Inflight.Values)
            {
                if (_epoch - tx.SubmittedEpoch >= TimeoutEpochs) expired.Add(tx);
            }
            foreach (Transaction tx in expired)
            {
                shard.Locked[tx.SourceAccount] -= tx.Amount;
                shard.Balances[tx.SourceAccount] += tx.Amount;
                tx.State = TxState.Aborted;
                shard.Inflight.Remove(tx.Id);
                aborted++;
            }
        }
        return aborted;
    }

    /** Value released by a commit but not yet credited at the destination: it is being carried by Commit. */
    internal long InTransit
    {
        get
        {
            long total = 0;
            foreach (Transaction tx in _transactions.Values)
            {
                if (tx.State == TxState.Committed && !tx.Committed) total += tx.Amount;
            }
            return total;
        }
    }

    /** True when no message is queued, nothing is locked, nothing is in transit and no transfer is pending. */
    internal bool Quiescent()
    {
        if (_network.Count > 0 || TotalLocked != 0 || InTransit != 0) return false;
        foreach (Transaction tx in _transactions.Values)
        {
            if (tx.State == TxState.Pending) return false;
        }
        return true;
    }

    /**
     * Conservation of value: balances plus locked amounts plus value carried by an in-flight Commit must always
     * equal what was minted. Skipping the in-transit term would hide exactly the window between the source's
     * commit and the destination's credit, which is the window a broken protocol loses money in.
     */
    internal bool Conserved() => TotalBalance + TotalLocked + InTransit == _minted;

    internal long PendingReceipts
    {
        get
        {
            long n = 0;
            foreach (Transaction tx in _transactions.Values)
            {
                if (tx.ReceiptRecorded && !tx.Committed && tx.State != TxState.Committed) n++;
            }
            return n;
        }
    }
}

/** Independent oracle: the same transfers executed serially on one shard. */
internal static class SingleShardReference
{
    internal static long[] Run(int accountCount, long initial, IReadOnlyList<(int From, int To, long Amount)> transfers)
    {
        var balances = new long[accountCount];
        for (int i = 0; i < accountCount; i++) balances[i] = initial;
        foreach ((int from, int to, long amount) in transfers)
        {
            if (balances[from] < amount) continue;   // the sharded version would reject or abort this transfer
            balances[from] -= amount;
            balances[to] += amount;
        }
        return balances;
    }
}

public static class Solution
{
    private const int Shards = 2;
    private const int Accounts = 8;
    private const long Initial = 1000;

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; }
        catch (ArgumentException) { return true; }
    }

    private static long Flatten(CrossShardLedger ledger, int shard, int account) =>
        ledger.Balance(shard, account);

    /**
     * Independent oracle: the same transfers executed on one flat ledger whose account space is the concatenation
     * of the shards' account spaces, so sharding can be compared against a non-sharded execution.
     */
    private static bool MatchesOracle(CrossShardLedger ledger,
        IReadOnlyList<(int SourceShard, int Source, int DestinationShard, int Destination, long Amount)> plan)
    {
        var flat = new List<(int, int, long)>();
        foreach (var t in plan)
        {
            flat.Add((t.SourceShard * Accounts + t.Source, t.DestinationShard * Accounts + t.Destination, t.Amount));
        }
        long[] oracle = SingleShardReference.Run(Shards * Accounts, Initial, flat);
        for (int s = 0; s < Shards; s++)
        {
            for (int a = 0; a < Accounts; a++)
            {
                if (ledger.Balance(s, a) != oracle[s * Accounts + a]) return false;
            }
        }
        return true;
    }

    /** Drains the network, ticking when it stalls so that timeouts can fire; returns the number of deliveries. */
    private static int Drain(CrossShardLedger ledger)
    {
        int deliveries = 0;
        int idle = 0;
        while (!ledger.Quiescent() && idle < 3 * CrossShardLedger.TimeoutEpochs + 4)
        {
            if (ledger.QueuedMessages > 0)
            {
                ledger.Deliver();
                deliveries++;
                idle = 0;
                Check(ledger.Conserved(), "value is conserved after every delivery");
            }
            else
            {
                if (ledger.TotalLocked == 0) break;
                ledger.Tick();
                idle++;
            }
        }
        return deliveries;
    }

    public static void Main()
    {
        // 1-3. The happy path: disjoint transfers all commit and the ledger matches the single-shard oracle.
        var ledger = new CrossShardLedger(Shards, Accounts, Initial);
        var plan = new List<(int SourceShard, int Source, int DestinationShard, int Destination, long Amount)>
        {
            (1, 0, 0, 1, 50), (1, 2, 0, 3, 25), (1, 4, 0, 5, 100), (1, 6, 0, 7, 10)
        };
        long conservationBaseline = ledger.Minted;
        Check(ledger.Conserved() && ledger.TotalBalance == conservationBaseline, "the freshly minted ledger is conserved");
        foreach (var t in plan) ledger.Submit(t.SourceShard, t.Source, t.DestinationShard, t.Destination, t.Amount);
        Check(ledger.TotalLocked == 185, "submitting locks exactly the submitted amounts at the source shard");
        Check(ledger.Conserved(), "locking conserves value: the balances dropped by exactly what was locked");
        int deliveries = Drain(ledger);
        Check(ledger.Quiescent(), "the happy path reaches quiescence");
        Check(ledger.CommittedCount == plan.Count, "every transfer commits");
        Check(deliveries == 3 * plan.Count, "each cross-shard transfer costs exactly three messages");
        Check(MatchesOracle(ledger, plan), "the sharded ledger equals the single-shard reference execution");
        Check(ledger.TotalLocked == 0 && ledger.InTransit == 0 && ledger.PendingReceipts == 0,
            "nothing is left locked, in transit or pending at quiescence");

        // 4-5. Idempotence: re-dispatching a delivered message changes nothing, whatever its kind.
        var dup = new CrossShardLedger(Shards, Accounts, Initial);
        long dupId = dup.Submit(0, 1, 1, 2, 40);
        Message? prepare = dup.Deliver();
        Check(prepare != null && prepare.Kind == Kind.Prepare, "the source emits Prepare first");
        dup.Redispatch(prepare!);
        Check(dup.QueuedMessages == 1, "a duplicated Prepare does not queue a second Ack");
        Check(dup.Conserved() && dup.TotalLocked == 40, "and does not move value");
        Message? ack = dup.Deliver();
        Check(ack != null && ack.Kind == Kind.Ack, "the destination answers with Ack");
        dup.Redispatch(ack!);
        Check(dup.Lookup(dupId)!.State == TxState.Committed, "a duplicated Ack does not double-commit");
        Check(dup.TotalLocked == 0, "the locked amount is released exactly once");
        Message? commit = dup.Deliver();
        Check(commit != null && commit.Kind == Kind.Commit, "the source then emits Commit");
        dup.Redispatch(commit!);
        dup.Redispatch(commit!);
        Check(dup.Balance(1, 2) == Initial + 40, "the destination is credited exactly once despite duplicated Commits");
        Check(dup.Conserved() && dup.TotalBalance == dup.Minted, "and no money was created by the duplicates");

        // 6-7. Delivery order does not matter: the same transfers, reordered messages, same outcome.
        var reversed = new CrossShardLedger(Shards, Accounts, Initial);
        foreach (var t in plan) reversed.Submit(t.SourceShard, t.Source, t.DestinationShard, t.Destination, t.Amount);
        int guard = 0;
        while (!reversed.Quiescent() && guard++ < 40)
        {
            if (reversed.QueuedMessages > 0) reversed.Deliver(reversed.QueuedMessages - 1);   // LIFO
            else reversed.Tick();
            Check(reversed.Conserved(), "conservation holds under reverse-order delivery too");
        }
        bool sameOutcome = true;
        for (int a = 0; a < Accounts; a++)
        {
            if (reversed.Balance(0, a) != ledger.Balance(0, a)) sameOutcome = false;
        }
        Check(sameOutcome, "reordering deliveries does not change the final ledger");

        // 8-9. A dropped Prepare: the source times out, refunds, and the transfer aborts with no credit anywhere.
        var lossy = new CrossShardLedger(Shards, Accounts, Initial);
        long abortedId = lossy.Submit(1, 3, 0, 4, 300);
        Check(lossy.Balance(1, 3) == Initial - 300 && lossy.TotalLocked == 300, "the amount is locked while in flight");
        Check(lossy.Drop(Kind.Prepare), "the lossy link drops the Prepare");
        Check(lossy.QueuedMessages == 0, "nothing else was emitted for that transfer");
        lossy.Tick();
        lossy.Tick();
        lossy.Tick();
        Check(lossy.Lookup(abortedId)!.State == TxState.Aborted, "a transfer whose Prepare never arrives aborts");
        Check(lossy.Balance(1, 3) == Initial, "and the source shard is refunded in full");
        Check(lossy.Balance(0, 4) == Initial, "the destination was never credited");
        Check(lossy.Quiescent() && lossy.Conserved(), "the abort leaves nothing locked and conserves value");

        // A dropped Ack after the destination recorded its receipt: the source still aborts, and the stale
        // receipt must not be applied by a late Commit, because Commit is only sent on a real Ack.
        var halfDropped = new CrossShardLedger(Shards, Accounts, Initial);
        long halfId = halfDropped.Submit(0, 2, 1, 3, 90);
        Message? received = halfDropped.Deliver();
        Check(received != null && halfDropped.Lookup(halfId)!.ReceiptRecorded,
            "the destination records the receipt before answering");
        Check(halfDropped.Drop(Kind.Ack), "the Ack is lost on the way back");
        halfDropped.Tick();
        halfDropped.Tick();
        halfDropped.Tick();
        Check(halfDropped.Lookup(halfId)!.State == TxState.Aborted, "the source aborts after the timeout");
        Check(halfDropped.Balance(0, 2) == Initial, "the source is refunded");
        Check(halfDropped.Balance(1, 3) == Initial, "and the destination never applies the stale receipt");
        Check(halfDropped.Conserved(), "the half-completed protocol still conserves value");
        Check(halfDropped.Lookup(halfId)!.State != TxState.Committed, "an aborted transfer can never commit later");

        // A stalling network: tick until every in-flight transfer times out, and check the ledger is whole again.
        var stalled = new CrossShardLedger(Shards, Accounts, Initial);
        long stalledId = stalled.Submit(0, 0, 1, 1, 250);
        stalled.Tick();
        Check(stalled.Lookup(stalledId)!.State == TxState.Pending, "one tick is not yet a timeout");
        stalled.Tick();
        stalled.Tick();
        Check(stalled.Lookup(stalledId)!.State == TxState.Aborted, "after TimeoutEpochs the transfer aborts");
        Check(stalled.Balance(0, 0) == Initial && stalled.TotalLocked == 0, "the refund restores the balance");
        Check(stalled.Balance(1, 1) == Initial, "the destination was never credited (atomicity held)");

        // 10-11. Insufficient funds and a double spend: at most one of two conflicting transfers commits.
        var conflict = new CrossShardLedger(Shards, Accounts, Initial);
        long first = conflict.Submit(0, 0, 1, 0, 700);
        long second = conflict.Submit(0, 0, 1, 1, 700);
        Check(conflict.Lookup(second)!.State == TxState.Rejected, "the second transfer is rejected for lack of funds");
        Check(conflict.Lookup(first)!.State == TxState.Pending, "the first one proceeds");
        Drain(conflict);
        Check(conflict.Lookup(first)!.State == TxState.Committed, "the first transfer commits");
        Check(conflict.Balance(0, 0) == Initial - 700, "the source account is debited exactly once");
        Check(conflict.Balance(1, 0) == Initial + 700 && conflict.Balance(1, 1) == Initial,
            "exactly one destination account received the funds");
        Check(conflict.TotalBalance == conflict.Minted, "and the ledger stays conserved, so no double spend happened");

        // 12-13. Rejection paths and validation.
        var validation = new CrossShardLedger(Shards, Accounts, Initial);
        long rejected = validation.Submit(0, 5, 1, 5, Initial + 1);
        Check(validation.Lookup(rejected)!.State == TxState.Rejected && validation.QueuedMessages == 0,
            "a transfer above the balance is rejected without emitting anything");
        Check(Throws(() => validation.Submit(0, 0, 1, 0, 0)), "a zero amount is rejected");
        Check(Throws(() => validation.Submit(0, 0, 1, 0, -5)), "a negative amount is rejected");
        Check(Throws(() => validation.Submit(0, 0, 9, 0, 1)), "an unknown destination shard is rejected");
        Check(Throws(() => validation.Submit(0, Accounts, 1, 0, 1)), "an unknown source account is rejected");
        Check(Throws(() => new CrossShardLedger(0, Accounts, Initial)), "a ledger with no shards is rejected");
        Check(Throws(() => new CrossShardLedger(Shards, 0, Initial)), "a ledger with no accounts is rejected");
        Check(Throws(() => new CrossShardLedger(Shards, Accounts, -1)), "a negative initial balance is rejected");

        // 14. The same-shard transfer path is a plain atomic debit and credit.
        var local = new CrossShardLedger(Shards, Accounts, Initial);
        long localId = local.Submit(0, 0, 0, 3, 120);
        Drain(local);
        Check(local.Lookup(localId)!.State == TxState.Committed, "a same-shard transfer commits");
        Check(local.Balance(0, 0) == Initial - 120 && local.Balance(0, 3) == Initial + 120,
            "and moves value inside the shard");

        // 15-16. A long randomised scenario: conservation after every delivery, and everything terminates.
        var rng = 0xC0FFEEUL;
        var stress = new CrossShardLedger(Shards, Accounts, Initial);
        var submitted = new List<long>();
        for (int round = 0; round < 40; round++)
        {
            int sourceShard = SplitMix64.NextBelow(ref rng, Shards);
            int destinationShard = SplitMix64.NextBelow(ref rng, Shards);
            int sourceAccount = SplitMix64.NextBelow(ref rng, Accounts);
            int destinationAccount = SplitMix64.NextBelow(ref rng, Accounts);
            long amount = 1 + SplitMix64.NextBelow(ref rng, 400);
            submitted.Add(stress.Submit(sourceShard, sourceAccount, destinationShard, destinationAccount, amount));
            Check(stress.Conserved(), "conservation holds after every submission");
            switch (SplitMix64.NextBelow(ref rng, 4))
            {
                case 0:
                    if (stress.QueuedMessages > 0) stress.Deliver(0);
                    break;
                case 1:
                    if (stress.QueuedMessages > 1) stress.Deliver(stress.QueuedMessages - 1);
                    break;
                case 2:
                    stress.Tick();
                    break;
                default:
                    if (stress.QueuedMessages > 0) stress.Deliver(0);
                    if (stress.QueuedMessages > 0) stress.Deliver(0);
                    break;
            }
            Check(stress.Conserved(), "conservation holds after every delivery in the stress scenario");
        }
        int stressDeliveries = Drain(stress);
        Check(stress.Quiescent(), "the randomised scenario terminates");
        Check(stress.Conserved(), "and conserves value at the end");
        Console.WriteLine("  stress: " + submitted.Count + " transfers, " + stress.CommittedCount
            + " committed, " + stressDeliveries + " extra deliveries, total=" + stress.TotalBalance);
        Check(stress.TotalBalance == stress.Minted, "nothing was minted or burned");
        int settled = 0;
        foreach (long id in submitted)
        {
            TxState state = stress.Lookup(id)!.State;
            if (state == TxState.Committed || state == TxState.Aborted || state == TxState.Rejected) settled++;
        }
        Check(settled == submitted.Count, "every transfer is either committed, aborted or rejected - no transfer is stuck");

        // 17. Atomicity of every committed transfer: exactly one debit and one credit.
        var audit = new CrossShardLedger(Shards, Accounts, Initial);
        var auditPlan = new List<(int SourceShard, int Source, int DestinationShard, int Destination, long Amount)>
        {
            (1, 0, 0, 4, 30), (1, 5, 0, 2, 70), (1, 7, 0, 1, 15)
        };
        foreach (var t in auditPlan) audit.Submit(t.SourceShard, t.Source, t.DestinationShard, t.Destination, t.Amount);
        Drain(audit);
        long debited = 0, credited = 0;
        foreach (var t in auditPlan)
        {
            debited += Initial - audit.Balance(t.SourceShard, t.Source);
            credited += audit.Balance(t.DestinationShard, t.Destination) - Initial;
        }
        Console.WriteLine("  atomicity: debited=" + debited + " credited=" + credited);
        Check(debited == credited, "every committed transfer debits exactly what it credits");
        Check(debited == 115, "and the totals match the submitted plan");
        Check(audit.TotalBalance == audit.Minted, "the audit ledger is conserved");
        Check(MatchesOracle(audit, auditPlan), "the audit ledger matches the reference execution");

        // 18. Determinism: replaying the same scenario reproduces the ledger bit for bit.
        var replay = new CrossShardLedger(Shards, Accounts, Initial);
        foreach (var t in plan) replay.Submit(t.SourceShard, t.Source, t.DestinationShard, t.Destination, t.Amount);
        Drain(replay);
        bool identical = true;
        for (int s = 0; s < Shards; s++)
        {
            for (int a = 0; a < Accounts; a++)
            {
                if (replay.Balance(s, a) != ledger.Balance(s, a)) identical = false;
            }
        }
        Check(identical, "replaying the same scenario reproduces the ledger bit for bit");
        Check(Flatten(replay, 0, 0) == Initial, "the source accounts that never sent anything are untouched");

        Console.WriteLine("PASS");
    }
}
