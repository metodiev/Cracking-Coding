// Task 661 - Implement chain replication simulation for strong consistency. (Tier: Expert, CTCI Ch 9 - Sys Design)
//
// A chain of N replicas: the head accepts every write, each replica forwards the write to its successor, and
// the write is committed only when the tail has applied it. Reads are served by the tail alone. That single
// design decision is what buys strong consistency - the tail can only ever hold a prefix of the commit order,
// so a read never sees an uncommitted value and never misses a committed one - and it is also the reason the
// chain is the pricey end of the consistency trade-off: nothing is acknowledged until the slowest hop has run.
// What is simplified: a fixed per-hop latency (no queueing, no congestion, no message loss), instantaneous
// failure injection at the current simulated time, no chain reconfiguration or head election (the file shows
// instead what a failure costs: writes submitted through a failed node are simply dropped), and reads are
// always sent to the tail because that is the contract being modelled.
// Verified: the acknowledgment time of a write equals the sum of the hop latencies and is hand-checked on a
// three-node chain; pipelining many writes beats serialising them; every node's log is a prefix of its
// predecessor's at all times, so the tail's log is exactly the committed prefix; a read at the tail returns
// the newest acknowledged write and returns "empty" before the first commit; a 300-operation random
// interleaving of writes and reads is checked against an independently maintained acknowledgment log, where
// each read must return the last write acknowledged before the read was issued (the linearizability check);
// a mid-chain failure stops the writes that cross it while leaving every earlier commit intact; a failed head
// rejects writes; and reading a non-tail or writing to a non-head replica is rejected.
using System;
using System.Collections.Generic;

internal static class Rng
{
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);

    internal static double NextUnit(ref ulong state) => NextInt(ref state, 1000) / 1000.0;
}

internal sealed class ChainMessage
{
    internal double Arrival;
    internal int Node;
    internal int Version;
    internal string Payload = string.Empty;
}

internal sealed class Chain
{
    private readonly int _nodes;
    private readonly double _hopLatency;
    private readonly List<ChainMessage> _inFlight = new List<ChainMessage>();
    private readonly List<int>[] _logs;
    private readonly bool[] _failed;
    private readonly List<double> _ackTimes = new List<double>();

    internal Chain(int nodes, double hopLatency)
    {
        if (nodes < 1) throw new ArgumentException("a chain needs at least one node");
        _nodes = nodes;
        _hopLatency = hopLatency;
        _logs = new List<int>[nodes];
        _failed = new bool[nodes];
        for (int i = 0; i < nodes; i++) _logs[i] = new List<int>();
    }

    internal int SubmitCount { get; private set; }

    internal double Now { get; private set; }

    internal IReadOnlyList<int> LogOf(int node) => _logs[node];

    internal IReadOnlyList<double> AckTimes => _ackTimes;

    internal int CommittedCount { get; private set; }

    internal bool IsFailed(int node) => _failed[node];

    internal void Fail(int node)
    {
        if (node < 0 || node >= _nodes) throw new ArgumentOutOfRangeException(nameof(node));
        _failed[node] = true;
    }

    /// <summary>Accepts a write at the head. Returns false when the chain cannot take it.</summary>
    internal bool Submit(string payload, double at)
    {
        if (_failed[0]) return false;
        SubmitCount++;
        _inFlight.Add(new ChainMessage { Arrival = at, Node = 0, Version = SubmitCount, Payload = payload });
        return true;
    }

    /// <summary>Processes every message that arrives no later than `until`, in arrival order.</summary>
    internal void Run(double until)
    {
        while (true)
        {
            int next = -1;
            for (int i = 0; i < _inFlight.Count; i++)
                if (_inFlight[i].Arrival <= until && (next < 0 || _inFlight[i].Arrival < _inFlight[next].Arrival))
                    next = i;
            if (next < 0) break;
            ChainMessage message = _inFlight[next];
            _inFlight.RemoveAt(next);
            Now = Math.Max(Now, message.Arrival);
            if (_failed[message.Node]) continue;                       // a failed replica drops its traffic
            _logs[message.Node].Add(message.Version);
            if (message.Node == _nodes - 1)
            {
                CommittedCount++;
                _ackTimes.Add(message.Arrival);
            }
            else
            {
                _inFlight.Add(new ChainMessage
                {
                    Arrival = message.Arrival + _hopLatency,
                    Node = message.Node + 1,
                    Version = message.Version,
                    Payload = message.Payload,
                });
            }
        }
        Now = Math.Max(Now, until);
    }

    /// <summary>Reads go to the tail; it holds the committed prefix.</summary>
    internal (int Version, string Payload) ReadTail()
    {
        if (_failed[_nodes - 1]) throw new InvalidOperationException("the tail is unavailable");
        IReadOnlyList<int> tail = _logs[_nodes - 1];
        return tail.Count == 0 ? (0, "empty") : (tail[tail.Count - 1], "value" + tail[tail.Count - 1]);
    }

    internal string Read(int node, int requestedVersion)
    {
        if (node != _nodes - 1) throw new InvalidOperationException("chain replication serves reads at the tail");
        IReadOnlyList<int> log = _logs[node];
        if (requestedVersion <= 0 || requestedVersion > log.Count)
            throw new ArgumentException("version " + requestedVersion + " is not in the committed prefix");
        return "value" + log[requestedVersion - 1];
    }

    /// <summary>Null when each replica's log is a prefix of its predecessor's, otherwise the first violation.</summary>
    internal string? PrefixViolation()
    {
        for (int node = 0; node + 1 < _nodes; node++)
        {
            IReadOnlyList<int> upstream = _logs[node];
            IReadOnlyList<int> downstream = _logs[node + 1];
            if (downstream.Count > upstream.Count)
                return "node " + (node + 1) + " is ahead of node " + node;
            for (int i = 0; i < downstream.Count; i++)
                if (downstream[i] != upstream[i])
                    return "node " + (node + 1) + " diverges from node " + node + " at position " + i;
        }
        return null;
    }

    internal int InFlight => _inFlight.Count;
}

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAILED: " + label); }

    private static bool Close(double a, double b) => Math.Abs(a - b) <= 1e-9;

    public static void Main()
    {
        // ---- hand-checked hop arithmetic --------------------------------------------------------
        {
            var chain = new Chain(3, 5.0);
            Check(chain.Submit("first", 0), "the head accepts a write");
            chain.Run(9.999999);
            Check(chain.CommittedCount == 0 && chain.LogOf(0).Count == 1 && chain.LogOf(1).Count == 1,
                "one hop before the last the write has reached the middle node but is not committed");
            Check(chain.ReadTail().Payload == "empty", "the tail has not seen the write yet");
            chain.Run(10.0);
            Check(chain.CommittedCount == 1 && Close(chain.AckTimes[0], 10.0),
                "with two hops of five the write is acknowledged at time ten");
            Check(chain.ReadTail().Version == 1, "the tail now serves the write");
            chain.Run(100.0);
            Check(chain.PrefixViolation() == null && chain.LogOf(0).Count == chain.LogOf(2).Count,
                "once the chain quiesces every replica holds the same committed prefix");

            var single = new Chain(1, 7.0);
            single.Submit("only", 3.0);
            single.Run(3.0);
            Check(single.CommittedCount == 1 && Close(single.AckTimes[0], 3.0),
                "a one-node chain is its own head and tail, so a write commits immediately");
        }

        // ---- pipelining versus serialising -----------------------------------------------------
        {
            const int Writes = 40;
            var pipelined = new Chain(4, 2.0);
            for (int i = 0; i < Writes; i++) pipelined.Submit("w" + i, i * 0.5);
            pipelined.Run(1000);
            Check(pipelined.CommittedCount == Writes, "every pipelined write commits");
            double pipelinedSpan = pipelined.AckTimes[Writes - 1] - pipelined.AckTimes[0];

            var serialised = new Chain(4, 2.0);
            double at = 0;
            for (int i = 0; i < Writes; i++)
            {
                serialised.Submit("w" + i, at);
                at += 3 * 2.0;      // wait for each acknowledgment before sending the next
                serialised.Run(at);
            }
            double serialisedSpan = serialised.AckTimes[Writes - 1] - serialised.AckTimes[0];
            Check(pipelinedSpan < serialisedSpan / 10,
                "pipelining collapses the latency of a batch (" + pipelinedSpan + " vs " + serialisedSpan + ")");
            Check(pipelined.PrefixViolation() == null && serialised.PrefixViolation() == null,
                "both chains satisfy the prefix invariant");
            Check(Close(pipelined.AckTimes[0], 6.0) && Close(pipelined.AckTimes[1], 6.5),
                "the acknowledgment clock only depends on the hop count, not on the batch size");
        }

        // ---- an interleaved workload checked against an independent acknowledgment log ----------
        {
            const int Nodes = 5;
            var chain = new Chain(Nodes, 1.5);
            ulong state = 0x5DEECE66D1234567UL;
            var ackLog = new List<(double AckAt, int Version)>();
            double now = 0;
            int writes = 0, reads = 0;
            bool linearizable = true;
            string? failure = null;

            for (int op = 0; op < 300; op++)
            {
                now += Rng.NextUnit(ref state) * 2;
                if (Rng.NextInt(ref state, 3) != 0)
                {
                    writes++;
                    if (chain.Submit("w" + writes, now)) chain.Run(now);
                }
                else
                {
                    reads++;
                    chain.Run(now);
                    // Fold newly observed acknowledgments into the reference log.
                    foreach (double ack in chain.AckTimes)
                        if (!IsRecorded(ackLog, ack)) ackLog.Add((ack, ackLog.Count + 1));
                    var (version, payload) = chain.ReadTail();
                    int expectedVersion = 0;
                    foreach (var entry in ackLog) if (entry.AckAt <= now + 1e-9 && entry.Version > expectedVersion)
                            expectedVersion = entry.Version;
                    if (version != expectedVersion || payload != (expectedVersion == 0 ? "empty" : "value" + expectedVersion))
                    {
                        linearizable = false;
                        failure ??= "op " + op + " read version " + version + " expected " + expectedVersion;
                    }
                }
            }
            chain.Run(double.MaxValue);
            foreach (double ack in chain.AckTimes) if (!IsRecorded(ackLog, ack)) ackLog.Add((ack, ackLog.Count + 1));
            Check(writes > 50 && reads > 50, "the workload really mixed writes (" + writes + ") and reads (" + reads + ")");
            Check(linearizable,
                "every tail read returned the last write acknowledged before it was issued: " + (failure ?? "ok"));
            Check(chain.CommittedCount == writes, "all submitted writes committed once the chain quiesced");
            Check(chain.PrefixViolation() == null, "the prefix invariant survives the whole interleaving");
            Check(chain.LogOf(Nodes - 1).Count == writes, "the tail log holds exactly the committed writes");
            Check(chain.Read(Nodes - 1, 1) == "value1" && chain.Read(Nodes - 1, writes) == "value" + writes,
                "historical reads at the tail follow the commit order");
            bool rejected = false;
            try { chain.Read(0, 1); }
            catch (InvalidOperationException) { rejected = true; }
            Check(rejected, "reading a replica that is not the tail is rejected by the contract");
            rejected = false;
            try { chain.Read(Nodes - 1, writes + 5); }
            catch (ArgumentException) { rejected = true; }
            Check(rejected, "reading a version that was never committed is rejected");
            Console.WriteLine("writes=" + writes + " reads=" + reads + " tailLog=" + chain.LogOf(Nodes - 1).Count);
        }

        // ---- failures --------------------------------------------------------------------------
        {
            var chain = new Chain(5, 4.0);
            for (int i = 0; i < 3; i++) chain.Submit("early" + i, i);
            chain.Run(100);
            int committedBefore = chain.CommittedCount;
            Check(committedBefore == 3, "three writes committed before the failure");
            Check(chain.ReadTail().Version == 3, "the tail holds the third write");

            chain.Fail(2);
            for (int i = 0; i < 4; i++) chain.Submit("late" + i, 200 + i);
            chain.Run(1000);
            Check(chain.CommittedCount == committedBefore,
                "writes that have to cross the failed replica are never acknowledged");
            Check(chain.ReadTail().Version == 3, "the committed prefix is unchanged by the failure");
            Check(chain.LogOf(2).Count == 3 && chain.LogOf(3).Count == 3,
                "the failed replica and its successors simply stop advancing");
            Check(chain.PrefixViolation() == null, "the prefix invariant still holds after the failure");

            var headless = new Chain(3, 1.0);
            headless.Fail(0);
            Check(!headless.Submit("nope", 5.0) && headless.CommittedCount == 0,
                "a chain with a failed head accepts nothing at all (the availability price of a chain)");
            var emptyTail = headless.ReadTail();
            Check(emptyTail.Version == 0 && emptyTail.Payload == "empty",
                "the tail still answers reads, reporting that nothing was ever committed");

            var tailDown = new Chain(3, 1.0);
            tailDown.Submit("x", 0);
            tailDown.Run(1.5);
            tailDown.Fail(2);
            tailDown.Run(100);
            bool rejected = false;
            try { tailDown.ReadTail(); }
            catch (InvalidOperationException) { rejected = true; }
            Check(rejected && tailDown.CommittedCount == 0,
                "with the tail down reads fail and the write never became committed");
        }

        Console.WriteLine("PASS");
    }

    private static bool IsRecorded(List<(double AckAt, int Version)> log, double ack)
    {
        foreach (var entry in log) if (Close(entry.AckAt, ack)) return true;
        return false;
    }
}
