// Task 851 - Implement latency-slash-throughput tradeoff benchmarks for various consistency levels.
// (Tier: Master, CTCI Ch 9 — System Design and Scalability)
//
// A deterministic in-process discrete-event model of a replicated register on N = 5 replicas whose
// one-way latency is latency[i] = base + i * step. A single client pipelines up to D operations; each
// operation reads or writes one key with a monotonically increasing version stamp, and every
// consistency level is a quorum policy over a fixed fan-out:
//
//   eventual      write to the primary only (W = 1, fan-out 1), read from the replica the client is
//                 attached to (attached = opId % N); the other replicas catch up by anti-entropy
//   causal        write goes to the two replicas nearest the client (W = 2, fan-out 2) and the others
//                 catch up by anti-entropy; the read goes to the attached replica with session
//                 guarantees: when the answer is older than the client's own last acknowledged write,
//                 the read escalates to one full fan-out round (read-your-writes repair)//   sequential    write fans out to every replica and is acknowledged only when all of them applied it
//                 (W = N), so coordinator replica 0 always sits at the write frontier; reads go to the
//                 coordinator and the version stamp defines one total order of writes
//   linearizable  majority write quorum and majority read quorum (W = R = N/2 + 1), so every read
//                 quorum intersects every write quorum and a read returns the freshest acknowledged
//                 version found in the intersection
//
// What is simplified (documented, not behavioural): replicas never fail, a replica has unbounded
// capacity so processing is folded into the one-way latency, the network is a star from the client to
// every replica, anti-entropy is a periodic push pulse every gossipInterval ticks, and the value a
// write stores is a deterministic stamp of its version. A second, independent writer session injects
// writes every ten completed operations without touching the pipelined client's session — that is what
// separates "stale with respect to the global frontier" from "violates read-your-writes". The quorum
// policies, the event-driven pipeline, the escalation rule, the anti-entropy convergence and the
// history audit are the real content; scale is expressed through the typed parameters (replica count,
// pipeline depth, op count).
//
// Invariants checked: (1) for every one of the ~4800 operations the measured latency equals the
// closed-form latency of its quorum (2 * latency[quorum-1], plus one fan-out round for an escalated
// causal read) — an oracle that shares no code with the event loop; (2) quorum arithmetic: the
// linearizable and sequential levels never serve a stale read, causal reads never violate
// read-your-writes (they escalate instead, at the measured rate reported in the run), eventual reads do
// both — measured in
// a hand-built deterministic probe; (3) Little's law: throughput * mean latency == pipeline depth;
// (4) the write-only throughput ordering eventual > causal > linearizable > sequential for quorums
// 1 < 2 < 3 < 5, i.e. the latency/throughput tradeoff itself; (5) convergence: every replica reaches
// the final (version, value) pair; the sequential and linearizable levels acknowledge a write only
// when every replica already holds it while the eventual and causal levels never do, and the eventual
// level provably depends on anti-entropy (with the pulses switched off replica 4 never learns
// anything); (6) message accounting equals the analytic 2 * fan-out per operation plus 2 * N per
// escalation; (7) determinism, single-replica degeneracy, zero-operation runs, rejected policies.
using System;
using System.Collections.Generic;
using System.Text;

internal static class Rng851
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound)
    {
        if (bound <= 0) throw new ArgumentOutOfRangeException(nameof(bound));
        return (int)(Next(ref state) % (ulong)bound);
    }
}

internal enum Consistency { Eventual, Causal, Sequential, Linearizable }

internal enum MessageKind { WriteRequest, WriteAck, ReadRequest, ReadResponse, Gossip }

internal sealed class LevelPolicy
{
    internal Consistency Level;
    internal int WriteFanout;
    internal int WriteQuorum;
    internal int ReadFanout;
    internal int ReadQuorum;
    internal bool SessionGuarantees;   // read-your-writes + monotonic reads, with escalation
    internal bool QuorumReads;         // the read quorum must intersect the write quorum
    internal bool ReadFromAttached;    // the read targets the replica the client is attached to
    internal string Name => Level.ToString();

    internal static LevelPolicy For(Consistency level, int replicas)
    {
        if (replicas < 1) throw new ArgumentException("a cluster needs at least one replica", nameof(replicas));
        int majority = replicas / 2 + 1;
        LevelPolicy policy = level switch
        {
            Consistency.Eventual => new LevelPolicy
            {
                WriteFanout = 1, WriteQuorum = 1, ReadFanout = 1, ReadQuorum = 1, ReadFromAttached = true,
            },
            Consistency.Causal => new LevelPolicy
            {
                WriteFanout = Math.Min(replicas, 2), WriteQuorum = Math.Min(replicas, 2), ReadFanout = 1,
                ReadQuorum = 1, SessionGuarantees = true, ReadFromAttached = true,
            },
            Consistency.Sequential => new LevelPolicy
            {
                WriteFanout = replicas, WriteQuorum = replicas, ReadFanout = 1, ReadQuorum = 1,
            },
            Consistency.Linearizable => new LevelPolicy
            {
                WriteFanout = replicas, WriteQuorum = majority, ReadFanout = replicas, ReadQuorum = majority,
                QuorumReads = true,
            },
            _ => throw new ArgumentOutOfRangeException(nameof(level)),
        };
        policy.Level = level;
        policy.Validate(replicas);
        return policy;
    }

    internal void Validate(int replicas)
    {
        if (WriteQuorum < 1 || WriteQuorum > WriteFanout)
            throw new ArgumentException("the write quorum must lie inside the write fan-out: " + Name);
        if (ReadQuorum < 1 || ReadQuorum > ReadFanout)
            throw new ArgumentException("the read quorum must lie inside the read fan-out: " + Name);
        if (QuorumReads && WriteQuorum + ReadQuorum <= replicas)
            throw new ArgumentException("a quorum read must intersect the write quorum: " + Name);
    }
}

internal sealed class Message
{
    internal MessageKind Kind;
    internal bool IsPulse;
    internal int From;
    internal long OpId;
    internal long Version;
    internal int Payload;
}

internal sealed class SimEvent
{
    internal long Time;
    internal long Seq;
    internal int To;
    internal Message Message = new Message();
}

// Binary min-heap ordered by (time, scheduling sequence), so equal-time events keep FIFO order and the
// whole simulation is reproducible.
internal sealed class EventQueue
{
    private readonly List<SimEvent> heap = new List<SimEvent>();

    internal int Count => heap.Count;

    internal void Push(SimEvent item)
    {
        heap.Add(item);
        int i = heap.Count - 1;
        while (i > 0)
        {
            int parent = (i - 1) / 2;
            if (!Less(heap[i], heap[parent])) break;
            Swap(i, parent);
            i = parent;
        }
    }

    internal SimEvent Pop()
    {
        SimEvent top = heap[0];
        int last = heap.Count - 1;
        heap[0] = heap[last];
        heap.RemoveAt(last);
        int i = 0;
        while (true)
        {
            int left = 2 * i + 1, right = left + 1, best = i;
            if (left < heap.Count && Less(heap[left], heap[best])) best = left;
            if (right < heap.Count && Less(heap[right], heap[best])) best = right;
            if (best == i) break;
            Swap(i, best);
            i = best;
        }
        return top;
    }

    private static bool Less(SimEvent a, SimEvent b) => a.Time != b.Time ? a.Time < b.Time : a.Seq < b.Seq;

    private void Swap(int i, int j)
    {
        SimEvent tmp = heap[i];
        heap[i] = heap[j];
        heap[j] = tmp;
    }
}

internal sealed class PendingOp
{
    internal long Id;
    internal bool IsWrite;
    internal long Start;
    internal int Attached;
    internal long Version;          // write: its own version stamp; read: best version observed
    internal int Payload;           // write: the stamp written
    internal int BestPayload;       // read: the stamp that came with the best version
    internal int Outstanding;       // acknowledgements / responses still awaited
    internal bool Foreign;          // written by an independent client session
    internal bool Escalated;
    internal bool Completed;
    internal bool Stale;
    internal bool Violates;
    internal long CompleteTime;
    internal long SessionAtStart;
    internal long AckedAtStart;
}

internal sealed class ProbeResult
{
    internal long Version;
    internal int Payload;
    internal bool Stale;
    internal bool Violates;
    internal bool Escalated;
}

internal sealed class Metrics
{
    internal string Level = "";
    internal int ReplicaCount;
    internal int Ops;
    internal double MeanLatency;
    internal double P50;
    internal double P95;
    internal double MaxLatency;
    internal double Throughput;
    internal double Makespan;
    internal long Messages;
    internal long GossipMessages;
    internal long StaleReads;
    internal long SessionViolations;
    internal long EscalatedReads;
    internal long PayloadMismatches;
    internal long AckedOnEveryReplica;
    internal long ClientWrites;
    internal long ForeignWrites;
    internal long FinalVersion;
    internal bool Converged;
    internal long[] Versions = Array.Empty<long>();
    internal int[] Payloads = Array.Empty<int>();
    internal long ReadCount;
    internal long OracleMismatches;
    internal readonly List<double> Latencies = new List<double>();
    internal readonly List<PendingOp> OpList = new List<PendingOp>();
}

internal sealed class Sim
{
    private const long ForeignCadence = 10;   // one write from the independent session per 10 completed ops

    private readonly int replicas;
    private readonly long baseLatency;
    private readonly long latencyStep;
    private readonly long gossipInterval;
    private readonly LevelPolicy policy;
    private readonly long[] version;
    private readonly int[] payload;
    private readonly EventQueue queue = new EventQueue();
    private readonly Dictionary<long, PendingOp> ops = new Dictionary<long, PendingOp>();
    private readonly List<PendingOp> pending = new List<PendingOp>();
    private readonly Metrics metrics = new Metrics();

    private ulong rng;
    private int writePercent;
    private long seq;
    private long versionCounter;
    private long opCounter;
    private long latestAckedVersion;
    private long sessionVersion;
    private long lastCompletion;
    private long finished;
    private long staleReads;
    private long sessionViolations;
    private long escalatedReads;
    private long payloadMismatches;
    private long messages;
    private long gossipMessages;
    private long foreignWrites;
    private long foreignIssued;
    private long ackedOnEveryReplica;
    private long clientWrites;
    private long readCount;
    private long oracleMismatches;
    private bool converged;

    internal Sim(int replicas, long baseLatency, long latencyStep, long gossipInterval, LevelPolicy policy)
    {
        if (replicas < 1) throw new ArgumentException("a cluster needs at least one replica", nameof(replicas));
        if (baseLatency < 0 || latencyStep < 0) throw new ArgumentException("latencies must be non-negative");
        if (gossipInterval < 1) throw new ArgumentException("the anti-entropy interval must be positive", nameof(gossipInterval));
        this.replicas = replicas;
        this.baseLatency = baseLatency;
        this.latencyStep = latencyStep;
        this.gossipInterval = gossipInterval;
        this.policy = policy ?? throw new ArgumentNullException(nameof(policy));
        version = new long[replicas];
        payload = new int[replicas];
    }

    internal long MessageCount => messages;

    private long Latency(int replica) => baseLatency + latencyStep * replica;

    // A quorum read completes on the arrival of the quorum-th response; the fan-out visits replicas in
    // increasing latency order, so that is replica ReadQuorum-1. A single-replica read completes on the
    // replica it was sent to: the attached one, or the coordinator for the sequential level.
    private int ReadCompletionIndex(int attached) => policy.ReadQuorum > 1
        ? policy.ReadQuorum - 1
        : (policy.ReadFromAttached ? attached : 0);

    private static int Stamp(long version) => version == 0 ? 0 : (int)((version * 7 + 1) % 1000003L);

    internal Metrics Run(int opCount, int depth, int writePercentValue, ulong seed, int pulses)
    {
        if (opCount < 0) throw new ArgumentOutOfRangeException(nameof(opCount));
        if (depth < 1) throw new ArgumentException("the pipeline depth must be positive", nameof(depth));
        if (writePercentValue < 0 || writePercentValue > 100) throw new ArgumentOutOfRangeException(nameof(writePercentValue));
        if (pulses < 0) throw new ArgumentOutOfRangeException(nameof(pulses));
        rng = seed;
        writePercent = writePercentValue;
        for (int k = 1; k <= pulses; k++)
            Push(k * gossipInterval, -1, new Message { Kind = MessageKind.Gossip, IsPulse = true });

        long nextOp = 0;
        while (finished < opCount)
        {
            while (nextOp < opCount && pending.Count < depth) IssueNext(nextOp++);
            if (queue.Count == 0) break;
            Deliver(queue.Pop());
            while ((foreignIssued + 1) * ForeignCadence <= finished)
            {
                IssueForeignWrite(lastCompletion);
                foreignIssued++;
            }
        }
        if (finished != opCount)
            throw new Exception("the pipeline stalled with " + (opCount - finished) + " operations unfinished");

        metrics.Makespan = lastCompletion;
        while (queue.Count > 0) Deliver(queue.Pop());
        converged = AllReplicasAgree();

        var sorted = new List<double>(metrics.Latencies);
        sorted.Sort();
        double total = 0.0;
        foreach (double value in sorted) total += value;
        metrics.ReplicaCount = replicas;
        metrics.Ops = opCount;
        metrics.Level = policy.Name;
        metrics.Messages = messages;
        metrics.GossipMessages = gossipMessages;
        metrics.ForeignWrites = foreignWrites;
        metrics.StaleReads = staleReads;
        metrics.SessionViolations = sessionViolations;
        metrics.EscalatedReads = escalatedReads;
        metrics.PayloadMismatches = payloadMismatches;
        metrics.AckedOnEveryReplica = ackedOnEveryReplica;
        metrics.ClientWrites = clientWrites;
        metrics.ReadCount = readCount;
        metrics.OracleMismatches = oracleMismatches;
        metrics.Converged = converged;
        metrics.FinalVersion = latestAckedVersion;
        metrics.Versions = (long[])version.Clone();
        metrics.Payloads = (int[])payload.Clone();
        metrics.MeanLatency = sorted.Count == 0 ? 0.0 : total / sorted.Count;
        metrics.P50 = Percentile(sorted, 0.50);
        metrics.P95 = Percentile(sorted, 0.95);
        metrics.MaxLatency = sorted.Count == 0 ? 0.0 : sorted[sorted.Count - 1];
        metrics.Throughput = lastCompletion > 0 ? opCount / (double)lastCompletion : 0.0;
        return metrics;
    }

    private static double Percentile(List<double> sorted, double q)
    {
        if (sorted.Count == 0) return 0.0;
        int index = (int)Math.Ceiling(q * sorted.Count) - 1;
        if (index < 0) index = 0;
        if (index >= sorted.Count) index = sorted.Count - 1;
        return sorted[index];
    }

    // One write followed by one read from a chosen replica, with no anti-entropy pulse in flight: the
    // deterministic micro-probe that exposes the consistency differences of the four levels.
    internal ProbeResult ProbeWriteThenRead(int readReplica)
    {
        PendingOp write = IssueWrite(0, 0);
        RunUntil(write);
        PendingOp read = IssueRead(write.CompleteTime + 1, readReplica);
        RunUntil(read);
        return new ProbeResult
        {
            Version = read.Version, Payload = read.BestPayload, Stale = read.Stale,
            Violates = read.Violates, Escalated = read.Escalated,
        };
    }

    private void IssueNext(long opId)
    {
        bool isWrite = Rng851.NextInt(ref rng, 100) < writePercent;
        int attached = (int)(opId % replicas);
        long now = lastCompletion;
        if (isWrite) IssueWrite(now, attached);
        else IssueRead(now, attached);
    }

    private PendingOp IssueWrite(long now, int attached)
    {
        long id = opCounter++;
        long stamp = ++versionCounter;
        var op = new PendingOp
        {
            Id = id, IsWrite = true, Start = now, Attached = attached, Version = stamp,
            Payload = Stamp(stamp), Outstanding = policy.WriteQuorum, AckedAtStart = latestAckedVersion,
        };
        ops[id] = op;
        pending.Add(op);
        metrics.OpList.Add(op);
        for (int r = 0; r < policy.WriteFanout; r++)
        {
            Send(now + Latency(r), r, new Message
            {
                Kind = MessageKind.WriteRequest, OpId = id, Version = stamp, Payload = op.Payload,
            });
        }
        return op;
    }

    private PendingOp IssueRead(long now, int attached)
    {
        long id = opCounter++;
        var op = new PendingOp
        {
            Id = id, IsWrite = false, Start = now, Attached = attached, Outstanding = policy.ReadQuorum,
            SessionAtStart = sessionVersion, AckedAtStart = latestAckedVersion,
        };
        ops[id] = op;
        pending.Add(op);
        metrics.OpList.Add(op);
        for (int r = 0; r < policy.ReadFanout; r++)
        {
            int target = policy.ReadFromAttached ? attached : r;
            Send(now + Latency(target), target, new Message { Kind = MessageKind.ReadRequest, OpId = id });
        }
        return op;
    }

    // An independent writer session whose writes enter the global version order but never touch our
    // client's session: it is exactly what separates "stale with respect to the global frontier" from
    // "violating read-your-writes", the distinction between causal and stronger levels.
    private void IssueForeignWrite(long now)
    {
        long id = opCounter++;
        long stamp = ++versionCounter;
        var op = new PendingOp
        {
            Id = id, IsWrite = true, Foreign = true, Start = now, Version = stamp,
            Payload = Stamp(stamp), Outstanding = policy.WriteQuorum,
        };
        ops[id] = op;
        foreignWrites++;
        for (int r = 0; r < policy.WriteFanout; r++)
        {
            Send(now + Latency(r), r, new Message
            {
                Kind = MessageKind.WriteRequest, OpId = id, Version = stamp, Payload = op.Payload,
            });
        }
    }

    private void RunUntil(PendingOp op)
    {
        while (!op.Completed)
        {
            if (queue.Count == 0) throw new Exception("the probe stalled");
            Deliver(queue.Pop());
        }
    }

    // Client traffic (requests, acknowledgements, responses, escalation rounds) is counted in
    // `messages`; the anti-entropy pulses are counted separately as `gossipMessages`.
    private void Send(long time, int to, Message message)
    {
        messages++;
        Push(time, to, message);
    }

    private void Push(long time, int to, Message message)
    {
        queue.Push(new SimEvent { Time = time, Seq = seq++, To = to, Message = message });
    }

    private void Deliver(SimEvent e)
    {
        Message m = e.Message;
        switch (m.Kind)
        {
            case MessageKind.Gossip:
                if (m.IsPulse) DeliverPulse(e);
                else if (m.Version > version[e.To])
                {
                    version[e.To] = m.Version;
                    payload[e.To] = m.Payload;
                }
                break;

            case MessageKind.WriteRequest:
                if (m.Version > version[e.To])
                {
                    version[e.To] = m.Version;
                    payload[e.To] = m.Payload;
                }
                Send(e.Time + Latency(e.To), e.To, new Message
                {
                    Kind = MessageKind.WriteAck, From = e.To, OpId = m.OpId, Version = m.Version,
                });
                break;

            case MessageKind.WriteAck:
                if (ops.TryGetValue(m.OpId, out PendingOp write) && !write.Completed)
                {
                    write.Outstanding--;
                    if (write.Outstanding <= 0) FinishWrite(e.Time, write);
                }
                break;

            case MessageKind.ReadRequest:
                Send(e.Time + Latency(e.To), e.To, new Message
                {
                    Kind = MessageKind.ReadResponse, From = e.To, OpId = m.OpId,
                    Version = version[e.To], Payload = payload[e.To],
                });
                break;

            case MessageKind.ReadResponse:
                if (ops.TryGetValue(m.OpId, out PendingOp read) && !read.Completed)
                {
                    if (m.Version > read.Version)
                    {
                        read.Version = m.Version;
                        read.BestPayload = m.Payload;
                    }
                    read.Outstanding--;
                    if (read.Outstanding > 0) break;
                    if (policy.SessionGuarantees && !read.Escalated && read.Version < read.SessionAtStart)
                        Escalate(e.Time, read);
                    else
                        FinishRead(e.Time, read);
                }
                break;
        }
    }

    private void DeliverPulse(SimEvent e)
    {
        for (int from = 0; from < replicas; from++)
        {
            for (int to = 0; to < replicas; to++)
            {
                if (from == to) continue;
                gossipMessages++;
                Push(e.Time + Latency(to), to, new Message
                {
                    Kind = MessageKind.Gossip, From = from, Version = version[from], Payload = payload[from],
                });
            }
        }
    }

    // A causal read that missed the client's own last acknowledged write re-reads every replica: that
    // write was acknowledged by WriteQuorum >= 1 replicas, so the escalated round always finds it, which
    // is exactly why the causal level cannot violate read-your-writes.
    private void Escalate(long now, PendingOp read)
    {
        read.Escalated = true;
        escalatedReads++;
        read.Outstanding = replicas;
        for (int target = 0; target < replicas; target++)
            Send(now + Latency(target), target, new Message { Kind = MessageKind.ReadRequest, OpId = read.Id });
    }

    private void FinishWrite(long time, PendingOp op)
    {
        op.Completed = true;
        op.CompleteTime = time;
        if (op.Version > latestAckedVersion) latestAckedVersion = op.Version;
        if (op.Foreign) return;
        // The client's own write is part of its session, so read-your-writes can be enforced on reads.
        if (op.Version > sessionVersion) sessionVersion = op.Version;
        clientWrites++;
        if (AllReplicasHave(op.Version)) ackedOnEveryReplica++;
        if (time > lastCompletion) lastCompletion = time;
        metrics.Latencies.Add(time - op.Start);
        long expected = 2 * Latency(policy.WriteQuorum - 1);
        if (time - op.Start != expected) oracleMismatches++;
        pending.Remove(op);
        finished++;
    }

    private void FinishRead(long time, PendingOp op)
    {
        op.Completed = true;
        op.CompleteTime = time;
        op.Stale = op.Version < op.AckedAtStart;
        op.Violates = op.Version < op.SessionAtStart;
        if (op.Stale) staleReads++;
        if (op.Violates) sessionViolations++;
        if (op.BestPayload != Stamp(op.Version)) payloadMismatches++;
        if (op.Version > sessionVersion) sessionVersion = op.Version;
        if (time > lastCompletion) lastCompletion = time;
        long expected = op.Escalated
            ? 2 * Latency(op.Attached) + 2 * Latency(replicas - 1)
            : 2 * Latency(ReadCompletionIndex(op.Attached));
        metrics.Latencies.Add(time - op.Start);
        if (time - op.Start != expected) oracleMismatches++;
        readCount++;
        pending.Remove(op);
        finished++;
    }

    private bool AllReplicasAgree()
    {
        for (int r = 1; r < replicas; r++)
        {
            if (version[r] != version[0] || payload[r] != payload[0]) return false;
        }
        return true;
    }

    // "Durable cluster-wide at the acknowledgement": every replica already holds at least this version,
    // whatever concurrent writes from other sessions are doing.
    private bool AllReplicasHave(long stamp)
    {
        for (int r = 0; r < replicas; r++)
        {
            if (version[r] < stamp) return false;
        }
        return true;
    }
}

public static class Solution
{
    private const int Replicas = 5;
    private const int Depth = 8;
    private const long BaseLatency = 1;
    private const long LatencyStep = 1;
    private const long GossipInterval = 64;
    private const long ForeignCadence = 10;
    private const int Ops = 600;
    private const int Pulses = 120;

    private static readonly Consistency[] Levels =
    {
        Consistency.Eventual, Consistency.Causal, Consistency.Sequential, Consistency.Linearizable,
    };

    public static void Main()
    {
        // 1-5. Quorum arithmetic of the four consistency levels, and the monotone quorum ladder.
        LevelPolicy eventual = LevelPolicy.For(Consistency.Eventual, Replicas);
        LevelPolicy causal = LevelPolicy.For(Consistency.Causal, Replicas);
        LevelPolicy sequential = LevelPolicy.For(Consistency.Sequential, Replicas);
        LevelPolicy strong = LevelPolicy.For(Consistency.Linearizable, Replicas);
        Check(eventual.WriteFanout == 1 && eventual.WriteQuorum == 1 && !eventual.QuorumReads,
            "the eventual level writes to the primary only");
        Check(causal.SessionGuarantees && causal.WriteQuorum == 2 && causal.WriteFanout == 2
            && causal.ReadFanout == 1,
            "the causal level uses two write acknowledgements and session-guaranteed single reads");
        Check(sequential.WriteQuorum == Replicas && sequential.ReadFanout == 1,
            "the sequential level acknowledges a write only when every replica applied it");
        Check(strong.WriteQuorum + strong.ReadQuorum > Replicas && strong.QuorumReads,
            "the linearizable read quorum intersects the write quorum");
        Check(eventual.WriteQuorum < causal.WriteQuorum && causal.WriteQuorum < strong.WriteQuorum
            && strong.WriteQuorum < sequential.WriteQuorum,
            "write quorums form the ladder 1 < 2 < 3 < 5");

        // 6-8. Malformed policies are rejected instead of silently benchmarking nonsense.
        Check(Throws(() => new LevelPolicy { WriteFanout = 1, WriteQuorum = 2, ReadFanout = 1, ReadQuorum = 1 }
            .Validate(Replicas)), "a write quorum outside the fan-out is rejected");
        Check(Throws(() => new LevelPolicy
        {
            WriteFanout = 5, WriteQuorum = 1, ReadFanout = 5, ReadQuorum = 1, QuorumReads = true,
        }.Validate(Replicas)), "non-intersecting quorums are rejected");
        Check(Throws(() => LevelPolicy.For(Consistency.Causal, 0)), "a zero-replica cluster is rejected");

        // 9-14. Write-only benchmark: the pure latency/throughput tradeoff of the quorum sizes.
        var writeRuns = new Dictionary<Consistency, Metrics>();
        foreach (Consistency level in Levels)
        {
            var sim = new Sim(Replicas, BaseLatency, LatencyStep, GossipInterval, LevelPolicy.For(level, Replicas));
            writeRuns[level] = sim.Run(Ops, Depth, 100, 0x851AUL, Pulses);
        }
        Console.WriteLine("write-only benchmark (ops=" + Ops + ", depth=" + Depth + "):");
        foreach (Consistency level in Levels)
        {
            Metrics m = writeRuns[level];
            Console.WriteLine("  " + m.Level.PadRight(13) + "mean latency=" + m.MeanLatency.ToString("F2")
                + " p95=" + m.P95.ToString("F2") + " throughput=" + m.Throughput.ToString("F4")
                + " acked-on-all=" + m.AckedOnEveryReplica + "/" + m.ClientWrites);
        }
        bool oracleHolds = true;
        foreach (Consistency level in Levels)
        {
            if (writeRuns[level].OracleMismatches != 0) oracleHolds = false;
        }
        Check(oracleHolds, "every write latency equals the closed-form 2 * latency[quorum-1]");
        Check(writeRuns[Consistency.Eventual].Throughput > writeRuns[Consistency.Causal].Throughput
            && writeRuns[Consistency.Causal].Throughput > writeRuns[Consistency.Linearizable].Throughput
            && writeRuns[Consistency.Linearizable].Throughput > writeRuns[Consistency.Sequential].Throughput,
            "write throughput decreases as the write quorum grows");
        Check(writeRuns[Consistency.Eventual].MeanLatency == 2 * BaseLatency
            && writeRuns[Consistency.Sequential].MeanLatency == 2 * (BaseLatency + LatencyStep * (Replicas - 1)),
            "the closed-form latencies of the cheapest and the most expensive write paths are reproduced exactly");
        foreach (Consistency level in Levels)
        {
            Metrics m = writeRuns[level];
            Check(Math.Abs(m.Throughput * m.MeanLatency - Depth) < 0.01 * Depth,
                "Little's law holds for the write-only pipeline at the " + m.Level + " level");
        }
        Check(writeRuns[Consistency.Eventual].MeanLatency < writeRuns[Consistency.Sequential].MeanLatency
            && writeRuns[Consistency.Eventual].Throughput > writeRuns[Consistency.Sequential].Throughput,
            "the eventual level is both faster and higher throughput than the sequential level on writes");

        // 15-21. Mixed read/write benchmark: consistency audit, distribution and Little's law.
        var mixedRuns = new Dictionary<Consistency, Metrics>();
        var mixedSims = new Dictionary<Consistency, Sim>();
        foreach (Consistency level in Levels)
        {
            var sim = new Sim(Replicas, BaseLatency, LatencyStep, GossipInterval, LevelPolicy.For(level, Replicas));
            mixedSims[level] = sim;
            mixedRuns[level] = sim.Run(Ops, Depth, 50, 0x851BUL, Pulses);
        }
        Console.WriteLine("mixed benchmark (50% writes):");
        foreach (Consistency level in Levels)
        {
            Metrics m = mixedRuns[level];
            Console.WriteLine("  " + m.Level.PadRight(13) + "mean=" + m.MeanLatency.ToString("F2")
                + " p50=" + m.P50.ToString("F2") + " p95=" + m.P95.ToString("F2")
                + " throughput=" + m.Throughput.ToString("F4") + " stale-reads=" + m.StaleReads
                + " session-violations=" + m.SessionViolations + " escalated=" + m.EscalatedReads
                + " acked-on-all=" + m.AckedOnEveryReplica + "/" + m.ClientWrites + " messages=" + m.Messages);
        }
        Check(mixedRuns[Consistency.Linearizable].StaleReads == 0
            && mixedRuns[Consistency.Sequential].StaleReads == 0,
            "the linearizable and sequential levels never serve a stale read");
        Check(mixedRuns[Consistency.Causal].SessionViolations == 0
            && mixedRuns[Consistency.Linearizable].SessionViolations == 0
            && mixedRuns[Consistency.Sequential].SessionViolations == 0,
            "causal, sequential and linearizable reads never violate read-your-writes");
        Check(mixedRuns[Consistency.Causal].StaleReads > 0
            && mixedRuns[Consistency.Eventual].StaleReads > mixedRuns[Consistency.Causal].StaleReads,
            "causal reads are stale only with respect to the global frontier (" + mixedRuns[Consistency.Causal].StaleReads
            + "), strictly fewer than the eventual level (" + mixedRuns[Consistency.Eventual].StaleReads + ")");
        Check(mixedRuns[Consistency.Eventual].SessionViolations > 0
            && mixedRuns[Consistency.Eventual].StaleReads > 0,
            "the eventual level does serve stale reads (" + mixedRuns[Consistency.Eventual].StaleReads
            + ") and does violate read-your-writes (" + mixedRuns[Consistency.Eventual].SessionViolations + ")");
        Check(mixedRuns[Consistency.Eventual].EscalatedReads == 0
            && mixedRuns[Consistency.Causal].EscalatedReads > 0,
            "only the causal level escalates reads, at the measured rate "
            + mixedRuns[Consistency.Causal].EscalatedReads + "/" + mixedRuns[Consistency.Causal].ReadCount);
        long mismatches = 0;
        foreach (Consistency level in Levels) mismatches += mixedRuns[level].PayloadMismatches;
        Check(mismatches == 0, "no read returned a value that disagrees with its version stamp");
        long readOracleMismatches = 0;
        foreach (Consistency level in Levels) readOracleMismatches += mixedRuns[level].OracleMismatches;
        Check(readOracleMismatches == 0,
            "every read latency equals its closed-form value (2 * latency[target], plus a round when escalated)");
        foreach (Consistency level in Levels)
        {
            Metrics m = mixedRuns[level];
            Check(m.P50 <= m.P95 && m.P95 <= m.MaxLatency && m.MeanLatency > 0,
                "the latency distribution is ordered at the " + m.Level + " level");
            Check(Math.Abs(m.Throughput * m.MeanLatency - Depth) < 0.10 * Depth,
                "Little's law holds for the mixed pipeline at the " + m.Level + " level");
        }

        // 22. Message accounting against the analytic fan-out cost, recomputed independently.
        bool accounting = true;
        foreach (Consistency level in Levels)
        {
            LevelPolicy policy = LevelPolicy.For(level, Replicas);
            long expected = 0;
            foreach (PendingOp op in mixedRuns[level].OpList)
            {
                expected += op.IsWrite ? 2L * policy.WriteFanout : 2L * policy.ReadFanout;
                if (op.Escalated) expected += 2L * Replicas;
            }
            expected += 2L * policy.WriteFanout * mixedRuns[level].ForeignWrites;
            if (expected != mixedSims[level].MessageCount) accounting = false;
        }
        Check(accounting, "message counts equal 2 * fan-out per operation plus 2 * N per escalation");

        // 23-26. Convergence: agreement at the acknowledgement, and anti-entropy as the catch-up mechanism.
        bool convergedAll = true;
        foreach (Consistency level in Levels)
        {
            Metrics m = mixedRuns[level];
            if (!m.Converged) convergedAll = false;
            for (int r = 0; r < Replicas; r++)
            {
                if (m.Versions[r] != m.FinalVersion || m.Payloads[r] != (int)((m.FinalVersion * 7 + 1) % 1000003L))
                    convergedAll = false;
            }
        }
        Check(convergedAll, "every replica converges to the final (version, value) pair");
        Check(mixedRuns[Consistency.Sequential].AckedOnEveryReplica == mixedRuns[Consistency.Sequential].ClientWrites
            && mixedRuns[Consistency.Linearizable].AckedOnEveryReplica == mixedRuns[Consistency.Linearizable].ClientWrites
            && mixedRuns[Consistency.Eventual].AckedOnEveryReplica == 0
            && mixedRuns[Consistency.Causal].AckedOnEveryReplica == 0,
            "only the sequential and linearizable levels acknowledge a write that every replica already holds");
        var quietEventual = new Sim(Replicas, BaseLatency, LatencyStep, GossipInterval, LevelPolicy.For(Consistency.Eventual, Replicas));
        Metrics quietEventualMetrics = quietEventual.Run(100, Depth, 100, 0x851EUL, 0);
        Check(!quietEventualMetrics.Converged && quietEventualMetrics.Versions[0] == quietEventualMetrics.FinalVersion
            && quietEventualMetrics.Versions[4] == 0,
            "without anti-entropy the eventual level never converges (replica 0 ahead, replica 4 empty)");
        var quietSequential = new Sim(Replicas, BaseLatency, LatencyStep, GossipInterval, LevelPolicy.For(Consistency.Sequential, Replicas));
        Metrics quietSequentialMetrics = quietSequential.Run(100, Depth, 100, 0x851EUL, 0);
        Check(quietSequentialMetrics.Converged,
            "the sequential level needs no anti-entropy because every write quorum is the whole cluster");

        // 26-29. Deterministic micro-probes of the consistency levels.
        ProbeResult eventualProbe = Probe(Consistency.Eventual, Replicas - 1);
        ProbeResult causalProbe = Probe(Consistency.Causal, Replicas - 1);
        ProbeResult sequentialProbe = Probe(Consistency.Sequential, Replicas - 1);
        ProbeResult strongProbe = Probe(Consistency.Linearizable, Replicas - 1);
        Console.WriteLine("read-your-writes probe (write on the primary, then read replica " + (Replicas - 1) + "):");
        Console.WriteLine("  eventual=" + eventualProbe.Version + " stale=" + eventualProbe.Stale
            + " | causal=" + causalProbe.Version + " escalated=" + causalProbe.Escalated
            + " | sequential=" + sequentialProbe.Version + " | linearizable=" + strongProbe.Version);
        Check(eventualProbe.Version == 0 && eventualProbe.Stale && eventualProbe.Violates,
            "an eventual read from a non-primary replica misses the acknowledged write");
        Check(causalProbe.Version == 1 && !causalProbe.Stale && !causalProbe.Violates && causalProbe.Escalated,
            "a causal read escalates and then observes the client's own write");
        Check(sequentialProbe.Version == 1 && !sequentialProbe.Stale
            && strongProbe.Version == 1 && !strongProbe.Stale && strongProbe.Payload == 8,
            "sequential and linearizable reads observe the acknowledged write on the first round");
        Check(!strongProbe.Escalated && !sequentialProbe.Escalated && causalProbe.Payload == 8,
            "only the causal probe escalates, and the returned payload matches the stamp of version 1");

        // 30-32. Determinism and degenerate configurations.
        var repeat = new Sim(Replicas, BaseLatency, LatencyStep, GossipInterval, LevelPolicy.For(Consistency.Causal, Replicas));
        Metrics repeatMetrics = repeat.Run(Ops, Depth, 50, 0x851BUL, Pulses);
        Metrics causalMixed = mixedRuns[Consistency.Causal];
        Check(repeatMetrics.MeanLatency == causalMixed.MeanLatency && repeatMetrics.Messages == causalMixed.Messages
            && repeatMetrics.StaleReads == causalMixed.StaleReads && repeatMetrics.EscalatedReads == causalMixed.EscalatedReads,
            "an identical seed reproduces the benchmark bit for bit");
        var single = new Sim(1, BaseLatency, LatencyStep, GossipInterval, LevelPolicy.For(Consistency.Linearizable, 1));
        Metrics singleMetrics = single.Run(50, 4, 50, 0x851CUL, Pulses);
        Check(singleMetrics.Converged && singleMetrics.AckedOnEveryReplica == singleMetrics.ClientWrites
            && singleMetrics.MeanLatency == 2 * BaseLatency && singleMetrics.StaleReads == 0
            && singleMetrics.SessionViolations == 0,
            "a single-replica cluster gives identical latency, no staleness and immediate convergence");
        var empty = new Sim(Replicas, BaseLatency, LatencyStep, GossipInterval, LevelPolicy.For(Consistency.Eventual, Replicas));
        Metrics emptyMetrics = empty.Run(0, Depth, 50, 0x851DUL, Pulses);
        Check(emptyMetrics.Ops == 0 && emptyMetrics.MeanLatency == 0.0 && emptyMetrics.Throughput == 0.0
            && emptyMetrics.Converged && emptyMetrics.FinalVersion == 0,
            "a zero-operation benchmark converges immediately and reports no latency");

        Console.WriteLine("PASS");
    }

    private static ProbeResult Probe(Consistency level, int readReplica)
    {
        var sim = new Sim(Replicas, BaseLatency, LatencyStep, GossipInterval, LevelPolicy.For(level, Replicas));
        return sim.ProbeWriteThenRead(readReplica);
    }

    private static bool Throws(Action action)
    {
        try
        {
            action();
            return false;
        }
        catch (ArgumentException)
        {
            return true;
        }
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
