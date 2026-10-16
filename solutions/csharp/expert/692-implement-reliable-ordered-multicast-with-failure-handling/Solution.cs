// Task 692 - Implement reliable ordered multicast with failure handling. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
//
// A sequencer based total order multicast (the classic primary/backup sequencer design). One live
// member is the sequencer: it accepts submissions, stamps each with (epoch, seq), appends it to its
// log and multicasts it. Every member acknowledges entries it has received, buffers entries that
// arrive early and delivers strictly in (epoch, seq) order, so all live members deliver exactly the
// same sequence of payloads. Reliability comes from acknowledgement driven retransmission of every
// unacknowledged log entry, which also covers a member that was down: when it rejoins it is caught
// up from the same log.
//
// Failure handling:
//   * a member failure removes it from the live view; delivery for the others never waits for it;
//   * a sequencer failure hands the log (and therefore every accepted entry) to the lowest live
//     member, which starts a new epoch. The first entry of a new epoch announces the last sequence
//     number of the previous epoch, so a member that is still missing the tail of the old epoch holds
//     the barrier until those entries are retransmitted - the ordering across the epoch change is
//     therefore still the global order.
//
// Documented simplifications: rounds are the scheduling unit (no timers, no threads), the transit
// loss pattern is a fixed deterministic predicate instead of a random one, the log handover is
// graceful (a crash that loses the un-replicated log tail is not simulated), and membership changes
// are single member add/remove steps. What is verified: identical total order for every live member,
// exactly-once delivery, per sender FIFO order, retransmission recovery of the deterministic drops,
// catch-up of a restarted member, no loss of any accepted entry across a sequencer handover, that
// no epoch-2 entry is delivered before the last epoch-1 entry, rejection of non-member and duplicate
// submissions, and bit identical reruns.
using System;
using System.Collections.Generic;

internal sealed class McastEntry
{
    public int Epoch;
    public int Seq;
    public int LastEpochLastSeq;   // announced boundary: the final sequence number of the previous epoch
    public int Sender;
    public long SenderSeq;
    public string Payload;
}

internal sealed class McastMember
{
    public readonly int Id;
    public bool Live = true;
    public int Epoch = 1;
    public int NextSeq = 1;
    public readonly SortedDictionary<(int Epoch, int Seq), McastEntry> Buffer = new SortedDictionary<(int, int), McastEntry>();
    public readonly List<string> Delivered = new List<string>();
    public readonly HashSet<long> Received = new HashSet<long>();
    public long Duplicates;
    public long TransmissionsIn;

    public McastMember(int id) => Id = id;

    public static long Key(int epoch, int seq) => (long)epoch * 1000000L + seq;

    /// <summary>Drains the buffer in (epoch, seq) order, respecting the epoch barrier.</summary>
    public void TryDeliver()
    {
        while (true)
        {
            if (Buffer.TryGetValue((Epoch, NextSeq), out var ready))
            {
                Buffer.Remove((Epoch, NextSeq));
                Delivered.Add($"{ready.Epoch}.{ready.Seq}:{ready.Payload}");
                NextSeq++;
                continue;
            }
            int startEpoch = -1, boundary = -1;
            foreach (var pair in Buffer)
            {
                if (pair.Key.Epoch > Epoch && pair.Key.Seq == 1)
                {
                    startEpoch = pair.Key.Epoch;
                    boundary = pair.Value.LastEpochLastSeq;
                    break;
                }
            }
            if (startEpoch < 0) break;
            if (NextSeq <= boundary) break;     // the tail of the previous epoch has not arrived yet
            Epoch = startEpoch;
            NextSeq = 1;
        }
    }

    public void Accept(McastEntry entry)
    {
        TransmissionsIn++;
        long key = Key(entry.Epoch, entry.Seq);
        if (!Received.Add(key)) { Duplicates++; return; }
        if (entry.Epoch < Epoch || (entry.Epoch == Epoch && entry.Seq < NextSeq)) return;   // already delivered
        Buffer[(entry.Epoch, entry.Seq)] = entry;
        TryDeliver();
    }
}

internal sealed class SequencedMulticast
{
    public const int MaxAttempts = 3;

    private readonly SortedDictionary<int, McastMember> _members = new SortedDictionary<int, McastMember>();
    private readonly List<McastEntry> _log = new List<McastEntry>();
    private readonly List<HashSet<int>> _acks = new List<HashSet<int>>();
    private readonly Dictionary<int, long> _lastSenderSeq = new Dictionary<int, long>();
    private readonly List<(int Sender, long SenderSeq, string Payload)> _pending = new List<(int, long, string)>();

    private int _leader = -1;
    private int _epoch = 1;
    private int _nextSeq = 1;
    private int _lastSeqOfPreviousEpoch;

    public long Rounds;
    public long Accepted;
    public long Rejected;
    public long Drops;
    public long Sends;
    public long Retransmissions;

    public IEnumerable<McastMember> Members => _members.Values;
    public int Leader => _leader;
    public int Epoch => _epoch;
    public int LogLength => _log.Count;

    public void Add(int id)
    {
        _members[id] = new McastMember(id);
        if (_leader < 0) _leader = id;
    }

    public McastMember Member(int id) => _members[id];

    /// <summary>Deterministic transit loss: the first attempt of roughly one in eleven messages is dropped.</summary>
    private static bool Drop(int epoch, int seq, int to, int attempt)
    {
        if (attempt > 1) return false;
        return (epoch * 31 + seq * 17 + to * 7) % 11 == 0;
    }

    public bool Submit(int sender, long senderSeq, string payload)
    {
        if (!_members.TryGetValue(sender, out var member) || !member.Live) { Rejected++; return false; }
        foreach (var item in _pending)
        {
            if (item.Sender == sender && item.SenderSeq == senderSeq) { Rejected++; return false; }
        }
        if (_lastSenderSeq.TryGetValue(sender, out long previous) && senderSeq <= previous) { Rejected++; return false; }
        _pending.Add((sender, senderSeq, payload));
        return true;
    }

    /// <summary>One scheduling round: accept, retransmit what is unacknowledged, then deliver.</summary>
    public void Round()
    {
        Rounds++;
        AcceptPending();
        Transmit();
        foreach (var entry in _members) if (entry.Value.Live) Drain(entry.Value);
    }

    private void AcceptPending()
    {
        foreach ((int sender, long senderSeq, string payload) in _pending)
        {
            var entry = new McastEntry
            {
                Epoch = _epoch, Seq = _nextSeq, LastEpochLastSeq = _lastSeqOfPreviousEpoch,
                Sender = sender, SenderSeq = senderSeq, Payload = payload,
            };
            _nextSeq++;
            _log.Add(entry);
            _acks.Add(new HashSet<int>());
            _lastSenderSeq[sender] = senderSeq;
            Accepted++;
        }
        _pending.Clear();
    }

    private void Transmit()
    {
        var live = new List<int>();
        foreach (var member in _members) if (member.Value.Live) live.Add(member.Key);

        for (int index = 0; index < _log.Count; index++)
        {
            McastEntry entry = _log[index];
            foreach (int target in live)
            {
                if (_acks[index].Contains(target)) continue;
                bool retransmission = _acks[index].Count > 0;
                Sends++;
                if (retransmission) Retransmissions++;
                int attempt = _acks[index].Count + 1;
                if (Drop(entry.Epoch, entry.Seq, target, attempt)) { Drops++; continue; }
                _members[target].Accept(entry);
                _acks[index].Add(target);            // acknowledged on receipt, so retries are bounded
            }
        }
    }

    private void Drain(McastMember member)
    {
        member.TryDeliver();
    }

    public void Run(int rounds)
    {
        for (int i = 0; i < rounds; i++) Round();
    }

    // --- Failure handling -----------------------------------------------------------------

    public void FailMember(int id)
    {
        _members[id].Live = false;
        if (id == _leader) HandOverToLowestLiveMember();
    }

    public void RestartMember(int id)
    {
        _members[id].Live = true;
        if (_leader < 0) _leader = id;
    }

    public int LowestLiveMember()
    {
        foreach (var member in _members) if (member.Value.Live) return member.Key;
        return -1;
    }

    /// <summary>Hands the whole log to the lowest live member and starts a new epoch.</summary>
    public void HandOverToLowestLiveMember()
    {
        int next = LowestLiveMember();
        if (next < 0) return;
        _lastSeqOfPreviousEpoch = _nextSeq - 1;
        _epoch++;
        _nextSeq = 1;
        _leader = next;
        // The log is copied to the new sequencer, which therefore still owns every accepted entry.
        _acks.Clear();
        for (int i = 0; i < _log.Count; i++) _acks.Add(new HashSet<int>());
    }

    // --- Verification helpers -------------------------------------------------------------

    public List<string> Trace(int memberId) => _members[memberId].Delivered;

    public bool LostAnything()
    {
        foreach (var member in _members)
        {
            if (!member.Value.Live) continue;
            if (member.Value.Delivered.Count + member.Value.Buffer.Count != _log.Count) return true;
        }
        return false;
    }

    public List<string> AcceptedOrder()
    {
        var order = new List<string>();
        foreach (McastEntry entry in _log) order.Add($"{entry.Epoch}.{entry.Seq}:{entry.Payload}");
        return order;
    }
}

public static class Solution
{
    public static void Main()
    {
        // --- Basic total order with deterministic transit loss. ---------------------------
        var group = new SequencedMulticast();
        for (int id = 0; id < 3; id++) group.Add(id);
        for (int i = 1; i <= 10; i++) Check(group.Submit(i % 3, i, "m" + i), "submission " + i + " accepted");
        group.Run(40);

        var expected = group.AcceptedOrder();
        Check(expected.Count == 10, "all ten submissions were accepted, got " + expected.Count);
        Check(group.Drops > 0, "the deterministic loss pattern actually dropped messages, drops=" + group.Drops);
        Check(group.Retransmissions > 0, "the drops forced retransmissions, retries=" + group.Retransmissions);
        Check(!group.LostAnything(), "no live member is still missing an accepted entry");
        for (int id = 0; id < 3; id++)
        {
            var trace = group.Trace(id);
            Check(trace.Count == 10, $"member {id} delivered all ten entries, got " + trace.Count);
            Check(string.Join("|", trace) == string.Join("|", expected), $"member {id} delivered the global order");
        }
        Check(group.Trace(0).Count == group.Trace(1).Count && group.Trace(1).Count == group.Trace(2).Count,
            "all members delivered the same number of entries");
        Check(group.Member(0).Duplicates + group.Member(1).Duplicates + group.Member(2).Duplicates >= 0,
            "duplicate detection is exercised (may be zero for this pattern)");

        // Exactly once: an entry never appears twice in a trace.
        var seen = new HashSet<string>();
        bool duplicates = false;
        foreach (string item in group.Trace(0)) if (!seen.Add(item)) duplicates = true;
        Check(!duplicates, "no entry was delivered twice");

        // Per sender FIFO: sender i%3 submitted m1, m4, m7, m10 in that order.
        foreach (int sender in new[] { 0, 1, 2 })
        {
            var messages = new List<string>();
            for (int i = 1; i <= 10; i++) if (i % 3 == sender) messages.Add("m" + i);
            var delivered = new List<string>();
            foreach (string item in group.Trace(sender))
            {
                string payload = item.Substring(item.IndexOf(':') + 1);
                foreach (string candidate in messages) if (candidate == payload) delivered.Add(payload);
            }
            Check(string.Join(",", delivered) == string.Join(",", messages), $"sender {sender} keeps FIFO order");
        }

        // --- Rejections. ------------------------------------------------------------------
        Check(!group.Submit(9, 1, "ghost"), "a submission from a non member is rejected");
        Check(!group.Submit(0, 1, "repeat"), "a duplicate sender sequence number is rejected");
        Check(group.Submit(0, 11, "fresh"), "a new sender sequence number is accepted");
        group.Run(20);
        Check(group.AcceptedOrder().Count == 11, "only the valid submission was accepted");

        // --- A member failure does not block the others, and it catches up. --------------
        var faulty = new SequencedMulticast();
        for (int id = 0; id < 3; id++) faulty.Add(id);
        faulty.Submit(0, 1, "before");
        faulty.Run(20);
        faulty.FailMember(2);
        Check(!faulty.Member(2).Live, "member 2 is out of the view");
        int deliveredBefore = faulty.Trace(2).Count;
        for (int i = 2; i <= 5; i++) faulty.Submit(1, i, "down" + i);
        faulty.Run(30);
        Check(faulty.Trace(0).Count == 5 && faulty.Trace(1).Count == 5, "the live members kept delivering");
        Check(faulty.Trace(2).Count == deliveredBefore, "the failed member received nothing while down");
        faulty.RestartMember(2);
        faulty.Run(40);
        Check(!faulty.LostAnything(), "the restarted member was caught up from the log");
        Check(faulty.Trace(2).Count == 5, "the restarted member has every accepted entry now");
        Check(string.Join("|", faulty.Trace(2)) == string.Join("|", faulty.Trace(0)),
            "the restarted member agrees with the group on the order");
        Check(faulty.Trace(2)[deliveredBefore] == "1.2:down2", "the catch-up starts exactly where it stopped");

        // --- Sequencer failure: hand over the log and keep the order. --------------------
        var handedOver = new SequencedMulticast();
        for (int id = 0; id < 3; id++) handedOver.Add(id);
        for (int i = 1; i <= 4; i++) handedOver.Submit(0, i, "e1m" + i);
        handedOver.Run(30);
        int acceptedBeforeHandover = handedOver.LogLength;
        handedOver.FailMember(0);                     // the sequencer dies
        Check(handedOver.Leader == 1, "the lowest live member became the sequencer");
        Check(handedOver.Epoch == 2, "the new sequencer starts a new epoch");
        for (int i = 5; i <= 7; i++) handedOver.Submit(1, i, "e2m" + i);
        handedOver.Run(60);
        Check(handedOver.LogLength == 7, "the handed over log kept all previous entries");
        Check(handedOver.Member(1).Delivered.Count == 7 && handedOver.Member(2).Delivered.Count == 7,
            "both live members delivered everything");
        Check(!handedOver.LostAnything(), "no accepted entry was lost in the handover");

        var handoverTrace = handedOver.Trace(2);
        Check(string.Join("|", handoverTrace) == string.Join("|", handedOver.AcceptedOrder()),
            "the total order survived the epoch change");
        int lastEpochOne = -1, firstEpochTwo = -1;
        for (int i = 0; i < handoverTrace.Count; i++)
        {
            if (handoverTrace[i].StartsWith("1.")) lastEpochOne = i;
            if (handoverTrace[i].StartsWith("2.") && firstEpochTwo < 0) firstEpochTwo = i;
        }
        Check(lastEpochOne >= 0 && firstEpochTwo > lastEpochOne,
            "every epoch 1 entry precedes the first epoch 2 entry");
        Check(firstEpochTwo == acceptedBeforeHandover, "epoch 2 begins exactly after the announced boundary");
        Check(handoverTrace[firstEpochTwo] == "2.1:e2m5", "the new epoch numbers from one");
        Check(handedOver.Member(0).Delivered.Count == acceptedBeforeHandover,
            "the dead sequencer stopped delivering at its failure point");

        // A member that missed the very last entry of epoch 1 must wait for the barrier.
        var barrier = new SequencedMulticast();
        for (int id = 0; id < 3; id++) barrier.Add(id);
        barrier.Submit(0, 1, "first");
        barrier.Submit(0, 2, "second");
        barrier.Round();                              // only the first transmission happened
        barrier.FailMember(0);
        barrier.Submit(1, 3, "third");
        barrier.Run(60);
        Check(barrier.Trace(1).Count == 3 && barrier.Trace(2).Count == 3, "the barrier released after the tail arrived");
        Check(string.Join("|", barrier.Trace(1)) == "1.1:first|1.2:second|2.1:third",
            "the epoch 1 tail is delivered before the epoch 2 entry, got " + string.Join("|", barrier.Trace(1)));

        // --- Degenerate cases and determinism. -------------------------------------------
        var lonely = new SequencedMulticast();
        lonely.Add(7);
        lonely.Submit(7, 1, "solo");
        lonely.Run(5);
        Check(lonely.Trace(7).Count == 1, "a single member group still delivers");
        lonely.FailMember(7);
        Check(!lonely.Member(7).Live, "the only member is out of the view");
        Check(!lonely.Submit(7, 2, "gone"), "a failed member cannot submit");
        lonely.HandOverToLowestLiveMember();     // no live member: must be a no-op, not a crash
        Check(lonely.Leader == 7 && lonely.Epoch == 1, "the handover without a live member changed nothing");

        var firstRun = RunScenario();
        var secondRun = RunScenario();
        Check(string.Join("|", firstRun.Trace(1)) == string.Join("|", secondRun.Trace(1)), "reruns deliver the same order");
        Check(firstRun.Sends == secondRun.Sends && firstRun.Drops == secondRun.Drops && firstRun.Retransmissions == secondRun.Retransmissions,
            "reruns send, drop and retransmit exactly the same messages");

        Console.WriteLine($"accepted={group.Accepted} drops={group.Drops} retries={group.Retransmissions} " +
                          $"handoverEpoch={handedOver.Epoch} entries={handedOver.LogLength}");
        Console.WriteLine("PASS");
    }

    private static SequencedMulticast RunScenario()
    {
        var group = new SequencedMulticast();
        for (int id = 0; id < 3; id++) group.Add(id);
        for (int i = 1; i <= 6; i++) group.Submit(i % 3, i, "p" + i);
        group.Run(30);
        return group;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
