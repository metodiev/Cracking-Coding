// Task 624 - Implement RCU-style read-copy-update sketch with tests. (Tier: Expert, CTCI Ch 15 - Threads and Locks)
//
// RCU splits an update into (a) copy, (b) mutate the copy, (c) publish the new pointer, and
// (d) defer freeing the old version until a grace period has elapsed. Readers never lock and
// never see a half-updated structure because a published version is never mutated again: a
// reader only has to snapshot the pointer once, and it may keep using that snapshot for as
// long as its critical section lasts.
//
//   * ReadLock(thread) grabs the current version and remembers the version's epoch; that epoch
//     is what identifies a reader as "pre-existing" with respect to a later publish.
//   * Publish moves the current version to the retired list and installs a brand new version.
//   * TrySynchronize is the grace-period poll. It succeeds only when no reader (on any
//     simulated thread) still holds a version with an epoch <= a retired version's epoch;
//     a reader that entered *after* the publish therefore never delays the grace period.
//     On success the retired versions are physically reclaimed exactly once and every pending
//     call_rcu-style callback runs.
//
// Concurrency is *simulated deterministically* - no threads, no sleeps, no timers: a thread is
// a slot index and the tests drive enter/read/exit and publish/poll steps explicitly.
//
// Simplification (documented per the authoring contract): the domain is a single-pointer
// published snapshot (the standard "RCU-protected pointer"), not a full RCU-protected hash
// table or a hierarchical RCU tree, and the memory barrier discussion is replaced by the
// epoch bookkeeping. Verified invariants: a version that a reader holds is never reclaimed,
// readers that enter after a publish never delay that publish's grace period, callbacks run
// exactly once, and published versions are immutable (checksum stable under later updates).
using System;
using System.Collections.Generic;

#nullable enable

internal sealed class RcuVersion
{
    public readonly long Epoch;
    public readonly Dictionary<string, int> Data;
    public readonly long Checksum;
    public bool Reclaimed;

    public RcuVersion(long epoch, Dictionary<string, int> data)
    {
        Epoch = epoch;
        Data = data;
        Checksum = ComputeChecksum(data);
    }

    /// <summary>Order-independent content hash of an immutable snapshot (FNV-1a over sorted keys).</summary>
    public static long ComputeChecksum(Dictionary<string, int> data)
    {
        var keys = new List<string>(data.Keys);
        keys.Sort(StringComparer.Ordinal);
        long hash = 1469598103934665603L;
        foreach (string key in keys)
        {
            foreach (char c in key)
            {
                hash ^= c;
                hash *= 1099511628211L;
            }
            hash ^= data[key];
            hash *= 1099511628211L;
        }
        return hash;
    }
}

internal sealed class RcuCallback
{
    public Action Body = null!;
    public bool Ran;
}

internal sealed class RcuDomain
{
    private readonly long[] readerEpoch;
    private readonly List<RcuVersion> retiredVersions = new List<RcuVersion>();
    private readonly List<RcuCallback> callbacks = new List<RcuCallback>();
    private RcuVersion current;

    public int Publishes;
    public int Reclaims;
    public int GracePeriodPolls;
    public int CallbacksRun;
    public readonly List<long> ReclaimedEpochs = new List<long>();

    public RcuDomain(int threadCount, Dictionary<string, int> initial)
    {
        readerEpoch = new long[threadCount];
        for (int t = 0; t < threadCount; t++) readerEpoch[t] = -1;
        current = new RcuVersion(0, initial);
    }

    public RcuVersion Current => current;
    public int PendingRetired => retiredVersions.Count;

    public RcuVersion ReadLock(int thread)
    {
        RcuVersion snapshot = current;
        readerEpoch[thread] = snapshot.Epoch;
        return snapshot;
    }

    public void ReadUnlock(int thread) => readerEpoch[thread] = -1;

    public bool IsReading(int thread) => readerEpoch[thread] >= 0;

    /// <summary>Copy step of an update: hand the writer a private copy of the current data.</summary>
    public Dictionary<string, int> StartUpdate() => new Dictionary<string, int>(current.Data);

    /// <summary>Publish step: install the new version, retire the old one for a grace period.</summary>
    public void Publish(Dictionary<string, int> updated)
    {
        retiredVersions.Add(current);
        current = new RcuVersion(current.Epoch + 1, updated);
        Publishes++;
    }

    public void CallRcu(Action body) => callbacks.Add(new RcuCallback { Body = body });

    /// <summary>Grace-period poll: true once every pre-existing reader has left its critical section.</summary>
    public bool TrySynchronize()
    {
        GracePeriodPolls++;
        foreach (RcuVersion old in retiredVersions)
        {
            for (int t = 0; t < readerEpoch.Length; t++)
            {
                long held = readerEpoch[t];
                if (held >= 0 && held <= old.Epoch) return false;
            }
        }
        foreach (RcuVersion old in retiredVersions)
        {
            old.Reclaimed = true;
            Reclaims++;
            ReclaimedEpochs.Add(old.Epoch);
        }
        retiredVersions.Clear();
        foreach (RcuCallback cb in callbacks)
        {
            if (cb.Ran) continue;
            cb.Ran = true;
            CallbacksRun++;
            cb.Body();
        }
        return true;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static int NextSeed(int seed) => (seed * 1103515245 + 12345) & 0x7FFFFFFF;

    private static void SnapshotStability()
    {
        var domain = new RcuDomain(2, new Dictionary<string, int> { ["a"] = 1 });
        RcuVersion v0 = domain.ReadLock(0);
        Check(v0.Epoch == 0 && v0.Data["a"] == 1, "reader sees the initial version");

        var copy = domain.StartUpdate();
        copy["a"] = 2;
        copy["b"] = 5;
        domain.Publish(copy);

        Check(domain.Current.Epoch == 1 && domain.Current.Data["b"] == 5, "writer published the new version");
        Check(v0.Data["a"] == 1 && !v0.Data.ContainsKey("b"), "old reader still sees the old snapshot");
        Check(v0.Checksum == RcuVersion.ComputeChecksum(v0.Data), "snapshot checksum is stable");

        RcuVersion v1 = domain.ReadLock(1);
        Check(v1.Epoch == 1 && v1.Data["a"] == 2, "reader entering after the publish sees the new version");
        Check(v0.Checksum != v1.Checksum, "the two versions differ");
        domain.ReadUnlock(0);
        domain.ReadUnlock(1);
    }

    private static void GracePeriodBlocksAndReclaims()
    {
        var domain = new RcuDomain(2, new Dictionary<string, int> { ["k"] = 0 });
        domain.ReadLock(0);
        var copy = domain.StartUpdate();
        copy["k"] = 1;
        domain.Publish(copy);

        Check(!domain.TrySynchronize(), "grace period waits for the pre-existing reader");
        Check(!domain.TrySynchronize(), "still waiting on the second poll");
        Check(domain.Reclaims == 0 && domain.PendingRetired == 1, "nothing reclaimed yet");
        Check(!domain.Current.Reclaimed, "the live version is never reclaimed");

        domain.ReadUnlock(0);
        Check(domain.TrySynchronize(), "grace period elapses once the reader exits");
        Check(domain.Reclaims == 1 && domain.PendingRetired == 0, "exactly one version reclaimed");
        Check(domain.ReclaimedEpochs.Count == 1 && domain.ReclaimedEpochs[0] == 0, "the retired epoch 0 was reclaimed");
    }

    private static void NewReadersDoNotDelay()
    {
        var domain = new RcuDomain(3, new Dictionary<string, int> { ["k"] = 0 });
        domain.ReadLock(0);                       // pre-existing reader of epoch 0
        var copy = domain.StartUpdate();
        copy["k"] = 1;
        domain.Publish(copy);                     // epoch 1 published, epoch 0 retired
        domain.ReadLock(2);                       // reader that entered after the publish
        Check(!domain.TrySynchronize(), "blocked by the pre-existing reader only");
        domain.ReadUnlock(0);
        Check(domain.TrySynchronize(), "post-publish reader does not delay the grace period");
        Check(domain.IsReading(2), "the new reader is still inside its critical section");
        domain.ReadUnlock(2);
    }

    private static void DeferredCallbacks()
    {
        var domain = new RcuDomain(1, new Dictionary<string, int> { ["k"] = 0 });
        int ran = 0;
        domain.ReadLock(0);
        var copy = domain.StartUpdate();
        copy["k"] = 1;
        domain.Publish(copy);
        domain.CallRcu(() => ran++);
        Check(ran == 0, "call_rcu callback is deferred");
        Check(!domain.TrySynchronize(), "callback still deferred because a reader is active");
        Check(ran == 0, "no callback before the grace period");
        domain.ReadUnlock(0);
        Check(domain.TrySynchronize(), "grace period passed");
        Check(ran == 1 && domain.CallbacksRun == 1, "callback ran exactly once");
        Check(domain.TrySynchronize(), "second synchronize is a no-op success");
        Check(ran == 1, "callback is not repeated");
    }

    private static void NoTornReads()
    {
        var domain = new RcuDomain(2, new Dictionary<string, int> { ["x"] = 0, ["y"] = 0 });
        RcuVersion snap = domain.ReadLock(0);
        long frozen = snap.Checksum;
        var seen = new List<long>();
        for (int i = 1; i <= 12; i++)
        {
            var copy = domain.StartUpdate();
            copy["x"] = i;
            copy["y"] = -i;
            domain.Publish(copy);
            // The reader re-reads its snapshot between publishes: content and checksum must be identical.
            var observed = new Dictionary<string, int>(snap.Data);
            Check(observed["x"] == 0 && observed["y"] == 0, "reader never observes a partially applied update");
            Check(RcuVersion.ComputeChecksum(observed) == frozen, "recomputed checksum matches the frozen snapshot");
            seen.Add(snap.Checksum);
            // It also protects the snapshot: the grace period may not reclaim epoch 0 yet.
            Check(!domain.TrySynchronize(), "publish " + i + " cannot pass the grace period");
        }
        Check(seen.Count == 12, "twelve publishes observed");
        Check(domain.Reclaims == 0, "no version reclaimed while the reader is active");
        domain.ReadUnlock(0);
        Check(domain.TrySynchronize(), "single synchronize reclaims every pending version");
        Check(domain.Reclaims == 12 && domain.PendingRetired == 0, "all twelve superseded versions reclaimed");
    }

    private static void InterleavedSchedule()
    {
        const int threads = 4;
        var domain = new RcuDomain(threads, new Dictionary<string, int> { ["counter"] = 0 });
        var activeVersions = new Dictionary<int, RcuVersion>();
        int seed = 777;
        int publishes = 0;

        void AssertHeldVersionsAlive()
        {
            foreach (KeyValuePair<int, RcuVersion> kv in activeVersions)
                Check(!kv.Value.Reclaimed, "reader " + kv.Key + " holds a reclaimed version");
        }

        for (int step = 0; step < 300; step++)
        {
            seed = NextSeed(seed);
            int thread = seed % threads;
            int choice = (seed / 13) % 4;
            if (choice == 0 && !domain.IsReading(thread))
            {
                RcuVersion snap = domain.ReadLock(thread);
                activeVersions[thread] = snap;
                Check(snap.Checksum == RcuVersion.ComputeChecksum(snap.Data), "snapshot checksum");
            }
            else if (choice == 1 && domain.IsReading(thread))
            {
                RcuVersion snap = activeVersions[thread];
                int expected = (int)snap.Epoch;   // the writer sets counter = new epoch on every publish
                Check(snap.Data["counter"] == expected, "snapshot content matches its epoch");
                Check(!snap.Reclaimed, "no torn read on a reclaimed version");
                domain.ReadUnlock(thread);
                activeVersions.Remove(thread);
            }
            else if (choice == 2)
            {
                var copy = domain.StartUpdate();
                copy["counter"] = (int)domain.Current.Epoch + 1;
                domain.Publish(copy);
                publishes++;
            }
            else
            {
                domain.TrySynchronize();
            }
            AssertHeldVersionsAlive();
        }

        foreach (int thread in new List<int>(activeVersions.Keys)) domain.ReadUnlock(thread);
        activeVersions.Clear();
        while (!domain.TrySynchronize()) { }
        Check(domain.Publishes == publishes, "publish count matches");
        Check(domain.Reclaims == publishes, "every superseded version reclaimed exactly once: "
            + domain.Reclaims + " of " + publishes);
        Check(domain.Current.Epoch == publishes, "epoch equals the number of publishes");
        Check(domain.Current.Data["counter"] == publishes, "final counter equals the publish count");
        Check(domain.PendingRetired == 0, "no version left retired");
        Check(domain.GracePeriodPolls > publishes, "grace periods were polled repeatedly: " + domain.GracePeriodPolls);
        Console.WriteLine("rcu: publishes=" + publishes + " polls=" + domain.GracePeriodPolls
            + " reclaims=" + domain.Reclaims + " callbacks=" + domain.CallbacksRun);
    }

    public static void Main()
    {
        SnapshotStability();
        GracePeriodBlocksAndReclaims();
        NewReadersDoNotDelay();
        DeferredCallbacks();
        NoTornReads();
        InterleavedSchedule();
        Console.WriteLine("PASS");
    }
}
