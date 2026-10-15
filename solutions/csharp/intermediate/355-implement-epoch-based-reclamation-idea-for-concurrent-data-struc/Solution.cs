// Task 355 - Implement epoch-based reclamation idea for concurrent data structure safe delete. (Tier: Intermediate, CTCI Ch 15)
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    private const int Epochs = 3;
    private const int ScriptSteps = 120;

    /// <summary>
    /// Epoch-based reclamation (EBR) sketch: a global epoch counter, per-thread pinning
    /// (pin() records the current epoch, unpin() clears it) and three retirement buckets
    /// keyed by epoch % Epochs. Documented free rule: an object retired in epoch r may be
    /// freed only when no thread is pinned in an epoch <= r - i.e. r <= minPinnedEpoch - 1,
    /// and with no pinned threads every retired object is safe. Collect() first tries to
    /// advance the epoch (an advance requires every pinned thread to sit in the current
    /// epoch), then drains every bucket entry the rule allows and keeps the rest on the
    /// deferred list. Real EBR frees the bucket from two epochs back; this conservative
    /// rule is strictly safer. A pinned epoch of -1 means "not pinned".
    /// </summary>
    private class EbrManager
    {
        private readonly int[] pinEpochs;
        private readonly List<List<int[]>> buckets = new List<List<int[]>>();   // {objectId, retirementEpoch}
        private readonly List<int[]> freeLog = new List<int[]>();               // {objectId, retirementEpoch, epochAtFree}
        private int globalEpoch;

        public int Violations { get; private set; }
        public int DeferredEntries { get; private set; }
        public int Advances { get; private set; }

        public EbrManager(int numThreads)
        {
            pinEpochs = new int[numThreads];
            for (int i = 0; i < numThreads; i++) pinEpochs[i] = -1;
            for (int i = 0; i < Epochs; i++) buckets.Add(new List<int[]>());
        }

        public void Pin(int thread)
        {
            if (pinEpochs[thread] != -1) throw new Exception("thread already pinned: " + thread);
            pinEpochs[thread] = globalEpoch;
        }

        public void Unpin(int thread)
        {
            if (pinEpochs[thread] == -1) throw new Exception("thread is not pinned: " + thread);
            pinEpochs[thread] = -1;
        }

        public bool IsPinned(int thread) { return pinEpochs[thread] != -1; }

        public void Retire(int thread, int objectId)
        {
            if (pinEpochs[thread] == -1) throw new Exception("retire requires a pinned thread");
            buckets[globalEpoch % Epochs].Add(new[] { objectId, globalEpoch });
        }

        /// <summary>Advance only when every pinned thread sits in the current epoch.</summary>
        public bool TryAdvance()
        {
            foreach (int epoch in pinEpochs)
            {
                if (epoch != -1 && epoch != globalEpoch) return false;
            }
            globalEpoch++;
            Advances++;
            return true;
        }

        public int MinPinnedEpoch()
        {
            int min = int.MaxValue;
            foreach (int epoch in pinEpochs) if (epoch != -1 && epoch < min) min = epoch;
            return min == int.MaxValue ? -1 : min;
        }

        /// <summary>Drain every safe deferred object; returns the objects freed by this call.</summary>
        public List<int> Collect()
        {
            TryAdvance();
            int minPinned = MinPinnedEpoch();
            int safeEpoch = minPinned == -1 ? -1 : minPinned - 1;
            bool everythingSafe = minPinned == -1;
            var freed = new List<int>();
            foreach (var bucket in buckets)
            {
                var kept = new List<int[]>();
                foreach (var entry in bucket)
                {
                    int objectId = entry[0];
                    int retiredEpoch = entry[1];
                    if (everythingSafe || retiredEpoch <= safeEpoch)
                    {
                        RecordFree(objectId, retiredEpoch);
                        freed.Add(objectId);
                    }
                    else
                    {
                        kept.Add(entry);
                    }
                }
                DeferredEntries += kept.Count;
                bucket.Clear();
                bucket.AddRange(kept);
            }
            return freed;
        }

        /// <summary>
        /// The safety invariant, checked at the moment of every free: no thread may be
        /// pinned in an epoch &lt;= the object's retirement epoch.
        /// </summary>
        private void RecordFree(int objectId, int retiredEpoch)
        {
            foreach (int epoch in pinEpochs)
            {
                if (epoch != -1 && epoch <= retiredEpoch) Violations++;
            }
            freeLog.Add(new[] { objectId, retiredEpoch, globalEpoch });
        }

        public bool BucketsEmpty() { return buckets.All(bucket => bucket.Count == 0); }

        public List<int> FreeLogObjectIds() { return freeLog.Select(entry => entry[0]).ToList(); }
    }

    public static void Main()
    {
        // Phase 1: retirement in epoch 0; the pin that predates an advance blocks the free.
        var mgr = new EbrManager(3);
        mgr.Pin(0);
        mgr.Retire(0, 1);                                    // object 1 retired in epoch 0
        Check(mgr.Collect().Count == 0, "object 1 must not be freed while thread 0 pins epoch 0");

        mgr.Pin(1);                                          // thread 1 pins the current epoch 1
        Check(mgr.Collect().Count == 0, "old pin (epoch 0) keeps object 1 deferred");
        Check(mgr.Violations == 0, "no free while a thread pinned an old epoch");

        mgr.Unpin(0);
        Check(ListsEqual(mgr.Collect(), new List<int> { 1 }), "object 1 frees once no thread pins an epoch <= 0");
        Check(mgr.Violations == 0, "freeing object 1 respected the pin rule");

        // Phase 2: a thread pinned in the retirement epoch itself blocks the free.
        mgr.Pin(0);                                          // thread 0 pins epoch 2
        mgr.Retire(0, 2);                                    // object 2 retired in epoch 2
        Check(mgr.Collect().Count == 0, "object 2 deferred while thread 1 still pins epoch 1");
        mgr.Unpin(1);
        Check(mgr.Collect().Count == 0, "object 2 still deferred: thread 0 pins its retirement epoch 2");
        mgr.Unpin(0);
        Check(ListsEqual(mgr.Collect(), new List<int> { 2 }), "object 2 frees at quiescence");
        Check(mgr.Violations == 0, "freeing object 2 respected the pin rule");

        // Phase 3: LCG-driven interleaving of pin / unpin / retire / collect across 3 threads.
        var scripted = new EbrManager(3);
        long seed = 4242;
        int nextObject = 100;
        var retired = new List<int>();
        for (int step = 0; step < ScriptSteps; step++)
        {
            seed = (seed * 75 + 74) % 65537;
            int thread = (int)(seed % 3);
            seed = (seed * 75 + 74) % 65537;
            int action = (int)(seed % 4);
            if (action == 0 && !scripted.IsPinned(thread))
            {
                scripted.Pin(thread);
            }
            else if (action == 1 && scripted.IsPinned(thread))
            {
                scripted.Unpin(thread);
            }
            else if (action >= 2 && scripted.IsPinned(thread))
            {
                scripted.Retire(thread, nextObject);
                retired.Add(nextObject);
                nextObject++;
            }
            scripted.Collect();
            Check(scripted.Violations == 0, "free-of-pinned violation at step " + step);
        }

        Check(retired.Count >= 10, "the interleaving retired a useful number of objects");
        Check(scripted.DeferredEntries > 0, "old pins actually deferred some objects");

        // Quiescence: unpin everything, then the deferred list must drain completely.
        for (int thread = 0; thread < 3; thread++)
        {
            if (scripted.IsPinned(thread)) scripted.Unpin(thread);
        }
        scripted.Collect();
        Check(scripted.BucketsEmpty(), "deferred lists drained");

        var freedIds = scripted.FreeLogObjectIds();
        freedIds.Sort();
        var sortedRetired = new List<int>(retired);
        sortedRetired.Sort();
        Check(ListsEqual(freedIds, sortedRetired), "every retired object freed exactly once");
        Check(freedIds.Distinct().Count() == freedIds.Count, "no double frees");
        Check(scripted.Violations == 0, "the whole run respected the pin rule");
        Check(scripted.Advances >= 1, "epochs actually advanced");

        Console.WriteLine("PASS");
    }

    private static bool ListsEqual(List<int> a, List<int> b)
    {
        if (a.Count != b.Count) return false;
        for (int i = 0; i < a.Count; i++) if (a[i] != b[i]) return false;
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
