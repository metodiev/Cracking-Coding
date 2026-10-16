// Task 626 - Implement work-stealing deque (Chase-Lev) for task scheduler. (Tier: Expert, CTCI Ch 15 - Threads and Locks)
//
// A Chase-Lev deque gives its owner a push/pop stack at one end (the bottom) and lets thieves
// steal from the other end (the top). The owner owns the bottom index outright, so push and
// pop need no CAS in the common case; only the *last* element is contested, which is the single
// CAS in Pop. Thieves read top and bottom, read the slot, and CAS top forward - if the owner or
// another thief moved top meanwhile, the steal aborts and the thief retries.
//
//   Push        : if the buffer is full, grow it (the old array is copied [top, bottom) into a
//                 twice-as-large ring, so the captured array reference stays valid for thieves);
//                 then write at bottom and publish bottom + 1.
//   Pop (owner) : decrement bottom (visible to thieves immediately), read top; if the slot is
//                 the last one, the pop is decided by CAS(top, top, top+1) - losing that CAS
//                 means a thief took the element, and the pop returns empty after restoring
//                 bottom. Otherwise the owner takes bottom's slot.
//   Steal       : snapshot (top, bottom, buffer slot), then CAS top forward.
//
// Concurrency is *simulated deterministically* - no threads, no sleeps, no timers. The owner's
// pop and the thief's steal are split into Begin/Finish halves, so a test can interleave a thief
// exactly inside the owner's read-then-CAS window. The shrink half of the real algorithm is
// omitted (the buffer only grows), and there is a single owner thread.
//
// Verified invariants: an element is handed out exactly once, owner pops return the newest
// element, stealths return the oldest, a failed owner pop happens only when the model is already
// empty (the intervening steal consumed the last element), and the aborted-steal path is
// exercised rather than assumed.
using System;
using System.Collections.Generic;
using System.Threading;

#nullable enable

internal enum StealStatus { Stolen, Aborted, Empty }

internal sealed class StealAttempt
{
    public long Top;
    public long Bottom;
    public int Task;
    public int[] Buffer = Array.Empty<int>();
}

internal sealed class PopAttempt
{
    public long Bottom;
    public long Top;
    public int Task;
    public bool IsLast;
}

internal sealed class ChaseLevDeque
{
    private int[] tasks;
    private long top;
    private long bottom;

    public int Grows;
    public int OwnerPops;
    public int Steals;
    public int Aborts;
    public int RejectedPops;
    public int MaxOccupancy;

    public ChaseLevDeque(int capacity)
    {
        tasks = new int[capacity];
        for (int i = 0; i < capacity; i++) tasks[i] = -1;
    }

    public int Capacity => tasks.Length;
    public long Top => Volatile.Read(ref top);
    public long Bottom => Volatile.Read(ref bottom);
    public int Occupancy => (int)(Bottom - Top);

    public void Push(int task)
    {
        long b = Volatile.Read(ref bottom);
        long t = Volatile.Read(ref top);
        if (b - t > tasks.Length - 1) Grow(b, t);
        tasks[(int)(b % tasks.Length)] = task;
        Volatile.Write(ref bottom, b + 1);
        int occupancy = (int)(b + 1 - t);
        if (occupancy > MaxOccupancy) MaxOccupancy = occupancy;
    }

    private void Grow(long b, long t)
    {
        var old = tasks;
        var bigger = new int[old.Length * 2];
        for (int i = 0; i < bigger.Length; i++) bigger[i] = -1;
        for (long i = t; i < b; i++) bigger[(int)(i % bigger.Length)] = old[(int)(i % old.Length)];
        tasks = bigger;
        Grows++;
    }

    public PopAttempt BeginPop()
    {
        long b = Volatile.Read(ref bottom) - 1;
        Interlocked.Exchange(ref bottom, b);            // the decrement is what thieves observe
        long t = Volatile.Read(ref top);
        var attempt = new PopAttempt { Bottom = b, Top = t, IsLast = t == b };
        attempt.Task = t <= b ? tasks[(int)(b % tasks.Length)] : -1;
        return attempt;
    }

    public bool FinishPop(PopAttempt attempt, out int task)
    {
        task = -1;
        if (attempt.Top > attempt.Bottom)
        {
            Interlocked.Exchange(ref bottom, attempt.Bottom + 1);   // the deque was empty
            return false;
        }
        if (attempt.IsLast)
        {
            // Only the last element is contested: the owner and a thief race for the same slot.
            if (Interlocked.CompareExchange(ref top, attempt.Top + 1, attempt.Top) != attempt.Top)
            {
                Interlocked.Exchange(ref bottom, attempt.Bottom + 1);
                RejectedPops++;
                return false;
            }
            // Winner: the deque is now empty, so bottom moves to meet top.
            task = attempt.Task;
            Interlocked.Exchange(ref bottom, attempt.Bottom + 1);
            OwnerPops++;
            return true;
        }
        // Common case: bottom keeps the decremented value, which is what reserves this slot.
        task = attempt.Task;
        OwnerPops++;
        return true;
    }

    public bool TryPop(out int task) => FinishPop(BeginPop(), out task);

    public StealAttempt BeginSteal()
    {
        var attempt = new StealAttempt();
        attempt.Top = Volatile.Read(ref top);
        attempt.Bottom = Volatile.Read(ref bottom);
        attempt.Buffer = tasks;                                  // captured once, like the real thief
        attempt.Task = attempt.Top < attempt.Bottom
            ? attempt.Buffer[(int)(attempt.Top % attempt.Buffer.Length)]
            : -1;
        return attempt;
    }

    public StealStatus FinishSteal(StealAttempt attempt, out int task)
    {
        task = -1;
        if (attempt.Top >= attempt.Bottom) return StealStatus.Empty;
        if (Interlocked.CompareExchange(ref top, attempt.Top + 1, attempt.Top) != attempt.Top)
        {
            Aborts++;
            return StealStatus.Aborted;
        }
        task = attempt.Task;
        Steals++;
        return StealStatus.Stolen;
    }

    public StealStatus TrySteal(out int task) => FinishSteal(BeginSteal(), out task);
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    /// <summary>Deterministic 64-bit mix (splitmix64 finalizer) used to drive the schedule.</summary>
    private static int NextSeed(int seed)
    {
        ulong z = (ulong)(uint)seed + 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        z ^= z >> 31;
        return (int)(z & 0x7FFFFFFF);
    }

    private static void OwnerIsAStack()
    {
        var deque = new ChaseLevDeque(4);
        Check(!deque.TryPop(out _), "pop on a fresh deque reports empty");
        for (int i = 1; i <= 5; i++) deque.Push(i);
        var got = new List<int>();
        while (deque.TryPop(out int v)) got.Add(v);
        Check(string.Join(",", got) == "5,4,3,2,1", "owner pops the newest first, got " + string.Join(",", got));
        Check(!deque.TryPop(out _), "empty again");
        Check(deque.Top == deque.Bottom, "top and bottom meet when empty");
        Check(deque.OwnerPops == 5, "five owner pops counted");
    }

    private static void ThievesAreFifo()
    {
        var deque = new ChaseLevDeque(4);
        for (int i = 1; i <= 5; i++) deque.Push(i);
        var got = new List<int>();
        while (deque.TrySteal(out int v) == StealStatus.Stolen) got.Add(v);
        Check(string.Join(",", got) == "1,2,3,4,5", "thieves steal the oldest first, got " + string.Join(",", got));
        Check(deque.TrySteal(out _) == StealStatus.Empty, "steal on an empty deque reports empty");
        Check(deque.Steals == 5, "five steals counted");

        // Mixed: steal from the top, pop from the bottom, and the two ends meet in the middle.
        for (int i = 1; i <= 4; i++) deque.Push(i * 7);
        Check(deque.TrySteal(out int first) == StealStatus.Stolen && first == 7, "steal the oldest");
        Check(deque.TryPop(out int last) && last == 28, "pop the newest");
        Check(deque.TryPop(out int mid) && mid == 21, "pop the next newest");
        Check(deque.TrySteal(out int other) == StealStatus.Stolen && other == 14, "the last element");
        Check(deque.TrySteal(out _) == StealStatus.Empty, "nothing left for thieves");
        Check(!deque.TryPop(out _), "nothing left for the owner");
    }

    private static void BufferGrowthPreservesBothEnds()
    {
        var deque = new ChaseLevDeque(4);
        for (int i = 1; i <= 20; i++) deque.Push(i);
        Check(deque.Grows >= 2, "the buffer grew, grows=" + deque.Grows);
        Check(deque.Capacity >= 32, "capacity doubled past twenty, capacity=" + deque.Capacity);
        Check(deque.MaxOccupancy == 20, "occupancy reached twenty, got " + deque.MaxOccupancy);
        var stolen = new List<int>();
        while (deque.TrySteal(out int v) == StealStatus.Stolen) stolen.Add(v);
        Check(string.Join(",", stolen) == string.Join(",", Sequence(1, 20)), "steals stay FIFO across growth");

        for (int i = 1; i <= 20; i++) deque.Push(i);
        var popped = new List<int>();
        while (deque.TryPop(out int v)) popped.Add(v);
        Check(string.Join(",", popped) == string.Join(",", Sequence(20, 1)), "pops stay LIFO across growth");

        // Ring wraparound: fill to the boundary, drain a little, push again past the boundary.
        var ring = new ChaseLevDeque(4);
        ring.Push(1);
        ring.Push(2);
        ring.Push(3);
        Check(ring.TryPop(out int r) && r == 3, "ring pop before the wrap");
        for (int i = 4; i <= 8; i++) ring.Push(i);
        Check(ring.Grows >= 1, "wraparound forced a grow, grows=" + ring.Grows);
        var wrapped = new List<int>();
        while (ring.TrySteal(out int v) == StealStatus.Stolen) wrapped.Add(v);
        Check(string.Join(",", wrapped) == "1,2,4,5,6,7,8", "wraparound preserves order, got " + string.Join(",", wrapped));
    }

    private static List<int> Sequence(int from, int to)
    {
        var list = new List<int>();
        int step = from <= to ? 1 : -1;
        for (int i = from; step > 0 ? i <= to : i >= to; i += step) list.Add(i);
        return list;
    }

    private static void StealAbortsWhenTopMoves()
    {
        var deque = new ChaseLevDeque(4);
        deque.Push(1);
        deque.Push(2);
        StealAttempt first = deque.BeginSteal();
        StealAttempt second = deque.BeginSteal();
        Check(first.Top == second.Top && first.Task == 1, "both thieves snapshot the same oldest task");
        Check(deque.FinishSteal(second, out int won) == StealStatus.Stolen && won == 1, "the second thief wins");
        Check(deque.FinishSteal(first, out _) == StealStatus.Aborted, "the stale snapshot aborts");
        Check(deque.Aborts == 1, "one abort recorded");
        Check(deque.TrySteal(out int next) == StealStatus.Stolen && next == 2, "the next oldest task is next");
    }

    private static void LastElementRace()
    {
        // A thief snapshots the single element, then the owner begins a pop, then the thief's CAS
        // lands first: the thief wins and the owner's pop must report empty, not the same task.
        var thiefWins = new ChaseLevDeque(4);
        thiefWins.Push(99);
        StealAttempt attempt = thiefWins.BeginSteal();
        Check(attempt.Top == 0 && attempt.Bottom == 1, "thief snapshot sees one element");
        PopAttempt pop = thiefWins.BeginPop();
        Check(pop.IsLast && pop.Task == 99, "owner plans to take the same last element");
        Check(thiefWins.FinishSteal(attempt, out int stolen) == StealStatus.Stolen && stolen == 99, "thief wins");
        Check(!thiefWins.FinishPop(pop, out _), "the owner's pop loses the race and reports empty");
        Check(thiefWins.RejectedPops == 1, "the lost race is counted");
        Check(thiefWins.Top == thiefWins.Bottom, "the deque is empty after the race");
        Check(dequeEmpty(thiefWins), "no element is left behind");

        // Mirror image: the owner's CAS lands first, so the thief's stale attempt must abort.
        var ownerWins = new ChaseLevDeque(4);
        ownerWins.Push(99);
        StealAttempt loser = ownerWins.BeginSteal();
        PopAttempt winningPop = ownerWins.BeginPop();
        Check(ownerWins.FinishPop(winningPop, out int taken) && taken == 99, "owner wins the race");
        Check(ownerWins.FinishSteal(loser, out _) == StealStatus.Aborted, "the thief's attempt aborts");
        Check(dequeEmpty(ownerWins), "nothing left after the owner wins");
        Check(ownerWins.Steals == 0 && ownerWins.OwnerPops == 1, "exactly one winner");
    }

    private static bool dequeEmpty(ChaseLevDeque deque)
    {
        Check(!deque.TryPop(out _), "no owner element left");
        Check(deque.TrySteal(out _) == StealStatus.Empty, "no thief element left");
        return true;
    }

    private static void InterleavedScheduler()
    {
        var deque = new ChaseLevDeque(4);
        var model = new List<int>();              // model[0] is the oldest element
        var inFlight = new List<StealAttempt>();
        PopAttempt? pop = null;
        int produced = 0, consumed = 0, aborts = 0, emptyPops = 0, emptySteals = 0;
        int seed = 20260116;

        void FinishStealOne(StealAttempt attempt)
        {
            StealStatus status = deque.FinishSteal(attempt, out int stolen);
            if (status == StealStatus.Stolen)
            {
                Check(model.Count > 0, "a successful steal implies a non-empty model");
                Check(stolen == model[0], "a steal takes the oldest element: " + stolen + " vs " + model[0]);
                model.RemoveAt(0);
                consumed++;
            }
            else if (status == StealStatus.Aborted) aborts++;
            else emptySteals++;
        }

        for (int step = 0; step < 3000; step++)
        {
            seed = NextSeed(seed);
            int choice = seed % 12;
            bool acted = false;
            StealAttempt? newest = inFlight.Count > 0 ? inFlight[inFlight.Count - 1] : null;

            if (pop == null && model.Count == 1 && newest != null && newest.Top < newest.Bottom)
            {
                // Scripted race: exactly one task is left and a thief already snapshotted it, so
                // the owner's pop and that thief contend for it - whichever lands first wins.
                pop = deque.BeginPop();
                acted = true;
            }
            else if (choice < 4 && pop == null)
            {
                int value = 100 + produced++;
                deque.Push(value);
                model.Add(value);
                acted = true;
            }
            else if (choice < 6 && pop == null)
            {
                pop = deque.BeginPop();           // split: a thief may interleave inside here
                acted = true;
            }
            else if (choice < 10 && inFlight.Count < 2)
            {
                inFlight.Add(deque.BeginSteal());
                acted = true;
            }

            if (!acted)
            {
                if (inFlight.Count > 0)
                {
                    StealAttempt attempt = inFlight[inFlight.Count - 1];
                    inFlight.RemoveAt(inFlight.Count - 1);
                    FinishStealOne(attempt);
                }
                else if (pop != null)
                {
                    bool ok = deque.FinishPop(pop, out int value);
                    pop = null;
                    if (ok)
                    {
                        Check(model.Count > 0, "a successful pop implies a non-empty model");
                        Check(value == model[model.Count - 1],
                            "an owner pop takes the newest element: " + value + " vs " + model[model.Count - 1]);
                        model.RemoveAt(model.Count - 1);
                        consumed++;
                    }
                    else
                    {
                        // Either the deque was empty or a thief already consumed the last element.
                        Check(model.Count == 0, "a failed pop must find the model empty, size=" + model.Count);
                        emptyPops++;
                    }
                }
                else
                {
                    int value = 100 + produced++;
                    deque.Push(value);
                    model.Add(value);
                }
            }

            // While the owner's pop is in flight its decrement is already visible, so the deque
            // reports one element fewer than the model.
            int expected = model.Count - (pop != null ? 1 : 0);
            Check((int)(deque.Bottom - deque.Top) == expected,
                "deque size matches the model at step " + step + ": " + (deque.Bottom - deque.Top) + " vs " + expected);
        }

        // Resolve anything still outstanding, then drain one end at a time against the model.
        if (pop != null)
        {
            bool ok = deque.FinishPop(pop, out int value);
            pop = null;
            if (ok)
            {
                Check(model.Count > 0 && value == model[model.Count - 1], "pending pop resolves against the model");
                model.RemoveAt(model.Count - 1);
                consumed++;
            }
            else
            {
                Check(model.Count == 0, "pending pop fails only on an empty model");
                emptyPops++;
            }
        }
        foreach (StealAttempt attempt in inFlight) FinishStealOne(attempt);
        inFlight.Clear();
        Check((int)(deque.Bottom - deque.Top) == model.Count, "deque and model agree before the drain");

        while (model.Count > 0)
        {
            if (model.Count % 2 == 0)
            {
                StealStatus status = deque.TrySteal(out int low);
                Check(status == StealStatus.Stolen && low == model[0], "drain-from-top matches the model");
                model.RemoveAt(0);
            }
            else
            {
                Check(deque.TryPop(out int high) && high == model[model.Count - 1], "drain-from-bottom matches the model");
                model.RemoveAt(model.Count - 1);
            }
            consumed++;
            Check((int)(deque.Bottom - deque.Top) == model.Count, "size still matches during the drain");
        }
        Check(deque.TrySteal(out _) == StealStatus.Empty, "deque empty for thieves at the end");
        Check(!deque.TryPop(out _), "deque empty for the owner at the end");
        Console.WriteLine("chase-lev: produced=" + produced + " consumed=" + consumed + " aborts=" + aborts
            + " steals=" + deque.Steals + " empty-steals=" + emptySteals + " lost-races=" + deque.RejectedPops
            + " empty-pops=" + emptyPops + " grows=" + deque.Grows + " capacity=" + deque.Capacity
            + " max-occupancy=" + deque.MaxOccupancy);
        Check(consumed == produced, "every produced task was handed out exactly once: " + consumed + " of " + produced);
        Check(aborts > 0, "the schedule exercised the aborted-steal path: " + aborts);
        Check(deque.RejectedPops > 0, "the schedule exercised the lost last-element race: " + deque.RejectedPops);
        Check(deque.Grows > 0, "the schedule forced the buffer to grow: " + deque.Grows);
    }

    public static void Main()
    {
        OwnerIsAStack();
        ThievesAreFifo();
        BufferGrowthPreservesBothEnds();
        StealAbortsWhenTopMoves();
        LastElementRace();
        InterleavedScheduler();
        Console.WriteLine("PASS");
    }
}
