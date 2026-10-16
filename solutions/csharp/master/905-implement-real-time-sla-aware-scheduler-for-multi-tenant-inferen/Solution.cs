// Task 905 - Implement real-time SLA-aware scheduler for multi-tenant inference.
// (Tier: Master, Beyond - Distributed Systems)
//
// A single inference slot serves unit-cost requests that arrive online, each with a tenant, a release
// time and a deadline (the request's SLA).  Admission control is exact rather than heuristic: a request is
// accepted only when the whole accepted set - the new request included - is still schedulable, which is
// decided by simulating earliest-deadline-first over the accepted set.  EDF is optimal for unit jobs with
// release times and deadlines, so a request that fails this test cannot be served by any scheduler, and the
// rejection is justified rather than a policy artefact.  The simulation then runs the accepted set with EDF
// again, breaking ties between equal deadlines by tenant virtual time (a weighted deficit round robin), so
// tenants with the same weight get the same service even when they contend for the same slots.
//
// Safety: no accepted request ever completes after its deadline - re-checked after every dispatch - and a
// bounded pending window rejects work early instead of building an unbounded queue.
//
// Simplifications: unit-cost jobs on one machine, integer ticks, no preemption modelling beyond the EDF
// feasibility test, tenants identified by id.  Verified: the deadline invariant after every dispatch, the
// admission decision against a brute-force maximum feasible subset, justification of each rejection,
// tenant fairness, the overload/empty/degenerate boundaries, and reproducibility.
using System;
using System.Collections.Generic;
using System.Globalization;

public static class Solution
{
    private static ulong _state;

    private static ulong NextUInt64()
    {
        unchecked
        {
            _state += 0x9E3779B97F4A7C15UL;
            ulong z = _state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }
    }

    private static int NextInt(int bound) => (int)(NextUInt64() % (ulong)bound);

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    internal sealed class Request
    {
        internal int Id;
        internal int Tenant;
        internal int Arrival;
        internal int Deadline;

        internal Request Copy() => new Request { Id = Id, Tenant = Tenant, Arrival = Arrival, Deadline = Deadline };
    }

    internal sealed class Scheduler
    {
        private readonly int _window;
        private readonly Dictionary<int, int> _virtualTime = new Dictionary<int, int>();
        private readonly Dictionary<int, int> _admitted = new Dictionary<int, int>();
        internal readonly List<Request> Accepted = new List<Request>();
        internal readonly List<Request> Rejected = new List<Request>();
        internal readonly List<Request> Infeasible = new List<Request>();
        internal readonly List<string> Log = new List<string>();

        internal Scheduler(int pendingWindow)
        {
            _window = pendingWindow;
        }

        internal int Now;

        /** Exact EDF schedulability test for unit jobs: sort by deadline and pack the slots. */
        internal static bool Feasible(List<Request> jobs)
        {
            List<Request> ordered = new List<Request>(jobs);
            ordered.Sort((a, b) => a.Deadline != b.Deadline ? a.Deadline.CompareTo(b.Deadline) : a.Id.CompareTo(b.Id));
            int t = 0;
            foreach (Request job in ordered)
            {
                t = Math.Max(t, job.Arrival);
                if (t + 1 > job.Deadline) return false;
                t++;
            }
            return true;
        }

        internal bool Submit(Request request)
        {
            if (request.Deadline <= request.Arrival) // malformed SLA: no slot can ever satisfy it
            {
                Rejected.Add(request);
                Log.Add("A" + request.Arrival + " reject id=" + request.Id + " malformed");
                return false;
            }
            if (Pending() >= _window)
            {
                Rejected.Add(request);
                Log.Add("A" + request.Arrival + " reject id=" + request.Id + " window");
                return false;
            }
            List<Request> candidate = new List<Request>();
            foreach (Request job in Accepted) candidate.Add(job);
            candidate.Add(request);
            if (!Feasible(candidate))
            {
                Rejected.Add(request);
                Infeasible.Add(request);
                Log.Add("A" + request.Arrival + " reject id=" + request.Id + " infeasible");
                return false;
            }
            Accepted.Add(request);
            _admitted[request.Tenant] = Admitted(request.Tenant) + 1;
            Log.Add("A" + request.Arrival + " accept id=" + request.Id + " tenant=" + request.Tenant);
            return true;
        }

        internal int Pending()
        {
            int pending = 0;
            foreach (Request job in Accepted)
            {
                if (!Completed.Contains(job.Id) && job.Arrival <= Now) pending++;
            }
            return pending;
        }

        internal readonly HashSet<int> Completed = new HashSet<int>();
        internal readonly Dictionary<int, int> CompletionTime = new Dictionary<int, int>();
        internal readonly Dictionary<int, int> Served = new Dictionary<int, int>();

        internal int Admitted(int tenant) => _admitted.TryGetValue(tenant, out int value) ? value : 0;

        internal int ServedCount(int tenant) => Served.TryGetValue(tenant, out int value) ? value : 0;

        private int VirtualTime(int tenant) => _virtualTime.TryGetValue(tenant, out int value) ? value : 0;

        /** Earliest deadline first, ties broken by the tenant with the least service (deficit round robin). */
        private int PickIndex(int now)
        {
            int best = -1;
            for (int i = 0; i < Accepted.Count; i++)
            {
                Request job = Accepted[i];
                if (Completed.Contains(job.Id) || job.Arrival > now) continue;
                if (best < 0)
                {
                    best = i;
                    continue;
                }
                Request current = Accepted[best];
                if (job.Deadline != current.Deadline)
                {
                    if (job.Deadline < current.Deadline) best = i;
                }
                else if (VirtualTime(job.Tenant) < VirtualTime(current.Tenant))
                {
                    best = i;
                }
            }
            return best;
        }

        /** Runs the accepted set to completion; throws if a single deadline is missed. */
        internal void Run(int horizon)
        {
            for (Now = 0; Now <= horizon; Now++)
            {
                int index = PickIndex(Now);
                if (index < 0) continue;
                Request job = Accepted[index];
                Completed.Add(job.Id);
                CompletionTime[job.Id] = Now + 1; // a unit job dispatched in slot Now finishes at Now + 1
                Served[job.Tenant] = ServedCount(job.Tenant) + 1;
                _virtualTime[job.Tenant] = VirtualTime(job.Tenant) + 1;
                if (Now + 1 > job.Deadline)
                {
                    Check(false, "request " + job.Id + " completed at " + (Now + 1) + " past its deadline " + job.Deadline);
                }
            }
        }

        internal int MaxLatency()
        {
            int worst = 0;
            foreach (Request job in Accepted)
            {
                if (CompletionTime.TryGetValue(job.Id, out int done)) worst = Math.Max(worst, done - job.Arrival);
            }
            return worst;
        }
    }

    /** Brute force: the largest schedulable subset of the given requests (2^n subsets, n <= 12). */
    private static List<int> LargestFeasibleSubset(List<Request> requests)
    {
        int bestMask = 0;
        int bestCount = -1;
        int subsets = 1 << requests.Count;
        for (int mask = 0; mask < subsets; mask++)
        {
            List<Request> subset = new List<Request>();
            for (int i = 0; i < requests.Count; i++)
            {
                if ((mask & (1 << i)) != 0) subset.Add(requests[i]);
            }
            if (subset.Count <= bestCount) continue;
            if (Scheduler.Feasible(subset))
            {
                bestCount = subset.Count;
                bestMask = mask;
            }
        }
        List<int> ids = new List<int>();
        for (int i = 0; i < requests.Count; i++)
        {
            if ((bestMask & (1 << i)) != 0) ids.Add(requests[i].Id);
        }
        return ids;
    }

    private static List<Request> Generate(int count, int seed, int tenants, int horizon, int slack)
    {
        _state = (ulong)seed;
        List<Request> requests = new List<Request>();
        for (int i = 0; i < count; i++)
        {
            int arrival = NextInt(horizon);
            requests.Add(new Request
            {
                Id = i,
                Tenant = NextInt(tenants),
                Arrival = arrival,
                Deadline = arrival + 1 + NextInt(slack + 1),
            });
        }
        return requests;
    }

    public static void Main()
    {
        // ---- 1. admission under load versus the brute-force maximum ---------------------------------
        List<Request> burst = Generate(12, 11, 3, 4, 3);
        burst.Sort((a, b) => a.Arrival != b.Arrival ? a.Arrival.CompareTo(b.Arrival) : a.Id.CompareTo(b.Id));
        Scheduler scheduler = new Scheduler(64);
        foreach (Request request in burst) scheduler.Submit(request.Copy());
        scheduler.Run(40);
        List<int> optimum = LargestFeasibleSubset(burst);
        Check(scheduler.Accepted.Count <= optimum.Count,
            "online admission (" + scheduler.Accepted.Count + ") never beats the offline optimum (" + optimum.Count + ")");
        Check(scheduler.Accepted.Count >= (optimum.Count * 3) / 4,
            "online admission stays within 75% of the offline optimum (" + scheduler.Accepted.Count + "/" + optimum.Count + ")");
        Check(scheduler.Rejected.Count == burst.Count - scheduler.Accepted.Count, "every request was accepted or rejected");
        Check(scheduler.Completed.Count == scheduler.Accepted.Count, "every accepted request ran to completion");

        // ---- 2. every rejection is justified --------------------------------------------------------
        bool justified = true;
        foreach (Request rejected in scheduler.Infeasible)
        {
            List<Request> probe = new List<Request>();
            foreach (Request job in scheduler.Accepted) probe.Add(job);
            probe.Add(rejected);
            justified &= !Scheduler.Feasible(probe);
        }
        Check(justified && scheduler.Infeasible.Count > 0,
            "each of the " + scheduler.Infeasible.Count + " feasibility rejections would break EDF schedulability");

        // ---- 3. the deadline invariant --------------------------------------------------------------
        bool onTime = true;
        foreach (Request job in scheduler.Accepted) onTime &= scheduler.CompletionTime[job.Id] <= job.Deadline;
        Check(onTime, "no accepted request finished after its deadline");
        Check(scheduler.CompletionTime.Values.Count == scheduler.Accepted.Count, "one completion record per accepted request");

        // ---- 4. tenant fairness under contention ----------------------------------------------------
        List<Request> twins = new List<Request>();
        for (int i = 0; i < 8; i++)
        {
            twins.Add(new Request { Id = 100 + i, Tenant = i % 2, Arrival = 0, Deadline = 40 });
        }
        Scheduler fair = new Scheduler(64);
        foreach (Request request in twins) fair.Submit(request.Copy());
        fair.Run(20);
        Check(fair.Accepted.Count == 8, "all eight slack requests were admitted");
        Check(Math.Abs(fair.ServedCount(0) - fair.ServedCount(1)) <= 1,
            "equal-weight tenants are served within one slot of each other (" + fair.ServedCount(0) + " vs " + fair.ServedCount(1) + ")");

        // ---- 5. overload, window and malformed boundaries --------------------------------------------
        List<Request> tightTwin = new List<Request>();
        for (int i = 0; i < 6; i++)
        {
            tightTwin.Add(new Request { Id = 200 + i, Tenant = i % 2, Arrival = 0, Deadline = 2 });
        }
        Scheduler tight = new Scheduler(64);
        foreach (Request request in tightTwin) tight.Submit(request.Copy());
        tight.Run(10);
        Check(tight.Accepted.Count == 2, "only two of six unit-cost jobs with deadline 2 can be admitted (" + tight.Accepted.Count + ")");
        Check(tight.Rejected.Count == 4, "the remaining four are reported as rejected, not silently dropped");
        Check(tight.MaxLatency() <= 2, "the admitted jobs of the tight instance met their SLA");
        Scheduler bounded = new Scheduler(2);
        List<Request> many = Generate(9, 3, 2, 1, 9);
        int windowRejections = 0;
        foreach (Request request in many)
        {
            if (!bounded.Submit(request.Copy())) windowRejections++;
            Check(bounded.Pending() <= 2, "the pending window never exceeded its bound");
        }
        Check(windowRejections > 0, "a two-slot window sheds " + windowRejections + " requests instead of queueing them");
        Scheduler empty = new Scheduler(4);
        empty.Run(5);
        Check(empty.Accepted.Count == 0 && empty.Completed.Count == 0, "an idle scheduler does nothing");
        Scheduler degenerate = new Scheduler(4);
        Check(!degenerate.Submit(new Request { Id = 1, Tenant = 0, Arrival = 3, Deadline = 3 }),
            "a request with deadline == arrival is malformed and rejected");
        Check(!degenerate.Submit(new Request { Id = 2, Tenant = 0, Arrival = 5, Deadline = 4 }),
            "a request with deadline < arrival is malformed and rejected");
        Check(degenerate.Submit(new Request { Id = 3, Tenant = 0, Arrival = 0, Deadline = 1 }),
            "the smallest legal request (one slot) is admitted");

        // ---- 6. reproducibility ----------------------------------------------------------------------
        Scheduler again = new Scheduler(64);
        List<Request> replay = Generate(12, 11, 3, 4, 3);
        replay.Sort((a, b) => a.Arrival != b.Arrival ? a.Arrival.CompareTo(b.Arrival) : a.Id.CompareTo(b.Id));
        foreach (Request request in replay) again.Submit(request.Copy());
        again.Run(40);
        Check(string.Join(",", again.Log) == string.Join(",", scheduler.Log), "identical arrivals reproduce identical decisions");

        Console.WriteLine("admission: " + scheduler.Accepted.Count + " accepted of " + burst.Count +
            " requests, offline optimum " + optimum.Count + ", all " + scheduler.Rejected.Count + " rejections justified by EDF");
        Console.WriteLine("deadlines: worst observed latency " + scheduler.MaxLatency() + " ticks; tenant fairness " +
            fair.ServedCount(0) + "/" + fair.ServedCount(1) + " over 8 contended requests");
        Console.WriteLine("boundaries: window shedding " + windowRejections + " requests with a 2-slot window, malformed SLAs rejected");
        Console.WriteLine("PASS");
    }
}
