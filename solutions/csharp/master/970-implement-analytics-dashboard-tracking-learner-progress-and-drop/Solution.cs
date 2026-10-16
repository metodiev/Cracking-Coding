// Task 970 - Implement analytics dashboard tracking learner progress and drop-off.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// Raw learner events are folded into a funnel over the ordered stages view -> start -> attempt ->
// solve. A learner only counts for stage i when every earlier stage was also reached, so a skipped
// stage truncates that learner's progress exactly as the real dashboard would. Stage-to-stage
// drop-off, a day-by-day retention curve for one cohort, and the stage with the worst drop-off are
// derived from the same fold.
//
// Verified: exact funnel counts on a crafted sample, the non-increasing funnel invariant, drop-off
// consistency (the survivors after all stages equal the product of the per-stage survival rates),
// the skipped-stage and never-viewed negative cases, a seeded random cross-check against an
// independent per-stage prefix-predicate oracle, retention endpoints and the empty-stream boundary.
// Simplification: events are in-memory records with a day counter; "time" is an integer day index
// and a learner is simply identified by a string id.
using System;
using System.Collections.Generic;

public static class Solution
{
    private static readonly string[] Stages = { "view", "start", "attempt", "solve" };

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

    private static double NextUnit() => (NextUInt64() >> 11) * (1.0 / 9007199254740992.0);

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    private sealed class Event
    {
        public readonly string Learner;
        public readonly string Stage;
        public readonly int Day;

        public Event(string learner, string stage, int day)
        {
            Learner = learner;
            Stage = stage;
            Day = day;
        }
    }

    private static Dictionary<string, HashSet<string>> StagesByLearner(List<Event> events)
    {
        var reached = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
        foreach (Event record in events)
        {
            if (!reached.TryGetValue(record.Learner, out HashSet<string> set))
            {
                set = new HashSet<string>(StringComparer.Ordinal);
                reached[record.Learner] = set;
            }
            set.Add(record.Stage);
        }
        return reached;
    }

    private static int[] Funnel(List<Event> events)
    {
        var funnel = new int[Stages.Length];
        foreach (KeyValuePair<string, HashSet<string>> pair in StagesByLearner(events))
        {
            int deepest = -1;
            for (int i = 0; i < Stages.Length; i++)
            {
                if (!pair.Value.Contains(Stages[i])) break;
                deepest = i;
            }
            for (int i = 0; i <= deepest; i++) funnel[i]++;
        }
        return funnel;
    }

    /** Oracle: per stage, count learners whose stage set contains the whole prefix. */
    private static int[] FunnelByPrefixPredicate(List<Event> events)
    {
        Dictionary<string, HashSet<string>> reached = StagesByLearner(events);
        var funnel = new int[Stages.Length];
        for (int stage = 0; stage < Stages.Length; stage++)
        {
            int count = 0;
            foreach (HashSet<string> set in reached.Values)
            {
                bool all = true;
                for (int j = 0; j <= stage; j++) all &= set.Contains(Stages[j]);
                if (all) count++;
            }
            funnel[stage] = count;
        }
        return funnel;
    }

    private static double[] DropOff(int[] funnel)
    {
        var rates = new double[funnel.Length];
        for (int i = 0; i + 1 < funnel.Length; i++)
            rates[i + 1] = funnel[i] == 0 ? 0.0 : 1.0 - funnel[i + 1] / (double)funnel[i];
        return rates;
    }

    private static string WorstDropStage(int[] funnel)
    {
        double[] rates = DropOff(funnel);
        int worst = 0;
        for (int i = 1; i < rates.Length; i++) if (rates[i] > rates[worst]) worst = i;
        return Stages[worst];
    }

    /** Share of the cohort that was active on each of the first `days` days after its day 0. */
    private static double[] Retention(List<Event> events, int cohortDay, int days)
    {
        var curve = new double[days];
        if (days <= 0) return curve;
        var firstDay = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (Event record in events)
            if (!firstDay.TryGetValue(record.Learner, out int day) || record.Day < day)
                firstDay[record.Learner] = record.Day;

        int cohortSize = 0;
        var active = new HashSet<string>[days];
        for (int d = 0; d < days; d++) active[d] = new HashSet<string>(StringComparer.Ordinal);
        foreach (KeyValuePair<string, int> pair in firstDay) if (pair.Value == cohortDay) cohortSize++;
        if (cohortSize == 0) return curve;
        foreach (Event record in events)
        {
            if (!firstDay.TryGetValue(record.Learner, out int start) || start != cohortDay) continue;
            int offset = record.Day - cohortDay;
            if (offset >= 0 && offset < days) active[offset].Add(record.Learner);
        }
        for (int d = 0; d < days; d++) curve[d] = active[d].Count / (double)cohortSize;
        return curve;
    }

    private static List<Event> SampleStream()
    {
        var stream = new List<Event>();
        string[][] journeys =
        {
            new[] { "view", "start", "attempt", "solve" },
            new[] { "view", "start", "attempt" },
            new[] { "view", "start" },
            new[] { "view" },
            new[] { "view", "solve" },
            new[] { "start", "solve" },
        };
        for (int learner = 0; learner < journeys.Length; learner++)
            foreach (string stage in journeys[learner])
                stream.Add(new Event("L" + learner, stage, 0));
        // L0 comes back on day 2 and L1 on day 1, which drives the retention curve.
        stream.Add(new Event("L0", "attempt", 2));
        stream.Add(new Event("L1", "attempt", 1));
        return stream;
    }

    public static void Main()
    {
        List<Event> stream = SampleStream();

        int[] funnel = Funnel(stream);
        Check(funnel[0] == 5 && funnel[1] == 3 && funnel[2] == 2 && funnel[3] == 1,
            "funnel counts are 5 viewed / 3 started / 2 attempted / 1 solved");
        int breaches = 0;
        for (int i = 1; i < funnel.Length; i++) if (funnel[i] > funnel[i - 1]) breaches++;
        Check(breaches == 0, "the funnel never grows between consecutive stages");

        double[] rates = DropOff(funnel);
        Check(Math.Abs(rates[1] - 0.4) < 1e-12 && Math.Abs(rates[3] - 0.5) < 1e-12,
            "stage drop-off rates are 40% at start and 50% at solve");
        double survivors = 1.0;
        for (int i = 1; i < rates.Length; i++) survivors *= 1.0 - rates[i];
        Check(Math.Abs(survivors * funnel[0] - funnel[3]) < 1e-9,
            "the solved count equals the product of the per-stage survival rates");

        // Negative cases: a skipped stage and a learner who never viewed anything.
        var skipOnly = new List<Event> { new Event("S", "view", 0), new Event("S", "solve", 0) };
        var startOnly = new List<Event> { new Event("T", "start", 0), new Event("T", "solve", 0) };
        Check(Funnel(skipOnly)[0] == 1 && Funnel(skipOnly)[3] == 0, "skipping 'start' truncates that learner at 'view'");
        Check(Funnel(startOnly)[1] == 0, "a learner who never viewed is not credited for starting");

        // Oracle: independent prefix-predicate formulation of the same funnel.
        int[] oracle = FunnelByPrefixPredicate(stream);
        bool agree = true;
        for (int i = 0; i < funnel.Length; i++) agree &= oracle[i] == funnel[i];
        Check(agree, "the fold matches the independent prefix-predicate oracle ("
            + string.Join("/", funnel) + " vs " + string.Join("/", oracle) + ")");

        // Seeded random cross-check on a synthetic event stream.
        _state = 970UL;
        var random = new List<Event>();
        for (int learner = 0; learner < 60; learner++)
        {
            int reached = (int)(NextUnit() * (Stages.Length + 1));
            for (int stage = 0; stage < reached; stage++)
                random.Add(new Event("R" + learner, Stages[stage], learner % 5));
        }
        int[] randomFunnel = Funnel(random);
        Check(string.Join("/", randomFunnel) == string.Join("/", FunnelByPrefixPredicate(random)),
            "60 seeded random learners: the fold still equals the oracle");
        Check(randomFunnel[0] >= randomFunnel[1] && randomFunnel[1] >= randomFunnel[2] && randomFunnel[2] >= randomFunnel[3],
            "the random funnel stays monotone, and every drop-off rate is inside [0,1]");
        double[] randomRates = DropOff(randomFunnel);
        bool bounded = true;
        for (int i = 0; i < randomRates.Length; i++) bounded &= randomRates[i] >= 0.0 && randomRates[i] <= 1.0;
        Check(bounded, "no random drop-off rate escapes [0,1]");

        Check(WorstDropStage(funnel) == "solve", "the stage with the worst drop-off is reported as 'solve' (50%)");

        double[] curve = Retention(stream, 0, 4);
        Check(curve[0] == 1.0, "the cohort is 100% active on day 0 by construction");
        Check(Math.Abs(curve[1] - 1.0 / 6.0) < 1e-12 && Math.Abs(curve[2] - 1.0 / 6.0) < 1e-12,
            "retention is 1/6 on day 1 and day 2 for the 6-learner day-0 cohort");
        Check(curve[3] == 0.0, "nobody returns on day 3, so retention reaches 0 inside the window");

        // Boundary: an empty stream, an unknown cohort day and a zero-length window.
        int[] emptyFunnel = Funnel(new List<Event>());
        double[] emptyRates = DropOff(emptyFunnel);
        bool clean = true;
        for (int i = 0; i < emptyRates.Length; i++) clean &= emptyRates[i] == 0.0 && !double.IsNaN(emptyRates[i]);
        Check(emptyFunnel[0] == 0 && clean, "an empty stream yields a zero funnel and no NaN drop-off rates");
        double[] unknownCohort = Retention(stream, 99, 4);
        bool zeroed = true;
        for (int d = 0; d < unknownCohort.Length; d++) zeroed &= unknownCohort[d] == 0.0;
        Check(zeroed, "an unknown cohort day produces an all-zero curve instead of dividing by zero");
        Check(Retention(stream, 0, 0).Length == 0, "a zero-length retention window returns an empty curve");

        Console.WriteLine("funnel: " + string.Join(" -> ", funnel) + " over stages "
            + string.Join("/", Stages));
        Console.WriteLine("drop-off: " + rates[1].ToString("F2") + " at start, worst stage = " + WorstDropStage(funnel));
        Console.WriteLine("retention(day 0 cohort, 4 days): "
            + string.Join(",", Array.ConvertAll(curve, v => v.ToString("F2"))));
        Console.WriteLine("PASS");
    }
}
