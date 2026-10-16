// Task 971 - Implement spaced-repetition scheduling for problem revisit recommendations.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// Implements the SM-2 scheduler: each card carries an ease factor, a repetition streak and an
// interval. A recall grade 0..5 first moves the ease factor by
//   EF' = max(1.3, EF + 0.1 - (5-q) * (0.08 + (5-q) * 0.02)),
// then either resets the streak to a one-day relearn (q < 3) or expands the interval along the
// 1, 6, interval*EF' ladder. Due cards are handed back as a queue ordered by due day, then id.
//
// Verified: the hand-computed 1/6/17/49/147 pure-success ladder (independent table oracle), the
// lowest-passing-grade boundary, failure reset, the 1.3 ease floor as an invariant re-checked after
// every one of 30 mutations, the queue tie-break, an independently written reference scheduler
// replayed over 24 seeded grades, rejection of out-of-range grades and the empty-queue boundary.
// Simplification: days are integer counters and grading is supplied by the caller, so no calendar,
// clock or learner model is involved.
using System;
using System.Collections.Generic;

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

    private static double NextUnit() => (NextUInt64() >> 11) * (1.0 / 9007199254740992.0);

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    private sealed class Card
    {
        public readonly string Id;
        public double Ease = 2.5;
        public int Repetitions;
        public int IntervalDays = 1;
        public int DueDay;

        public Card(string id, int dueDay)
        {
            Id = id;
            DueDay = dueDay;
        }
    }

    private static void Grade(Card card, int quality, int today)
    {
        if (quality < 0 || quality > 5)
            throw new ArgumentException("quality must be within 0..5");
        double delta = 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02);
        card.Ease = Math.Max(1.3, card.Ease + delta);
        if (quality < 3)
        {
            card.Repetitions = 0;
            card.IntervalDays = 1;   // the ease penalty stands, the streak does not
        }
        else
        {
            card.Repetitions++;
            if (card.Repetitions == 1) card.IntervalDays = 1;
            else if (card.Repetitions == 2) card.IntervalDays = 6;
            else card.IntervalDays = (int)Math.Round(Math.Max(card.IntervalDays, 1) * card.Ease);
        }
        card.DueDay = today + card.IntervalDays;
    }

    /** Independently written reference scheduler used as the oracle for the incremental card walk. */
    private static List<int> ReferenceIntervals(List<int> qualities)
    {
        double ease = 2.5;
        int repetitions = 0;
        int interval = 1;
        var intervals = new List<int>();
        foreach (int quality in qualities)
        {
            ease = Math.Max(1.3, ease + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02));
            if (quality < 3)
            {
                repetitions = 0;
                interval = 1;
            }
            else
            {
                repetitions = repetitions + 1;
                if (repetitions == 1) interval = 1;
                else if (repetitions == 2) interval = 6;
                else interval = (int)Math.Round(interval * ease);
            }
            intervals.Add(interval);
        }
        return intervals;
    }

    private static List<string> Schedule(List<Card> cards)
    {
        var queue = new List<Card>(cards);
        queue.Sort((a, b) => a.DueDay != b.DueDay
            ? a.DueDay.CompareTo(b.DueDay)
            : string.CompareOrdinal(a.Id, b.Id));
        var order = new List<string>();
        foreach (Card card in queue) order.Add(card.Id);
        return order;
    }

    public static void Main()
    {
        // Happy path: a perfect recall streak follows the doubling ladder.
        var perfect = new Card("perfect", 0);
        var ladder = new List<int>();
        for (int i = 0; i < 5; i++)
        {
            Grade(perfect, 5, 0);
            ladder.Add(perfect.IntervalDays);
        }
        Check(string.Join(",", ladder) == "1,6,17,49,147",
            "a q=5 streak schedules 1,6,17,49,147 days (hand-computed oracle, got " + string.Join(",", ladder) + ")");
        Check(Math.Abs(perfect.Ease - 3.0) < 1e-12, "five perfect recalls raise the ease to 2.5 + 5*0.1 = 3.0");
        Check(perfect.DueDay == 147, "the due day is today plus the freshly computed interval");

        // Boundary: the lowest passing grade advances the streak but by only one day.
        var borderline = new Card("borderline", 0);
        Grade(borderline, 3, 0);
        Check(borderline.Repetitions == 1 && borderline.IntervalDays == 1,
            "the first review at the lowest passing grade is scheduled one day out");
        Check(Math.Abs(borderline.Ease - 2.36) < 1e-12, "q=3 lowers the ease by 0.14 to 2.36");

        Card struggling = new Card("struggling", 0);
        for (int i = 0; i < 3; i++) Grade(struggling, 5, 0);
        Check(struggling.IntervalDays == 17 && struggling.Repetitions == 3, "three recalls reach the 17 day interval");
        Grade(struggling, 2, 17);
        Check(struggling.Repetitions == 0 && struggling.IntervalDays == 1 && struggling.DueDay == 18,
            "a failed recall resets the streak to a one day relearn");

        // Invariant re-checked after every mutation: the ease factor never drops below 1.3.
        var floor = new Card("floor", 0);
        double lowest = 2.5;
        for (int i = 0; i < 30; i++)
        {
            Grade(floor, 0, 0);
            lowest = Math.Min(lowest, floor.Ease);
        }
        Check(floor.Ease == 1.3 && lowest == 1.3, "30 consecutive failures hold the ease exactly at the 1.3 floor");
        Check(floor.IntervalDays == 1 && floor.Repetitions == 0, "the floored card stays in relearning instead of drifting");

        // Oracle: replay a seeded grade stream against the independently written reference scheduler.
        _state = 971UL;
        var stream = new List<int>();
        for (int i = 0; i < 24; i++) stream.Add((int)(NextUnit() * 6.0));
        var replay = new Card("replay", 0);
        var actual = new List<int>();
        int day = 0;
        foreach (int quality in stream)
        {
            Grade(replay, quality, day);
            actual.Add(replay.IntervalDays);
            day += replay.IntervalDays;
        }
        Check(string.Join(",", actual) == string.Join(",", ReferenceIntervals(stream)),
            "the incremental card walk matches the reference schedule over 24 seeded grades");
        Check(replay.DueDay == day, "every replayed review moved the due day by exactly its interval");
        int shortened = 0, nonPositive = 0;
        for (int i = 0; i < actual.Count; i++)
        {
            if (actual[i] < 1) nonPositive++;
            if (i > 0 && stream[i] >= 3 && stream[i - 1] >= 3 && actual[i] <= actual[i - 1]) shortened++;
        }
        Check(nonPositive == 0, "no seeded grade sequence ever schedules a non-positive interval");
        Check(shortened == 0, "two consecutive passing grades never shorten the interval");

        // Queue: due day first, id as the deterministic tie-break.
        var queue = new List<Card> { new Card("c", 3), new Card("a", 1), new Card("b", 1), new Card("d", 7) };
        Check(string.Join(",", Schedule(queue)) == "a,b,c,d", "the revisit queue is ordered by due day then id");
        Check(Schedule(new List<Card>()).Count == 0, "an empty queue schedules nothing");

        bool rejected = false;
        try { Grade(new Card("bad", 0), 6, 0); }
        catch (ArgumentException) { rejected = true; }
        Check(rejected, "a grade outside 0..5 is rejected instead of being silently clamped");

        Console.WriteLine("ladder q=5: " + string.Join(",", ladder) + " days, ease = " + perfect.Ease.ToString("F1"));
        Console.WriteLine("floor: ease pinned at " + floor.Ease.ToString("F1") + " after 30 failures");
        Console.WriteLine("seeded replay: " + actual.Count + " reviews, total horizon " + day + " days");
        Console.WriteLine("PASS");
    }
}
