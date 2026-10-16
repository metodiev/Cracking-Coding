// Task 960 - Implement gamification layer with badges for completing tiers and streaks.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// The layer folds an append-only activity log into per-user profiles: distinct completed tasks (repeats
// are ignored), weighted points, per-tier completion counters, and both the longest and the current daily
// streak.  Badge rules are pure functions of those counters, so awarding is idempotent - replaying the
// log into a ledger that already holds the badges cannot duplicate them.
//
// Verified: points and streak figures are cross-checked against a brute-force oracle that recounts with
// different loops (nested scan for distinct tasks, boolean day array for the streak), a gap in the log
// splits the streak, tier badges need *all* tier tasks, repeated events do not inflate anything, the
// streak-7 boundary sits exactly at seven days, and the empty and single-event logs are well formed.
using System;
using System.Collections.Generic;
using System.Globalization;

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    private sealed class Activity
    {
        public readonly string User;
        public readonly int Day;
        public readonly string Tier;
        public readonly string TaskId;

        public Activity(string user, int day, string tier, string taskId)
        {
            User = user;
            Day = day;
            Tier = tier;
            TaskId = taskId;
        }
    }

    private static readonly Dictionary<string, int> TierWeight = new Dictionary<string, int>(StringComparer.Ordinal)
    {
        ["beginner"] = 1, ["intermediate"] = 2, ["advanced"] = 3, ["expert"] = 5, ["master"] = 8,
    };

    private static readonly Dictionary<string, int> TierSize = new Dictionary<string, int>(StringComparer.Ordinal)
    {
        ["beginner"] = 3, ["intermediate"] = 3, ["advanced"] = 3, ["expert"] = 2, ["master"] = 2,
    };

    private sealed class Profile
    {
        public int Points;
        public int CurrentStreak;
        public int LongestStreak;
        public readonly SortedSet<string> Tasks = new SortedSet<string>(StringComparer.Ordinal);
        public readonly SortedDictionary<string, int> PerTier = new SortedDictionary<string, int>(StringComparer.Ordinal);
        public readonly SortedSet<string> Badges = new SortedSet<string>(StringComparer.Ordinal);
    }

    private static (int Longest, int Current) Streaks(SortedSet<int> days)
    {
        if (days.Count == 0) return (0, 0);
        int longest = 1, run = 1;
        bool first = true;
        int previous = 0;
        foreach (int day in days)
        {
            if (!first)
            {
                run = day == previous + 1 ? run + 1 : 1;
                longest = Math.Max(longest, run);
            }
            previous = day;
            first = false;
        }
        int current = 0;
        int cursor = days.Max;
        while (days.Contains(cursor))
        {
            current++;
            cursor--;
        }
        return (longest, current);
    }

    private static Profile Summarize(List<Activity> log)
    {
        var profile = new Profile();
        var days = new SortedSet<int>();
        var tierTasks = new SortedDictionary<string, SortedSet<string>>(StringComparer.Ordinal);
        foreach (Activity activity in log)
        {
            days.Add(activity.Day);
            if (profile.Tasks.Add(activity.TaskId))
            {
                if (!tierTasks.TryGetValue(activity.Tier, out SortedSet<string> bucket))
                {
                    bucket = new SortedSet<string>(StringComparer.Ordinal);
                    tierTasks[activity.Tier] = bucket;
                }
                bucket.Add(activity.TaskId);
                profile.Points += TierWeight.TryGetValue(activity.Tier, out int weight) ? weight : 0;
            }
        }
        foreach (KeyValuePair<string, SortedSet<string>> pair in tierTasks) profile.PerTier[pair.Key] = pair.Value.Count;
        (profile.LongestStreak, profile.CurrentStreak) = Streaks(days);
        if (profile.Tasks.Count >= 1) profile.Badges.Add("first-steps");
        if (profile.Tasks.Count >= 10) profile.Badges.Add("ten-tasks");
        if (profile.Points >= 100) profile.Badges.Add("hundred-club");
        if (profile.LongestStreak >= 7) profile.Badges.Add("streak-7");
        if (profile.PerTier.Count >= 3) profile.Badges.Add("polyglot");
        foreach (KeyValuePair<string, int> pair in profile.PerTier)
        {
            if (TierSize.TryGetValue(pair.Key, out int size) && pair.Value >= size) profile.Badges.Add("tier-complete-" + pair.Key);
        }
        return profile;
    }

    /** Independent oracle: recount points, distinct tasks, tiers and the longest streak with other loops. */
    private static (int Points, int Longest, int Tiers, int Tasks) BruteForce(List<Activity> log)
    {
        var distinct = new List<Activity>();
        foreach (Activity candidate in log)
        {
            bool seen = false;
            foreach (Activity known in distinct)
            {
                if (known.TaskId == candidate.TaskId)
                {
                    seen = true;
                    break;
                }
            }
            if (!seen) distinct.Add(candidate);
        }
        int points = 0;
        var tiers = new SortedSet<string>(StringComparer.Ordinal);
        foreach (Activity activity in distinct)
        {
            points += TierWeight[activity.Tier];
            tiers.Add(activity.Tier);
        }
        int longest = 0;
        if (log.Count > 0)
        {
            int min = int.MaxValue, max = int.MinValue;
            foreach (Activity activity in log)
            {
                min = Math.Min(min, activity.Day);
                max = Math.Max(max, activity.Day);
            }
            var active = new bool[max - min + 1];
            foreach (Activity activity in log) active[activity.Day - min] = true;
            int run = 0;
            foreach (bool day in active)
            {
                run = day ? run + 1 : 0;
                longest = Math.Max(longest, run);
            }
        }
        return (points, longest, tiers.Count, distinct.Count);
    }

    public static void Main()
    {
        var log = new List<Activity>
        {
            new Activity("ada", 1, "beginner", "b1"),
            new Activity("ada", 2, "beginner", "b2"),
            new Activity("ada", 3, "beginner", "b3"),
            new Activity("ada", 3, "intermediate", "i1"),
            new Activity("ada", 7, "intermediate", "i2"),
            new Activity("ada", 8, "expert", "e1"),
            new Activity("ada", 8, "expert", "e1"), // duplicate completion
            new Activity("grace", 4, "master", "m1"),
        };

        Profile ada = Summarize(log.FindAll(a => a.User == "ada"));
        Profile grace = Summarize(log.FindAll(a => a.User == "grace"));
        Check(ada.Tasks.Count == 6, "duplicate completions do not inflate the distinct task count (" + ada.Tasks.Count + ")");
        Check(ada.Points == 1 + 1 + 1 + 2 + 2 + 5, "points equal the hand-summed tier weights (" + ada.Points + ")");

        var brute = BruteForce(log.FindAll(a => a.User == "ada"));
        Check(brute.Points == ada.Points && brute.Tasks == ada.Tasks.Count && brute.Tiers == ada.PerTier.Count,
            "the brute-force oracle agrees on points, distinct tasks and tiers");
        Check(brute.Longest == ada.LongestStreak, "the boolean-array oracle agrees on the longest streak (" + ada.LongestStreak + ")");

        Check(ada.LongestStreak == 3 && ada.CurrentStreak == 2,
            "the gap between days 3 and 7 splits the run: longest 3, current 2 (" + ada.LongestStreak + "/" + ada.CurrentStreak + ")");
        Check(ada.Badges.Contains("tier-complete-beginner"), "completing all three beginner tasks awards the tier badge");
        Check(!ada.Badges.Contains("tier-complete-intermediate"), "a partially completed intermediate tier awards nothing");
        Check(!ada.Badges.Contains("streak-7"), "a three-day longest streak does not award the seven-day badge");
        Check(ada.Badges.Contains("polyglot") && ada.Badges.Contains("first-steps"), "the beginner profile earns first-steps and polyglot");
        Check(grace.Points == 8 && grace.PerTier.Count == 1, "grace's single master task scores 8 points in one tier");

        // streak-7 boundary: six active days is not enough, seven is.
        var sixDays = new List<Activity>();
        for (int day = 1; day <= 6; day++) sixDays.Add(new Activity("six", day, "beginner", "t" + day));
        var sevenDays = new List<Activity>(sixDays) { new Activity("seven", 7, "beginner", "t7") };
        Check(!Summarize(sixDays).Badges.Contains("streak-7") && Summarize(sixDays).LongestStreak == 6,
            "six consecutive days stops one short of the streak badge");
        Check(Summarize(sevenDays).Badges.Contains("streak-7"), "seven consecutive days earns the streak badge");

        var empty = Summarize(new List<Activity>());
        Check(empty.Points == 0 && empty.LongestStreak == 0 && empty.CurrentStreak == 0 && empty.Badges.Count == 0,
            "an empty log yields no points, no streak and no badges");
        var single = Summarize(new List<Activity> { new Activity("solo", 5, "beginner", "b1") });
        Check(single.CurrentStreak == 1 && single.LongestStreak == 1 && single.Badges.Count == 1 && single.Badges.Contains("first-steps"),
            "the single-event boundary produces a one-day streak and only first-steps");

        // Idempotent awarding: replaying the log into an existing ledger adds nothing new.
        var ledger = new SortedSet<string>(StringComparer.Ordinal);
        foreach (string badge in ada.Badges) ledger.Add(badge);
        int before = ledger.Count;
        foreach (string badge in Summarize(log.FindAll(a => a.User == "ada")).Badges) ledger.Add(badge);
        Check(ledger.Count == before, "replaying the same log into the ledger awards no duplicate badges");

        var repeat = Summarize(log.FindAll(a => a.User == "ada"));
        Check(string.Join(",", repeat.Badges) == string.Join(",", ada.Badges), "badge evaluation is deterministic and sorted");

        Console.WriteLine("ada: " + ada.Tasks.Count + " tasks, " + ada.Points + " points, tiers " + ada.PerTier.Count +
            ", streak " + ada.CurrentStreak + "/" + ada.LongestStreak);
        Console.WriteLine("badges: " + string.Join(", ", ada.Badges));
        Console.WriteLine("oracle: points=" + brute.Points + " tasks=" + brute.Tasks + " tiers=" + brute.Tiers + " longest=" + brute.Longest);
        Console.WriteLine("PASS");
    }
}
