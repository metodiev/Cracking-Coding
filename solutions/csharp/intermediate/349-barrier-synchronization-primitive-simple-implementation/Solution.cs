// Task 349 - Reusable barrier synchronization primitive. (Tier: Intermediate, CTCI Ch 15)
using System;
using System.Collections.Generic;
using System.Threading;

public class Barrier
{
    private readonly object gate = new object();
    private readonly int parties;
    private int count;
    private int generation;

    public Barrier(int parties)
    {
        if (parties <= 0) throw new ArgumentException("parties must be positive");
        this.parties = parties;
    }

    /// <summary>
    /// Generation-based: the last arriving participant resets the counter, flips the generation and
    /// wakes everyone, so the barrier is instantly reusable.  Returns false when the caller timed
    /// out before all parties arrived, withdrawing its own count so the barrier stays usable.
    /// </summary>
    public bool Await(long timeoutMillis)
    {
        lock (gate)
        {
            int observed = generation;
            count++;
            if (count == parties)
            {
                count = 0;
                generation++;
                Monitor.PulseAll(gate);
                return true;
            }
            long deadline = timeoutMillis > 0 ? Environment.TickCount64 + timeoutMillis : 0;
            while (observed == generation)
            {
                if (timeoutMillis <= 0)
                {
                    Monitor.Wait(gate);
                }
                else
                {
                    long remaining = deadline - Environment.TickCount64;
                    if (remaining <= 0)
                    {
                        count--;                       // withdraw; the barrier stays usable
                        return false;
                    }
                    Monitor.Wait(gate, (int)Math.Min(remaining, int.MaxValue));
                }
            }
            return true;
        }
    }

    public int[] State()
    {
        lock (gate) return new[] { count, generation };
    }
}

public static class Solution
{
    private class Event
    {
        public Event(int seq, string kind, int round, string name)
        {
            Seq = seq;
            Kind = kind;
            Round = round;
            Name = name;
        }

        public int Seq { get; }
        public string Kind { get; }
        public int Round { get; }
        public string Name { get; }
    }

    private static readonly object recordGate = new object();

    private static void Record(List<Event> events, string kind, int round, string name)
    {
        lock (recordGate) events.Add(new Event(events.Count, kind, round, name));
    }

    public static void Main()
    {
        int rounds = 5;
        var barrier = new Barrier(4);
        var events = new List<Event>();
        var failures = new List<string>();
        string[] names = { "P0", "P1", "P2", "P3" };
        var threads = new List<Thread>();
        foreach (string name in names)
        {
            string participant = name;
            threads.Add(new Thread(() =>
            {
                for (int round = 0; round < rounds; round++)
                {
                    Record(events, "arrive", round, participant);
                    if (!barrier.Await(15000))
                    {
                        lock (recordGate) failures.Add(participant + ": timed out");
                        return;
                    }
                    Record(events, "depart", round, participant);
                }
            })
            { IsBackground = true, Name = name });
        }
        foreach (Thread thread in threads) thread.Start();
        foreach (Thread thread in threads) Check(thread.Join(TimeSpan.FromSeconds(15)), "participant terminated");
        Check(failures.Count == 0, "no participant failed: " + string.Join("; ", failures));

        int arrivals = 0;
        int departures = 0;
        foreach (Event entry in events)
        {
            if (entry.Kind == "arrive") arrivals++;
            else departures++;
        }
        Check(arrivals == 4 * rounds, "20 arrivals");
        Check(departures == 4 * rounds, "20 departures");

        for (int round = 0; round < rounds; round++)
        {
            int maxArrival = -1;
            int minDeparture = int.MaxValue;
            int arrivalCount = 0;
            int departureCount = 0;
            foreach (Event entry in events)
            {
                if (entry.Round != round) continue;
                if (entry.Kind == "arrive")
                {
                    arrivalCount++;
                    maxArrival = Math.Max(maxArrival, entry.Seq);
                }
                else
                {
                    departureCount++;
                    minDeparture = Math.Min(minDeparture, entry.Seq);
                }
            }
            Check(arrivalCount == 4 && departureCount == 4, "round " + round + " counts");
            Check(maxArrival < minDeparture, "round " + round + " released early");
        }

        foreach (string name in names)
        {
            var arrivalRounds = new List<int>();
            var departureRounds = new List<int>();
            foreach (Event entry in events)
            {
                if (entry.Name != name) continue;
                if (entry.Kind == "arrive") arrivalRounds.Add(entry.Round);
                else departureRounds.Add(entry.Round);
            }
            Check(arrivalRounds.Count == rounds && departureRounds.Count == rounds, name + " passed every round");
            for (int round = 0; round < rounds; round++)
            {
                Check(arrivalRounds[round] == round, name + " arrival round order");
                Check(departureRounds[round] == round, name + " departure round order");
            }
        }

        Check(string.Join(",", barrier.State()) == "0," + rounds, "barrier reset after every round");

        var solo = new Barrier(1);
        for (int i = 0; i < 3; i++) Check(solo.Await(0), "single-participant barrier is a no-op");
        Check(string.Join(",", solo.State()) == "0,3", "solo generation counted");

        var pair = new Barrier(2);
        Check(!pair.Await(50), "timeout returns false");
        Check(string.Join(",", pair.State()) == "0,0", "timed-out caller withdrew cleanly");

        var results = new List<bool>();
        var threadA = new Thread(() =>
        {
            bool released = pair.Await(15000);        // compute first: never hold a lock while waiting
            lock (recordGate) results.Add(released);
        })
        { IsBackground = true };
        var threadB = new Thread(() =>
        {
            bool released = pair.Await(15000);
            lock (recordGate) results.Add(released);
        })
        { IsBackground = true };
        threadA.Start();
        threadB.Start();
        Check(threadA.Join(TimeSpan.FromSeconds(15)) && threadB.Join(TimeSpan.FromSeconds(15)), "pair threads terminated");
        Check(results.Count == 2 && results[0] && results[1], "both parties released");
        Check(string.Join(",", pair.State()) == "0,1", "pair generation advanced once");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
