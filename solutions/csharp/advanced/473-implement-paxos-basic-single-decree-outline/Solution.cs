// Task 473 - Implement Paxos basic single-decree outline. (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)
using System;
using System.Collections.Generic;

internal class Acceptor
{
    public readonly int Id;
    public int Promised = -1;
    public int AcceptedRound = -1;
    public int? AcceptedValue = null;
    public readonly List<int[]> History = new List<int[]>(); // {round, value}

    public Acceptor(int id) => Id = id;
}

internal class Proposer
{
    public readonly int Id;
    public string Phase = "idle"; // idle | P1 | P2 | done
    public int RoundNo = -1;
    public int? OwnValue = null;
    public int? DecidedValue = null;
    public readonly List<object[]> Promises = new List<object[]>(); // {acceptorId, acceptedRound, acceptedValue}
    public readonly HashSet<int> Accepts = new HashSet<int>();
    public int[] Chosen = null; // {round, value}

    public Proposer(int id) => Id = id;
}

internal class PaxosSim
{
    private const int Majority = 2; // of 3 acceptors

    public readonly Acceptor[] Acceptors = new Acceptor[3];
    public readonly Proposer[] Proposers = new Proposer[2];
    private readonly Queue<object[]> queue = new Queue<object[]>(); // {destKind, destId, msg}
    public readonly List<int[]> ChosenEvents = new List<int[]>();
    public readonly HashSet<int> OracleChosen = new HashSet<int>();
    private readonly int[] lastPromised = { -1, -1, -1 };

    public PaxosSim()
    {
        for (int i = 0; i < 3; i++) Acceptors[i] = new Acceptor(i);
        for (int i = 0; i < 2; i++) Proposers[i] = new Proposer(i);
    }

    public void Propose(int p, int roundNo, int value)
    {
        Proposer pr = Proposers[p];
        pr.Phase = "P1";
        pr.RoundNo = roundNo;
        pr.OwnValue = value;
        pr.DecidedValue = null;
        pr.Promises.Clear();
        pr.Accepts.Clear();
        for (int a = 0; a < 3; a++)
        {
            queue.Enqueue(new object[] { "A", a, new object[] { "prepare", roundNo, p } });
        }
    }

    private void DeliverOne()
    {
        object[] item = queue.Dequeue();
        string destKind = (string)item[0];
        int destId = (int)item[1];
        object[] msg = (object[])item[2];
        string kind = (string)msg[0];
        if (destKind == "A")
        {
            Acceptor a = Acceptors[destId];
            if (kind == "prepare")
            {
                int roundNo = (int)msg[1];
                int p = (int)msg[2];
                if (roundNo > a.Promised)
                {
                    a.Promised = roundNo;
                    queue.Enqueue(new object[] { "P", p, new object[] { "promise", roundNo, a.Id, a.AcceptedRound, a.AcceptedValue } });
                }
                else
                {
                    queue.Enqueue(new object[] { "P", p, new object[] { "nack", roundNo, a.Id } });
                }
            }
            else
            {
                int roundNo = (int)msg[1];
                int value = (int)msg[2];
                int proposer = (int)msg[3];
                if (roundNo >= a.Promised)
                {
                    a.Promised = roundNo;
                    a.AcceptedRound = roundNo;
                    a.AcceptedValue = value;
                    a.History.Add(new[] { roundNo, value });
                    queue.Enqueue(new object[] { "P", proposer, new object[] { "accepted", roundNo, a.Id, value } });
                }
                else
                {
                    queue.Enqueue(new object[] { "P", proposer, new object[] { "nack", roundNo, a.Id } });
                }
            }
        }
        else
        {
            Proposer pr = Proposers[destId];
            if (kind == "promise")
            {
                int roundNo = (int)msg[1];
                int accId = (int)msg[2];
                int accRound = (int)msg[3];
                int? accValue = (int?)msg[4];
                if (pr.Phase != "P1" || roundNo != pr.RoundNo) return;
                pr.Promises.Add(new object[] { accId, accRound, accValue });
                if (pr.Promises.Count >= Majority)
                {
                    int bestRound = -1;
                    int bestValue = pr.OwnValue.Value;
                    foreach (object[] promise in pr.Promises)
                    {
                        int? value = (int?)promise[2];
                        if (value.HasValue && (int)promise[1] > bestRound)
                        {
                            bestRound = (int)promise[1];
                            bestValue = value.Value;
                        }
                    }
                    pr.DecidedValue = bestValue;
                    pr.Phase = "P2";
                    for (int a = 0; a < 3; a++)
                    {
                        queue.Enqueue(new object[] { "A", a, new object[] { "accept", roundNo, bestValue, pr.Id } });
                    }
                }
            }
            else if (kind == "accepted")
            {
                int roundNo = (int)msg[1];
                int accId = (int)msg[2];
                int value = (int)msg[3];
                if (pr.Phase != "P2" || roundNo != pr.RoundNo) return;
                pr.Accepts.Add(accId);
                if (pr.Accepts.Count >= Majority)
                {
                    pr.Chosen = new[] { roundNo, value };
                    pr.Phase = "done";
                    ChosenEvents.Add(new[] { roundNo, value });
                }
            }
        }
    }

    public void Deliver(int k)
    {
        for (int i = 0; i < k && queue.Count > 0; i++) DeliverOne();
    }

    public void Drain()
    {
        while (queue.Count > 0) DeliverOne();
    }

    public void Drop(int k)
    {
        for (int i = 0; i < k && queue.Count > 0; i++) queue.Dequeue();
    }

    public void Check()
    {
        var oracle = new HashSet<int>();
        foreach (Acceptor a in Acceptors)
        {
            foreach (int[] entry in a.History)
            {
                int count = 0;
                foreach (Acceptor b in Acceptors)
                {
                    foreach (int[] other in b.History)
                    {
                        if (other[0] == entry[0] && other[1] == entry[1]) count++;
                    }
                }
                if (count >= Majority) oracle.Add(entry[1]);
            }
        }
        foreach (int v in oracle) OracleChosen.Add(v);
        var chosenValues = new HashSet<int>(OracleChosen);
        foreach (int[] ev in ChosenEvents) chosenValues.Add(ev[1]);
        if (chosenValues.Count > 1) throw new Exception("FAILED: at most one value may be chosen: " + string.Join(",", chosenValues));
        foreach (Acceptor a in Acceptors)
        {
            if (a.Promised < lastPromised[a.Id]) throw new Exception("FAILED: promise regression");
            lastPromised[a.Id] = a.Promised;
            for (int i = 1; i < a.History.Count; i++)
            {
                if (a.History[i][0] < a.History[i - 1][0]) throw new Exception("FAILED: accept rounds not monotone");
            }
        }
    }

    public void Run(List<object[]> actions)
    {
        foreach (object[] action in actions)
        {
            string kind = (string)action[0];
            if (kind == "propose") Propose((int)action[1], (int)action[2], (int)action[3]);
            else if (kind == "deliver") Deliver((int)action[1]);
            else if (kind == "drain") Drain();
            else if (kind == "drop") Drop((int)action[1]);
            Check();
        }
    }

    public static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}

public static class Solution
{
    private static int Lcg(int seed) => (seed * 75 + 74) % 65537;

    private static List<object[]> RandomActions(int count, int seed, int startRound)
    {
        var actions = new List<object[]>();
        int rnd = startRound;
        for (int i = 0; i < count; i++)
        {
            seed = Lcg(seed);
            int a = seed % 8;
            seed = Lcg(seed);
            int p = seed % 2;
            seed = Lcg(seed);
            int value = seed % 20;
            seed = Lcg(seed);
            int k = seed % 3;
            if (a < 3)
            {
                actions.Add(new object[] { "propose", p, rnd, value });
                rnd++;
            }
            else if (a == 3)
            {
                actions.Add(new object[] { "deliver", 1 + k });
            }
            else if (a < 6)
            {
                actions.Add(new object[] { "drain" });
            }
            else if (a == 6)
            {
                actions.Add(new object[] { "drop", k });
            }
            else
            {
                actions.Add(new object[] { "deliver", 1 });
            }
        }
        return actions;
    }

    private static List<object[]> Actions(params object[][] items) => new List<object[]>(items);

    public static void Main()
    {
        var sim = new PaxosSim();

        // P0's value 7 is accepted by a majority (A0, A1) but all replies are dropped.
        sim.Run(Actions(
            new object[] { "propose", 0, 1000, 7 }, new object[] { "deliver", 3 }, new object[] { "deliver", 2 },
            new object[] { "deliver", 3 }, new object[] { "drop", 3 }));
        PaxosSim.Check(sim.ChosenEvents.Count == 0, "P0 unaware of the choice");
        PaxosSim.Check(sim.OracleChosen.Count == 1 && sim.OracleChosen.Contains(7), "7 chosen by oracle");
        PaxosSim.Check(sim.Acceptors[0].History.Count == 1 && sim.Acceptors[0].History[0][1] == 7, "A0 accepted 7");
        PaxosSim.Check(sim.Acceptors[1].History.Count == 1 && sim.Acceptors[1].History[0][1] == 7, "A1 accepted 7");
        PaxosSim.Check(sim.Acceptors[2].History.Count == 0, "A2 has no accept");

        // P1 proposes 9 with a higher round; the promises carry (1000, 7), so it must re-propose 7.
        sim.Run(Actions(new object[] { "propose", 1, 1001, 9 }, new object[] { "drain" }));
        PaxosSim.Check(sim.Proposers[1].DecidedValue == 7, "P1 must adopt the chosen value, not its own 9");
        PaxosSim.Check(sim.Proposers[1].Chosen != null && sim.Proposers[1].Chosen[0] == 1001 && sim.Proposers[1].Chosen[1] == 7,
            "P1 chose 7 at round 1001");
        PaxosSim.Check(sim.ChosenEvents.Count == 1, "one engine chosen event");

        // P0 retries with round 1002 and its own value 8; it must also re-propose 7.
        sim.Run(Actions(new object[] { "propose", 0, 1002, 8 }, new object[] { "drain" }));
        PaxosSim.Check(sim.Proposers[0].DecidedValue == 7, "P0 retry must also re-propose 7");
        PaxosSim.Check(sim.ChosenEvents.Count == 2 && sim.ChosenEvents[1][0] == 1002 && sim.ChosenEvents[1][1] == 7,
            "round 1002 chose 7");
        var acceptedValues = new HashSet<int>();
        foreach (Acceptor acc in sim.Acceptors) foreach (int[] entry in acc.History) acceptedValues.Add(entry[1]);
        PaxosSim.Check(acceptedValues.Count == 1 && acceptedValues.Contains(7),
            "values 9 and 8 must never be accepted anywhere");

        // Randomized churn with losses: safety (at most one chosen value) must hold throughout.
        sim.Run(RandomActions(30, 13579, 2000));

        // Fair finale: fresh proposals with all messages delivered; liveness must return.
        for (int p = 0; p < 2; p++)
        {
            sim.Run(Actions(new object[] { "propose", p, 3000 + p, 5 + p }, new object[] { "drain" }));
        }
        PaxosSim.Check(sim.ChosenEvents.Count > 0, "fair run must choose a value");
        var chosenValues = new HashSet<int>();
        foreach (int[] ev in sim.ChosenEvents) chosenValues.Add(ev[1]);
        PaxosSim.Check(chosenValues.Count == 1 && chosenValues.Contains(7), "every choice is 7");
        PaxosSim.Check(sim.OracleChosen.Count == 1 && sim.OracleChosen.Contains(7), "oracle agrees");
        Console.WriteLine("PASS");
    }
}
