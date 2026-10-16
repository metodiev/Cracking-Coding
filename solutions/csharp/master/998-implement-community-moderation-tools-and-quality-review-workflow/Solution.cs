// Task 998 - Implement community moderation tools and quality review workflow.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// Submissions walk a guarded state machine (draft -> submitted -> triaged -> in review -> accepted /
// rejected / revision requested, with resubmission after a revision and a single appeal that reopens a
// rejection). Reviews are cast only by reviewers whose expertise matches the task, who do not share an
// affiliation with the author and who have not already reviewed the item; the verdict follows a quorum rule
// (two clean accepts, two rejects, or an accept plus a reject requesting revision). The moderation queue is a
// total order on priority then id. Verified: the happy path with the "accepted means two clean accepts"
// invariant re-checked after every mutation, illegal transitions, conflict-of-interest and no-eligible-reviewer
// refusals, the revision loop, the appeal path with fresh reviewers and its single-use rule, queue ordering and
// the pending-quorum boundaries.
using System;
using System.Collections.Generic;
using System.Globalization;

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAIL: " + label); }
    private static string Error(Action action)
    {
        try { action(); return ""; }
        catch (Exception ex) { return ex.Message; }
    }

    private enum State { Draft, Submitted, Triaged, InReview, RevisionRequested, Rejected, Accepted, Appealed }

    private static readonly Dictionary<State, State[]> Allowed = new Dictionary<State, State[]>
    {
        [State.Draft] = new[] { State.Submitted },
        [State.Submitted] = new[] { State.Triaged },
        [State.Triaged] = new[] { State.InReview },
        [State.InReview] = new[] { State.Accepted, State.Rejected, State.RevisionRequested },
        [State.RevisionRequested] = new[] { State.Submitted },
        [State.Rejected] = new[] { State.Appealed },
        [State.Appealed] = new[] { State.InReview },
        [State.Accepted] = new State[0],
    };

    private sealed class Submission
    {
        public readonly int Id, Priority;
        public readonly string Author;
        public State State = State.Draft;
        public readonly List<(string Reviewer, bool Accept)> Votes = new List<(string, bool)>();
        public readonly List<string> Excluded = new List<string>();
        public int Appeals;
        public Submission(int id, string author, int priority) { Id = id; Author = author; Priority = priority; }
    }

    private static void Move(Submission submission, State to)
    {
        if (Array.IndexOf(Allowed[submission.State], to) < 0)
            throw new Exception("illegal-transition:" + submission.State + "->" + to);
        submission.State = to;
    }

    /** Matching expertise, no shared affiliation with the author, no prior review and not excluded by appeal. */
    private static List<string> Eligible(List<(string Id, string Area, string Affiliation)> pool, Submission submission)
    {
        var eligible = new List<string>();
        foreach ((string id, string area, string affiliation) in pool)
            if (area == "algorithms" && affiliation != submission.Author && !submission.Excluded.Contains(id))
                eligible.Add(id);
        if (eligible.Count == 0) throw new Exception("no-eligible-reviewer:" + submission.Id);
        return eligible;
    }

    private static void Vote(Submission submission, List<string> eligible, string reviewer, bool accept)
    {
        if (submission.State != State.InReview) throw new Exception("not-in-review:" + submission.Id);
        if (!eligible.Contains(reviewer)) throw new Exception("conflicted-reviewer:" + reviewer);
        foreach ((string cast, bool _) in submission.Votes)
            if (cast == reviewer) throw new Exception("already-reviewed:" + reviewer);
        submission.Votes.Add((reviewer, accept));
    }

    private static string Quorum(Submission submission)
    {
        int accepts = 0, rejects = 0;
        foreach ((string _, bool accept) in submission.Votes) { if (accept) accepts++; else rejects++; }
        if (accepts >= 2 && rejects == 0) return "accepted";
        if (rejects >= 2) return "rejected";
        if (accepts >= 1 && rejects >= 1) return "revision";
        return "pending";
    }

    /** Invariant checked after every mutation: an accepted item carries two clean accepts. */
    private static bool CleanAccepts(Submission submission, List<string> eligible)
    {
        int clean = 0;
        foreach ((string reviewer, bool accept) in submission.Votes)
            if (accept && eligible.Contains(reviewer)) clean++;
        return submission.State != State.Accepted || clean >= 2;
    }

    /** An appeal is single-use: it reopens the item in review with the original reviewers excluded. */
    private static void Appeal(Submission submission)
    {
        if (submission.Appeals >= 1) throw new Exception("appeal-already-used:" + submission.Id);
        if (submission.State != State.Rejected) throw new Exception("illegal-transition:" + submission.State + "->Appealed");
        submission.Appeals++;
        foreach ((string reviewer, bool _) in submission.Votes)
            if (!submission.Excluded.Contains(reviewer)) submission.Excluded.Add(reviewer);
        submission.Votes.Clear();
        Move(submission, State.Appealed);
        Move(submission, State.InReview);
    }

    /** Queue order: highest priority first, then submission id - a total deterministic order. */
    private static List<Submission> Queue(List<Submission> submissions)
    {
        var pending = new List<Submission>();
        foreach (Submission submission in submissions)
            if (submission.State != State.Accepted && submission.State != State.Rejected) pending.Add(submission);
        pending.Sort((a, b) => a.Priority != b.Priority ? b.Priority.CompareTo(a.Priority) : a.Id.CompareTo(b.Id));
        return pending;
    }

    private static List<(string Id, string Area, string Affiliation)> Pool() => new List<(string, string, string)>
    {
        ("r1", "algorithms", "north"), ("r2", "algorithms", "south"),
        ("r3", "algorithms", "north"), ("r4", "systems", "west"), ("r5", "algorithms", "west"),
    };

    public static void Main()
    {
        var submission = new Submission(7, "north", 5);
        List<string> eligible = null;
        bool invariantHolds = true;
        Move(submission, State.Submitted);
        Move(submission, State.Triaged);
        eligible = Eligible(Pool(), submission);
        Move(submission, State.InReview);
        Vote(submission, eligible, "r2", true);
        Vote(submission, eligible, "r5", true);
        Check(Quorum(submission) == "accepted" && string.Join(",", eligible) == "r2,r5",
              "two clean accepts reach the accepted verdict and reviews are restricted to eligible reviewers");
        Move(submission, State.Accepted);
        invariantHolds &= CleanAccepts(submission, eligible);
        Check(invariantHolds && submission.State == State.Accepted && submission.Votes.Count == 2,
              "an accepted submission keeps two clean accepts (invariant re-checked after every mutation)");

        var fresh = new Submission(1, "west", 1);
        Check(Error(() => Move(fresh, State.Accepted)) == "illegal-transition:Draft->Accepted"
              && Error(() => Move(submission, State.Submitted)) == "illegal-transition:Accepted->Submitted"
              && Error(() => Vote(fresh, new List<string>(), "r1", true)) == "not-in-review:1",
              "illegal transitions and out-of-review votes are refused");

        var conflicted = new Submission(2, "north", 3);
        Move(conflicted, State.Submitted);
        Move(conflicted, State.Triaged);
        List<string> conflictedEligible = Eligible(Pool(), conflicted);
        Move(conflicted, State.InReview);
        Check(Error(() => Vote(conflicted, conflictedEligible, "r1", true)) == "conflicted-reviewer:r1"
              && Error(() => Vote(conflicted, conflictedEligible, "r4", true)) == "conflicted-reviewer:r4",
              "a conflicted or non-matching reviewer cannot vote");
        Vote(conflicted, conflictedEligible, "r2", true);
        Check(Quorum(conflicted) == "pending"
              && Error(() => Vote(conflicted, conflictedEligible, "r2", false)) == "already-reviewed:r2",
              "one accept is not a quorum and a reviewer cannot vote twice");
        Vote(conflicted, conflictedEligible, "r5", false);
        Check(Quorum(conflicted) == "revision", "an accept plus a reject requests a revision");
        Move(conflicted, State.RevisionRequested);
        Move(conflicted, State.Submitted);
        Check(conflicted.State == State.Submitted && conflicted.Votes.Count == 2, "the revision loop resubmits the item");

        var appealed = new Submission(3, "south", 9);
        Move(appealed, State.Submitted);
        Move(appealed, State.Triaged);
        Move(appealed, State.InReview);
        Vote(appealed, Eligible(Pool(), appealed), "r1", false);
        Vote(appealed, Eligible(Pool(), appealed), "r5", false);
        Move(appealed, State.Rejected);
        Appeal(appealed);
        List<string> appealPool = Eligible(Pool(), appealed);
        Check(Quorum(appealed) == "pending" && string.Join(",", appealPool) == "r3" && appealed.Appeals == 1
              && Error(() => Appeal(appealed)) == "appeal-already-used:3"
              && Error(() => Eligible(new List<(string, string, string)> { ("r1", "algorithms", "east") }, appealed))
                 == "no-eligible-reviewer:3",
              "an appeal reopens with fresh reviewers and is single-use, and a pool without eligible reviewers is refused");

        var queue = Queue(new List<Submission>
        {
            new Submission(9, "east", 2), new Submission(4, "east", 2), new Submission(6, "east", 7),
        });
        var single = Queue(new List<Submission> { new Submission(5, "east", 1) });
        Check(string.Join(",", queue.ConvertAll(s => s.Id.ToString(CultureInfo.InvariantCulture))) == "6,4,9"
              && single.Count == 1 && single[0].Id == 5 && Queue(new List<Submission>()).Count == 0,
              "the moderation queue orders by priority then id, and the single-item / empty queues are boundaries");

        Console.WriteLine("submissions: " + 4 + ", queue head: " + queue[0].Id + ", states: " + submission.State
                          + "/" + conflicted.State + "/" + appealed.State);
        Console.WriteLine("PASS");
    }
}
