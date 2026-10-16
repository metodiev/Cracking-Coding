// Task 982 - Implement monitoring and alerting playbooks for platform incidents.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// Traffic is modelled as per-minute success/error counters. The burn rate is the share of the error
// budget being consumed (errorRatio / (1 - SLO)), and a playbook pages only when both a short and a
// long window breach the threshold, so a one-minute spike is filtered out. Alerts are deduplicated
// per rule inside a cooldown, and every alert drives an incident through the open -> acknowledged ->
// resolved state machine with a deterministic escalation ladder while it stays unacknowledged.
//
// Verified: the burn-rate arithmetic, the healthy-traffic false-positive guard, short-versus-sustained
// multi-window behaviour, cooldown deduplication, the empty and zero-traffic boundaries, legal and
// illegal state transitions, the escalation ladder, the boundary case exactly at the SLO, and an
// independent recomputation of the burn rate from raw counters.
// Simplification: minutes are integer buckets in memory, the SLO is measured over a small fixed window
// and there is no real metric backend, pager or ticketing integration.
using System;
using System.Collections.Generic;

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    private sealed class Minute
    {
        public readonly int Index;
        public readonly long Success;
        public readonly long Errors;

        public Minute(int index, long success, long errors)
        {
            Index = index;
            Success = success;
            Errors = errors;
        }
    }

    private sealed class Alert
    {
        public readonly string Rule;
        public readonly int Minute;
        public readonly double ShortBurn;
        public readonly double LongBurn;

        public Alert(string rule, int minute, double shortBurn, double longBurn)
        {
            Rule = rule;
            Minute = minute;
            ShortBurn = shortBurn;
            LongBurn = longBurn;
        }
    }

    private sealed class Incident
    {
        public readonly string Id;
        public string State = "open";
        public string Owner = "primary";
        public int Escalations;

        public Incident(string id)
        {
            Id = id;
        }
    }

    private const double Slo = 0.99;          // 99% availability target, so the budget is 1%
    private const int ShortWindow = 5;
    private const int LongWindow = 60;
    private const double BurnThreshold = 2.0; // burning twice the sustainable budget

    private static double BurnRate(List<Minute> minutes, int from, int count)
    {
        long total = 0, errors = 0;
        for (int i = from; i < Math.Min(from + count, minutes.Count); i++)
        {
            if (i < 0) continue;
            total += minutes[i].Success + minutes[i].Errors;
            errors += minutes[i].Errors;
        }
        if (total == 0) return 0.0;
        return errors / (double)total / (1.0 - Slo);
    }

    private static List<Alert> Evaluate(List<Minute> minutes, int cooldown)
    {
        var alerts = new List<Alert>();
        var lastFired = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < minutes.Count; i++)
        {
            double shortBurn = BurnRate(minutes, Math.Max(0, i - ShortWindow + 1), ShortWindow);
            double longBurn = BurnRate(minutes, Math.Max(0, i - LongWindow + 1), LongWindow);
            bool breached = shortBurn > BurnThreshold && longBurn > BurnThreshold;
            foreach (string rule in new[] { "availability-burn" })
            {
                if (!breached) continue;
                if (lastFired.TryGetValue(rule, out int previous) && i - previous < cooldown) continue;
                lastFired[rule] = i;
                alerts.Add(new Alert(rule, i, shortBurn, longBurn));
            }
        }
        return alerts;
    }

    private static bool Transition(Incident incident, string next)
    {
        if (next == "acknowledged" && incident.State == "open")
        {
            incident.State = next;
            return true;
        }
        if (next == "resolved" && (incident.State == "open" || incident.State == "acknowledged"))
        {
            incident.State = next;
            return true;
        }
        return false;
    }

    /** Escalation ladder: unacknowledged incidents move one tier every `tierMinutes`. */
    private static Incident Escalate(Incident incident, int unackedMinutes, int tierMinutes)
    {
        if (incident.State != "open") return incident;
        int tier = unackedMinutes / tierMinutes;
        incident.Escalations = tier;
        incident.Owner = tier == 0 ? "primary" : tier == 1 ? "secondary" : "engineering-manager";
        return incident;
    }

    private static List<Minute> Healthy(int count)
    {
        var minutes = new List<Minute>();
        for (int i = 0; i < count; i++) minutes.Add(new Minute(i, 1000, i == 7 ? 5 : 1));
        return minutes;
    }

    private static List<Minute> Outage(int count, int at)
    {
        var minutes = Healthy(count);
        for (int i = at; i < count; i++) minutes[i] = new Minute(i, 500, 500);
        return minutes;
    }

    public static void Main()
    {
        // Burn-rate arithmetic: 5% errors against a 1% budget is a burn rate of 5.
        var burned = new List<Minute> { new Minute(0, 950, 50) };
        Check(Math.Abs(BurnRate(burned, 0, 1) - 5.0) < 1e-12, "5% errors against a 1% budget burns at 5x");

        // False-positive guard: healthy traffic never pages, even with one noisy minute.
        List<Alert> healthy = Evaluate(Healthy(90), 10);
        Check(healthy.Count == 0, "ninety healthy minutes produce no page");

        // Multi-window: a short spike is filtered, a sustained outage pages.
        List<Minute> spike = Healthy(90);
        spike[80] = new Minute(80, 100, 900);
        List<Alert> spikeAlerts = Evaluate(spike, 1);
        Check(spikeAlerts.Count == 0, "a single 90% error minute does not page because the long window stays healthy");
        List<Minute> outageTelemetry = Outage(90, 60);
        List<Alert> outageAlerts = Evaluate(outageTelemetry, 100);
        Check(outageAlerts.Count == 1 && outageAlerts[0].Minute >= 60 && outageAlerts[0].LongBurn > BurnThreshold,
            "a sustained outage pages once with both windows breaching");

        // Cooldown deduplication: with a 10 minute cooldown a 30 minute outage pages repeatedly but
        // much less often than once per minute.
        List<Minute> longOutage = Outage(120, 10);
        List<Alert> cold = Evaluate(longOutage, 10);
        List<Alert> hot = Evaluate(longOutage, 1);
        Check(cold.Count < hot.Count, "the cooldown reduces the page count (" + cold.Count + " < " + hot.Count + ")");
        bool spaced = true;
        for (int i = 1; i < cold.Count; i++) spaced &= cold[i].Minute - cold[i - 1].Minute >= 10;
        Check(spaced, "no two cooled-down pages are closer than the cooldown window");

        // Oracle: recompute the burn rate of the alerting minute straight from the counters.
        Alert first = outageAlerts[0];
        long total = 0, errors = 0;
        for (int i = first.Minute - ShortWindow + 1; i <= first.Minute; i++)
        {
            total += outageTelemetry[i].Success + outageTelemetry[i].Errors;
            errors += outageTelemetry[i].Errors;
        }
        Check(Math.Abs(errors / (double)total / (1.0 - Slo) - first.ShortBurn) < 1e-12,
            "an independent recount of the short window reproduces the alerting burn rate");

        // Incident state machine: only the documented transitions are accepted.
        var incident = new Incident("inc-1");
        Check(Transition(incident, "acknowledged") && incident.State == "acknowledged",
            "an open incident can be acknowledged");
        Check(Transition(incident, "resolved") && incident.State == "resolved", "an acknowledged incident can be resolved");
        Check(!Transition(incident, "acknowledged") && incident.State == "resolved",
            "a resolved incident cannot be acknowledged again");
        var reopened = new Incident("inc-2");
        Check(Transition(reopened, "resolved") && reopened.State == "resolved", "an open incident can be resolved directly");
        Check(!Transition(new Incident("inc-3"), "escalated"), "an unknown transition is rejected");

        // Escalation ladder while unacknowledged.
        var waiting = new Incident("inc-4");
        Check(Escalate(waiting, 5, 15).Owner == "primary" && waiting.Escalations == 0,
            "an incident unacknowledged for five minutes stays with the primary");
        Check(Escalate(waiting, 20, 15).Owner == "secondary" && waiting.Escalations == 1,
            "twenty minutes without acknowledgement escalates to the secondary");
        Check(Escalate(waiting, 45, 15).Owner == "engineering-manager" && waiting.Escalations == 3,
            "forty-five minutes escalates to the engineering manager");
        var acked = new Incident("inc-5");
        Transition(acked, "acknowledged");
        Check(Escalate(acked, 60, 15).Owner == "primary" && acked.Escalations == 0,
            "an acknowledged incident is never escalated");

        // Boundaries: an empty window and traffic with zero requests, plus the exact SLO edge.
        Check(BurnRate(new List<Minute>(), 0, 5) == 0.0 && Evaluate(new List<Minute>(), 10).Count == 0,
            "an empty telemetry window produces no alerts and no NaN");
        Check(BurnRate(new List<Minute> { new Minute(0, 0, 0) }, 0, 1) == 0.0,
            "a minute with no requests burns nothing instead of dividing by zero");
        var exact = new List<Minute> { new Minute(0, 990, 10) };
        Check(Math.Abs(BurnRate(exact, 0, 1) - 1.0) < 1e-12,
            "traffic exactly on the 99% SLO burns the budget at 1x and does not page");
        Check(Evaluate(exact, 10).Count == 0, "the exact-SLO minute stays below the 2x page threshold");

        Console.WriteLine("burn rates: healthy " + BurnRate(Healthy(10), 0, 5).ToString("F2")
            + ", outage " + BurnRate(longOutage, 60, 5).ToString("F2") + ", slo edge 1.00");
        Console.WriteLine("pages: sustained outage " + outageAlerts.Count + ", cooldown 10 -> " + cold.Count
            + ", cooldown 1 -> " + hot.Count);
        Console.WriteLine("escalation: 5min primary, 20min secondary, 45min engineering-manager");
        Console.WriteLine("PASS");
    }
}
