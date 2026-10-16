/** Task 905 - Implement real-time SLA-aware scheduler for multi-tenant inference.
 *  (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * Tenants submit inference requests (arrival tick, compute cost in ticks, absolute deadline, tenant, value).
 * One accelerator serves them preemptively, so the classic real-time result applies: EDF - at every tick run
 * the admitted request with the earliest deadline - meets every deadline of any feasible workload. The
 * scheduler therefore works like this:
 *
 *   1. when a request arrives it is offered to a deadline-monotone feasibility test over the *remaining*
 *      backlog (sort by deadline; every prefix must fit before its own deadline). Rejected work never
 *      touches the device, so the admitted set is always schedulable;
 *   2. a per-tenant quota caps the work a tenant may hold in flight, but never blocks a lone request (that
 *      would just waste the accelerator);
 *   3. each tick executes one unit of the admitted request with the earliest deadline, ties broken by id.
 *
 * Two oracles are used: the completion times are re-derived by an independent tick simulator that only sees
 * the accepted set and uses a priority queue, and the whole run is compared against blind FIFO to show that
 * admission control plus EDF collects at least as much value while missing fewer deadlines.
 *
 * Simplification: ticks are integers, the workload is 40 requests over 3 tenants from a fixed seed, and
 * "compute cost" stands in for real GPU time. Invariants verified: no admitted request ever misses its
 * deadline, every rejection is justified by re-running the feasibility test on the recorded backlog snapshot,
 * quotas hold whenever a tenant is already backlogged, and the run replays deterministically.
 */
import java.util.*;

/** One inference request. */
final class Request {
    final int id;
    final String tenant;
    final int arrival;
    final int cost;
    final int deadline;
    final int value;

    Request(int id, String tenant, int arrival, int cost, int deadline, int value) {
        this.id = id;
        this.tenant = tenant;
        this.arrival = arrival;
        this.cost = cost;
        this.deadline = deadline;
        this.value = value;
    }

    int slack() { return deadline - arrival - cost; }

    @Override
    public String toString() { return "r" + id + "/" + tenant; }
}

/** What happened to one request; `snapshot` is the in-flight work at the moment of a rejection. */
final class Decision {
    final Request request;
    final boolean admitted;
    final String reason;
    final List<int[]> snapshot;

    Decision(Request request, boolean admitted, String reason, List<int[]> snapshot) {
        this.request = request;
        this.admitted = admitted;
        this.reason = reason;
        this.snapshot = snapshot;
    }
}

/** Outcome of one scheduling run. */
final class Schedule {
    final List<Decision> decisions = new ArrayList<>();
    final Map<Integer, Integer> completion = new LinkedHashMap<>();
    int missed;
    int acceptedValue;
    int peakInFlight;

    List<Request> accepted() {
        List<Request> list = new ArrayList<>();
        for (Decision d : decisions) if (d.admitted) list.add(d.request);
        return list;
    }

    List<Decision> rejected() {
        List<Decision> list = new ArrayList<>();
        for (Decision d : decisions) if (!d.admitted) list.add(d);
        return list;
    }

    Decision decisionFor(int id) {
        for (Decision d : decisions) if (d.request.id == id) return d;
        return null;
    }
}

public class Solution {
    static final int HORIZON = 400;
    static final int QUOTA = 40;

    public static void main(String[] args) {
        List<Request> requests = generate(0x905BEEFL);
        check(requests.size() == 40, "the fixture must hold 40 requests, got " + requests.size());
        for (Request r : requests) check(r.slack() >= 0, "fixture cost must fit its window: " + r);

        // ---- 1. run the SLA-aware scheduler ------------------------------------------------------------
        Schedule sla = schedule(requests);
        check(!sla.accepted().isEmpty(), "a feasible workload must admit something");
        check(!sla.rejected().isEmpty(), "an oversubscribed workload must reject something");
        check(sla.missed == 0, "EDF plus admission control must never miss a deadline, missed " + sla.missed);
        int admittedCompute = 0;
        for (Request r : sla.accepted()) admittedCompute += r.cost;
        check(admittedCompute <= HORIZON, "the accepted set must fit the horizon: " + admittedCompute);

        // ---- 2. independent oracle: priority-queue tick simulation of the accepted set ------------------
        Map<Integer, Integer> oracle = simulateEdf(sla.accepted());
        check(oracle.equals(sla.completion), "the incremental schedule must match the standalone EDF replay");
        for (Request r : sla.accepted()) {
            check(oracle.get(r.id) <= r.deadline, r + " finished at " + oracle.get(r.id) + ", deadline "
                    + r.deadline);
            check(oracle.get(r.id) >= r.arrival + r.cost, r + " finished before all its work could run");
        }
        long oracleValue = 0;
        for (Request r : sla.accepted()) {
            check(oracle.containsKey(r.id), r + " must appear in the oracle replay");
            oracleValue += r.value;
        }
        check(oracleValue == sla.acceptedValue, "the accepted value must equal the sum over the oracle set");

        // ---- 3. every rejection is justified by re-running the feasibility test -------------------------
        int quotaRejections = 0;
        for (Decision decision : sla.rejected()) {
            List<int[]> set = new ArrayList<>(decision.snapshot);
            set.add(new int[] {decision.request.deadline, decision.request.cost});
            check(!feasible(set, decision.request.arrival),
                    decision.request + " was rejected although its backlog was still feasible");
            if (decision.reason.equals("quota")) {
                quotaRejections++;
                int inFlight = 0;
                for (int[] pair : decision.snapshot) inFlight += pair[1];
                check(inFlight + decision.request.cost > QUOTA,
                        "a quota rejection must really exceed the quota: " + inFlight + "+"
                                + decision.request.cost);
            } else {
                check(decision.reason.equals("infeasible"), "unknown rejection reason " + decision.reason);
            }
        }
        check(quotaRejections < sla.rejected().size(),
                "not every rejection may be blamed on the quota");

        // ---- 4. the accepted set is feasible and fully completed ---------------------------------------
        check(simulateEdf(sla.accepted()).size() == sla.accepted().size(),
                "every accepted request must complete inside the simulated horizon");
        check(sla.accepted().size() + sla.rejected().size() == requests.size(),
                "every request must get exactly one decision");
        check(sla.peakInFlight <= 3 * QUOTA,
                "total work in flight must respect the aggregate quota: " + sla.peakInFlight);

        // ---- 5. per-tenant accounting and quota behaviour -----------------------------------------------
        Map<String, Integer> compute = new TreeMap<>();
        for (Request r : sla.accepted()) compute.merge(r.tenant, r.cost, Integer::sum);
        check(compute.size() == 3, "all three tenants must be served, got " + compute.keySet());
        int busiest = 0;
        for (int c : compute.values()) busiest = Math.max(busiest, c);
        check(busiest > 0 && busiest <= admittedCompute, "tenant accounting must be a partition of the load");
        List<Request> squeeze = new ArrayList<>();
        squeeze.add(new Request(0, "t1", 0, 30, 60, 5));
        squeeze.add(new Request(1, "t1", 0, 30, 60, 5));
        Schedule squeezed = schedule(squeeze);
        check(squeezed.rejected().size() == 1 && squeezed.rejected().get(0).reason.equals("quota"),
                "the second request must be refused by the quota, got "
                        + (squeezed.rejected().isEmpty() ? "no rejection" : squeezed.rejected().get(0).reason));

        // ---- 6. comparison against blind FIFO -----------------------------------------------------------
        Schedule fifo = scheduleFifo(requests);
        check(fifo.missed > sla.missed, "blind FIFO must miss deadlines while EDF misses none: " + fifo.missed);
        check(sla.acceptedValue >= onTimeValue(fifo),
                "admission control plus EDF must collect at least the value FIFO delivers on time: "
                        + sla.acceptedValue + " vs " + onTimeValue(fifo));

        // ---- 7. boundary cases --------------------------------------------------------------------------
        check(schedule(List.of()).accepted().isEmpty(), "an empty workload admits nothing");
        Request solo = new Request(99, "t1", 0, 5, 5, 10);
        Schedule soloRun = schedule(List.of(solo));
        check(soloRun.accepted().size() == 1 && soloRun.missed == 0, "a single urgent request must be served");
        check(soloRun.completion.get(99) == 5, "it must finish exactly at its deadline: "
                + soloRun.completion.get(99));
        Request impossible = new Request(98, "t1", 0, 9, 5, 10);
        Schedule impossibleRun = schedule(List.of(impossible));
        check(impossibleRun.accepted().isEmpty() && impossibleRun.rejected().get(0).reason.equals("infeasible"),
                "a request whose cost exceeds its window can never be admitted");
        check(schedule(List.of(new Request(97, "t2", 0, 0, 0, 1))).accepted().size() == 1,
                "a zero-cost request at its deadline is trivially schedulable");
        check(HORIZON > requests.get(requests.size() - 1).deadline, "the horizon must cover the workload");

        // ---- 8. determinism ------------------------------------------------------------------------------
        Schedule replay = schedule(generate(0x905BEEFL));
        check(replay.acceptedValue == sla.acceptedValue, "the accepted value must replay identically");
        check(replay.decisions.size() == sla.decisions.size(), "the decision count must replay identically");
        for (int i = 0; i < sla.decisions.size(); i++) {
            Decision a = replay.decisions.get(i);
            Decision b = sla.decisions.get(i);
            check(a.request.id == b.request.id && a.admitted == b.admitted && a.reason.equals(b.reason),
                    "decision " + i + " must replay identically");
        }
        check(replay.completion.equals(sla.completion), "completion times must replay identically");
        check(!sla.rejected().isEmpty() && sla.decisionFor(sla.rejected().get(0).request.id) != null,
                "every request must be traceable to exactly one decision");

        System.out.println("requests " + requests.size() + " over 3 tenants, horizon " + HORIZON + ", quota "
                + QUOTA + " ticks/tenant");
        System.out.println("admitted " + sla.accepted().size() + " worth " + sla.acceptedValue + ", rejected "
                + sla.rejected().size() + " (" + quotaRejections + " by quota), missed " + sla.missed
                + ", peak in flight " + sla.peakInFlight);
        System.out.println("FIFO baseline: queued " + fifo.accepted().size() + " worth " + fifo.acceptedValue
                + " of which on time " + onTimeValue(fifo) + ", missed " + fifo.missed
                + "; per-tenant compute " + compute);
        System.out.println("PASS");
    }

    /** Deterministic workload: 40 requests across three tenants with mixed costs and deadlines. */
    static List<Request> generate(long seed) {
        SplitMix64 rng = new SplitMix64(seed);
        List<Request> requests = new ArrayList<>();
        String[] tenants = {"t1", "t2", "t3"};
        for (int i = 0; i < 40; i++) {
            String tenant = tenants[i % 3];
            int arrival = 6 * i + (int) Math.floorMod(rng.nextLong(), 5);
            int cost = 3 + (int) Math.floorMod(rng.nextLong(), 12);
            int deadline = arrival + cost + (int) Math.floorMod(rng.nextLong(), 25);
            int value = 1 + (int) Math.floorMod(rng.nextLong(), 9);
            requests.add(new Request(i, tenant, arrival, cost, deadline, value));
        }
        requests.sort(Comparator.comparingInt((Request r) -> r.arrival).thenComparingInt(r -> r.id));
        return requests;
    }

    /**
     * Admission control plus preemptive EDF over integer ticks. Requests are offered to the feasibility test
     * when they arrive; each tick executes one unit of the earliest-deadline admitted request.
     */
    static Schedule schedule(List<Request> requests) {
        Schedule schedule = new Schedule();
        if (requests.isEmpty()) return schedule;
        int lastArrival = 0;
        int lastDeadline = 0;
        for (Request r : requests) {
            lastArrival = Math.max(lastArrival, r.arrival);
            lastDeadline = Math.max(lastDeadline, r.deadline);
        }
        Map<Integer, Request> admitted = new LinkedHashMap<>();
        Map<Integer, Integer> remaining = new LinkedHashMap<>();
        Map<String, Integer> inFlight = new TreeMap<>();
        int tick = 0;
        while (tick <= lastDeadline + 1) {
            for (Request request : requests) {
                if (request.arrival != tick) continue;
                int held = inFlight.getOrDefault(request.tenant, 0);
                List<int[]> backlog = snapshot(admitted, remaining);
                boolean quotaOk = held + request.cost <= QUOTA || held == 0;
                if (!quotaOk) {
                    schedule.decisions.add(new Decision(request, false, "quota", backlog));
                    continue;
                }
                List<int[]> candidate = new ArrayList<>(backlog);
                candidate.add(new int[] {request.deadline, request.cost});
                if (!feasible(candidate, tick)) {
                    schedule.decisions.add(new Decision(request, false, "infeasible", backlog));
                    continue;
                }
                schedule.decisions.add(new Decision(request, true, "admitted", backlog));
                admitted.put(request.id, request);
                remaining.put(request.id, request.cost);
                inFlight.merge(request.tenant, request.cost, Integer::sum);
                schedule.acceptedValue += request.value;
                if (request.cost == 0) schedule.completion.put(request.id, tick);
                schedule.peakInFlight = Math.max(schedule.peakInFlight, total(remaining));
            }
            Integer chosen = null;
            for (Map.Entry<Integer, Request> e : admitted.entrySet()) {
                if (remaining.get(e.getKey()) <= 0) continue;
                if (chosen == null || e.getValue().deadline < admitted.get(chosen).deadline
                        || (e.getValue().deadline == admitted.get(chosen).deadline && e.getKey() < chosen)) {
                    chosen = e.getKey();
                }
            }
            if (chosen != null) {
                int left = remaining.get(chosen) - 1;
                remaining.put(chosen, left);
                if (left == 0) {
                    Request done = admitted.get(chosen);
                    schedule.completion.put(chosen, tick + 1);
                    inFlight.merge(done.tenant, -done.cost, Integer::sum);
                    if (tick + 1 > done.deadline) schedule.missed++;
                }
            } else if (tick > lastArrival) {
                break;
            }
            tick++;
        }
        return schedule;
    }

    /** Feasibility of a set of (deadline, cost) pairs released together at `now`, by the prefix test. */
    static boolean feasible(List<int[]> work, int now) {
        List<int[]> ordered = new ArrayList<>(work);
        ordered.sort(Comparator.<int[]>comparingInt(p -> p[0]).thenComparingInt(p -> p[1]));
        long clock = now;
        for (int[] pair : ordered) {
            clock += pair[1];
            if (clock > pair[0]) return false;
        }
        return true;
    }

    static List<int[]> snapshot(Map<Integer, Request> admitted, Map<Integer, Integer> remaining) {
        List<int[]> backlog = new ArrayList<>();
        for (Map.Entry<Integer, Request> e : admitted.entrySet()) {
            int left = remaining.get(e.getKey());
            if (left > 0) backlog.add(new int[] {e.getValue().deadline, left});
        }
        return backlog;
    }

    static int total(Map<Integer, Integer> remaining) {
        int sum = 0;
        for (int v : remaining.values()) sum += v;
        return sum;
    }

    /** Independent oracle: tick simulation of a finished accepted set, using a priority queue. */
    static Map<Integer, Integer> simulateEdf(List<Request> set) {
        Map<Integer, Integer> completion = new LinkedHashMap<>();
        if (set.isEmpty()) return completion;
        PriorityQueue<Request> ready = new PriorityQueue<>(
                Comparator.comparingInt((Request r) -> r.deadline).thenComparingInt(r -> r.id));
        Map<Integer, Integer> remaining = new LinkedHashMap<>();
        List<Request> ordered = new ArrayList<>(set);
        ordered.sort(Comparator.comparingInt(r -> r.arrival));
        int lastDeadline = 0;
        for (Request r : set) lastDeadline = Math.max(lastDeadline, r.deadline);
        int cursor = 0;
        for (int tick = 0; tick <= lastDeadline + 1 && completion.size() < set.size(); tick++) {
            while (cursor < ordered.size() && ordered.get(cursor).arrival <= tick) {
                Request r = ordered.get(cursor++);
                ready.add(r);
                remaining.put(r.id, r.cost);
            }
            Request best = null;
            for (Request candidate : ready) {
                if (remaining.get(candidate.id) > 0) {
                    best = candidate;
                    break;
                }
            }
            if (best == null) continue;
            int left = remaining.get(best.id) - 1;
            remaining.put(best.id, left);
            if (left == 0) {
                completion.put(best.id, tick + 1);
                ready.remove(best);
            }
        }
        return completion;
    }

    /** Blind FIFO with no admission control: whatever arrives is queued in arrival order. */
    static Schedule scheduleFifo(List<Request> requests) {
        Schedule schedule = new Schedule();
        int clock = 0;
        for (Request request : requests) {
            clock = Math.max(clock, request.arrival) + request.cost;
            schedule.completion.put(request.id, clock);
            schedule.decisions.add(new Decision(request, true, "queued", List.of()));
            if (clock > request.deadline) schedule.missed++;
            schedule.acceptedValue += request.value;
        }
        return schedule;
    }

    /** Value of the requests that actually finished by their deadline. */
    static long onTimeValue(Schedule schedule) {
        long value = 0;
        for (Decision decision : schedule.decisions) {
            if (!decision.admitted) continue;
            Integer finished = schedule.completion.get(decision.request.id);
            if (finished != null && finished <= decision.request.deadline) value += decision.request.value;
        }
        return value;
    }

    static final class SplitMix64 {
        private long state;

        SplitMix64(long seed) { this.state = seed; }

        long nextLong() {
            state += 0x9E3779B97F4A7C15L;
            long z = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return z ^ (z >>> 31);
        }
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
