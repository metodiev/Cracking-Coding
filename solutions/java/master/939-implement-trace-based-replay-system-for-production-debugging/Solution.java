/** Task 939 - Implement trace-based replay system for production debugging.
 *  (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 *  A distributed-trace debugger: spans (id, parent, service, recorded self time) are validated, linked
 *  into a call tree and laid out as back-to-back nested intervals - a parent's duration is its own self
 *  time plus the slowest child - then queried for the hottest chain (the classic critical path, which for
 *  this layout must equal the root's duration), for per-service self-time attribution and for which spans
 *  were live at any logical instant (time travel). Replay re-derives the whole schedule bottom-up from
 *  the recorded nondeterminism log (one self time per span) and reports the first divergence in
 *  post-order, which is how a record/replay debugger pinpoints where production and its repro disagree.
 *  Simplified: no clock skew, no async spans, no cross-service ordering; validation, the critical-path
 *  derivation and the divergence reporting are the real ones.
 *  Verified: an intact log reproduces the trace, a mutated or missing decision diverges at the right span,
 *  the critical path matches exhaustive path enumeration, time travel matches a binary-search sweep, self
 *  times are conserved, and malformed traces are rejected with a specific reason.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }
    static final class Node {
        final String id, parent, service;
        final int selfTime;
        final List<Node> children = new ArrayList<>();
        int duration, start;
        Node(String id, String parent, String service, int selfTime) {
            this.id = id; this.parent = parent; this.service = service; this.selfTime = selfTime;
        }
    }
    /** id, parent, service, recorded self time - the production trace under debug. */
    static final String[][] SPEC = {
        {"root", "", "gateway", "8"}, {"auth", "root", "auth", "12"}, {"cart", "root", "cart", "6"},
        {"cartdb", "cart", "cartdb", "20"}, {"pay", "root", "payments", "10"},
        {"pspauth", "pay", "psp", "35"}, {"paydb", "pay", "paymentsdb", "15"},
        {"render", "root", "web", "7"}, {"tmpl", "render", "web", "18"}, {"cdn", "render", "cdn", "9"},
    };

    static List<Node> fixture() {
        List<Node> nodes = new ArrayList<>();
        for (String[] s : SPEC) nodes.add(new Node(s[0], s[1], s[2], Integer.parseInt(s[3])));
        return nodes;
    }
    static List<Node> malformed(String kind) {
        List<Node> nodes = fixture();
        switch (kind) {
            case "orphan" -> nodes.remove(2);                                        // drops cart, keeps cartdb
            case "duplicate" -> nodes.add(new Node("cartdb", "cart", "cartdb", 4));
            case "cycle" -> nodes.set(2, new Node("cart", "cartdb", "cart", 6));     // cart <-> cartdb
            case "negative" -> nodes.set(6, new Node("paydb", "pay", "paymentsdb", -3));
            default -> throw new IllegalStateException(kind);
        }
        return nodes;
    }
    static Node find(List<Node> nodes, String id) {
        for (Node n : nodes) if (n.id.equals(id)) return n;
        return null;
    }
    /** "" when the trace is well formed, otherwise the reason a debugger would report. */
    static String validate(List<Node> nodes) {
        if (nodes.isEmpty()) return "empty-trace";
        Set<String> ids = new TreeSet<>();
        for (Node n : nodes) {
            if (n.id.isEmpty()) return "empty-id";
            if (!ids.add(n.id)) return "duplicate-id " + n.id;
            if (n.selfTime < 0) return "negative-duration " + n.id;
        }
        int roots = 0;
        for (Node n : nodes) {
            if (n.parent.isEmpty()) roots++;
            else if (!ids.contains(n.parent)) return "orphan-span " + n.id;
        }
        if (roots != 1) return "root-count " + roots;
        for (Node n : nodes) {                                    // every parent chain must terminate at the root
            Set<String> seen = new HashSet<>();
            String cursor = n.parent;
            while (!cursor.isEmpty()) {
                if (cursor.equals(n.id) || !seen.add(cursor)) return "cycle " + n.id;
                cursor = find(nodes, cursor).parent;
            }
        }
        return "";
    }
    static final class Trace {
        final List<Node> all;
        final List<Node> pre = new ArrayList<>();                 // parents before children
        final List<Node> post = new ArrayList<>();                // the replay comparison order
        final String problem;
        Node root;

        Trace(List<Node> nodes) {
            all = nodes;
            problem = validate(nodes);
            if (!problem.isEmpty()) return;
            for (Node n : nodes) {
                if (n.parent.isEmpty()) root = n;
                else find(nodes, n.parent).children.add(n);
            }
            computeDuration(root);
            place(root);
            walk(root);
        }
        int computeDuration(Node n) {
            int slowest = 0;
            for (Node c : n.children) slowest = Math.max(slowest, computeDuration(c));
            n.duration = n.selfTime + slowest;                    // nested spans end with the slowest child
            return n.duration;
        }
        void place(Node n) {
            for (Node c : n.children) {
                c.start = n.start + n.duration - c.duration;
                place(c);
            }
        }
        void walk(Node n) {
            pre.add(n);
            for (Node c : n.children) walk(c);
            post.add(n);
        }
        /** Hottest chain: the root-to-leaf path with the largest total self time (memo-free DP). */
        int hotness(Node n) {
            int best = 0;
            for (Node c : n.children) best = Math.max(best, hotness(c));
            return n.selfTime + best;
        }
        List<String> criticalPath() {
            List<String> path = new ArrayList<>();
            for (Node n = root; n != null; n = hottestChild(n)) path.add(n.id);
            return path;
        }
        Node hottestChild(Node n) {
            Node best = null;
            for (Node c : n.children) if (best == null || hotness(c) > hotness(best)) best = c;
            return best;
        }
    }
    /** Oracle: enumerate every root-to-leaf path and keep its total self time. */
    static void enumeratePaths(Node n, int carried, List<Integer> totals) {
        int total = carried + n.selfTime;
        if (n.children.isEmpty()) {
            totals.add(total);
            return;
        }
        for (Node c : n.children) enumeratePaths(c, total, totals);
    }
    /** Independent time-travel oracle: binary search over sorted starts and ends. */
    static int coveringBySweep(List<Node> nodes, int t) {
        int[] starts = new int[nodes.size()], ends = new int[nodes.size()];
        for (int i = 0; i < nodes.size(); i++) {
            starts[i] = nodes.get(i).start;
            ends[i] = nodes.get(i).start + nodes.get(i).duration;
        }
        Arrays.sort(starts);
        Arrays.sort(ends);
        return upperBound(starts, t) - upperBound(ends, t);
    }
    static int upperBound(int[] sorted, int value) {
        int lo = 0, hi = sorted.length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (sorted[mid] <= value) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }
    static List<String> coveringNaive(List<Node> nodes, int t) {
        List<String> ids = new ArrayList<>();
        for (Node n : nodes) if (t >= n.start && t < n.start + n.duration) ids.add(n.id);
        return ids;
    }
    static final class ReplayResult {
        final String status, span, expected, actual;
        ReplayResult(String status, String span, String expected, String actual) {
            this.status = status; this.span = span; this.expected = expected; this.actual = actual;
        }
    }
    /** Re-derive durations bottom-up from the recorded self times, reporting the first divergence. */
    static ReplayResult replay(Trace trace, Map<String, Integer> decisions) {
        Map<String, Integer> derived = new HashMap<>();
        for (Node n : trace.post) {                               // post-order: children first
            Integer recorded = decisions.get(n.id);
            if (recorded == null) return new ReplayResult("missing-decision", n.id, "" + n.selfTime, "none");
            if (recorded < 0) return new ReplayResult("invalid-decision", n.id, "" + n.selfTime, "" + recorded);
            int slowest = 0;
            for (Node c : n.children) slowest = Math.max(slowest, derived.get(c.id));
            int duration = recorded + slowest;
            derived.put(n.id, duration);
            if (duration != n.duration) return new ReplayResult("diverged", n.id, "" + n.duration, "" + duration);
        }
        return new ReplayResult("reproduced", "", "", "");
    }
    static Map<String, Integer> log(Trace trace) {
        Map<String, Integer> decisions = new LinkedHashMap<>();
        for (Node n : trace.all) decisions.put(n.id, n.selfTime);
        return decisions;
    }
    public static void main(String[] args) {
        Trace trace = new Trace(fixture());
        check(trace.problem.isEmpty() && trace.all.size() == 10 && trace.post.size() == 10
                        && trace.root.id.equals("root") && trace.root.children.size() == 4,
                "the fixture trace must ingest as a single-rooted tree: \"" + trace.problem + "\"");

        List<Integer> totals = new ArrayList<>();
        enumeratePaths(trace.root, 0, totals);
        check(totals.size() == 6 && trace.hotness(trace.root) == Collections.max(totals)
                        && trace.hotness(trace.root) == trace.root.duration
                        && trace.criticalPath().equals(List.of("root", "pay", "pspauth")),
                "the critical path must match exhaustive enumeration and the root duration: "
                        + trace.criticalPath() + " " + trace.hotness(trace.root) + " vs " + Collections.max(totals)
                        + " over " + totals.size() + " paths");

        int selfSum = 0, leafViolations = 0;
        for (Node n : trace.all) {
            selfSum += n.selfTime;
            if (n.children.isEmpty() && n.duration != n.selfTime) leafViolations++;
        }
        check(leafViolations == 0 && selfSum == 140 && trace.root.start == 0 && trace.root.duration == 53
                        && selfSum >= trace.root.duration,
                "self times must be conserved and lay out a nested schedule: selfSum " + selfSum + ", root "
                        + trace.root.duration);

        int travelViolations = 0;
        for (int t : new int[] {-1, 0, 7, 8, 18, 40, 52, 53, 54}) {
            List<String> naive = coveringNaive(trace.all, t);
            if (naive.size() != coveringBySweep(trace.all, t)) travelViolations++;
            if (t >= 0 && t < trace.root.duration && !naive.contains("root")) travelViolations++;
            if (t < 0 || t >= trace.root.duration) {
                if (!naive.isEmpty()) travelViolations++;
            }
        }
        check(travelViolations == 0, travelViolations + " time-travel queries disagree with the sweep oracle");
        check(coveringNaive(trace.all, 0).equals(List.of("root")) && coveringNaive(trace.all, 52).size() == 10
                        && coveringNaive(trace.all, 53).isEmpty(),
                "the trace window must be half open at its end: 0 -> " + coveringNaive(trace.all, 0)
                        + ", 52 -> " + coveringNaive(trace.all, 52).size());

        Map<String, Integer> perService = new TreeMap<>();
        for (Node n : trace.all) perService.merge(n.service, n.selfTime, Integer::sum);
        String hottest = "";
        int attributed = 0;
        for (Map.Entry<String, Integer> e : perService.entrySet()) {
            attributed += e.getValue();
            if (hottest.isEmpty() || e.getValue() > perService.get(hottest)) hottest = e.getKey();
        }
        check(attributed == selfSum && perService.size() == 9 && perService.get("web") == 25
                        && perService.get("psp") == 35 && hottest.equals("psp"),
                "per-service attribution must conserve the self time: " + attributed + " vs " + selfSum
                        + ", hottest " + hottest);

        ReplayResult intact = replay(trace, log(trace));
        Trace again = new Trace(fixture());
        check(intact.status.equals("reproduced") && replay(again, log(again)).status.equals("reproduced")
                        && again.criticalPath().equals(trace.criticalPath()),
                "an intact decision log must reproduce the trace: " + intact.status + " " + intact.span);

        Map<String, Integer> leafLog = log(trace);
        leafLog.put("pspauth", 50);
        ReplayResult leaf = replay(trace, leafLog);
        Map<String, Integer> innerLog = log(trace);
        innerLog.put("render", 5);
        ReplayResult inner = replay(trace, innerLog);
        check(leaf.status.equals("diverged") && leaf.span.equals("pspauth") && leaf.expected.equals("35")
                        && leaf.actual.equals("50") && inner.status.equals("diverged")
                        && inner.span.equals("render") && Integer.parseInt(inner.actual) < Integer.parseInt(inner.expected),
                "a mutated decision must diverge at its own span, not at an ancestor: " + leaf.span + " "
                        + leaf.expected + "->" + leaf.actual + ", " + inner.span + " " + inner.expected + "->"
                        + inner.actual);

        Map<String, Integer> gapLog = log(trace);
        gapLog.remove("cdn");
        ReplayResult gap = replay(trace, gapLog);
        Map<String, Integer> ghostLog = log(trace);
        ghostLog.put("ghost", 3);
        check(gap.status.equals("missing-decision") && gap.span.equals("cdn")
                        && replay(trace, ghostLog).status.equals("reproduced"),
                "a missing decision must stop the replay while stray decisions are ignored: " + gap.status + " "
                        + gap.span);

        String orphan = new Trace(malformed("orphan")).problem;
        String duplicate = new Trace(malformed("duplicate")).problem;
        String cycle = new Trace(malformed("cycle")).problem;
        String negative = new Trace(malformed("negative")).problem;
        check(orphan.startsWith("orphan-span") && duplicate.startsWith("duplicate-id") && cycle.startsWith("cycle")
                        && negative.startsWith("negative-duration")
                        && new Trace(new ArrayList<>()).problem.equals("empty-trace"),
                "malformed traces must be rejected with a specific reason: " + orphan + " | " + duplicate + " | "
                        + cycle + " | " + negative);

        Trace single = new Trace(List.of(new Node("only", "", "solo", 11)));
        check(single.problem.isEmpty() && single.root.duration == 11 && single.hotness(single.root) == 11
                        && coveringNaive(single.all, 10).equals(List.of("only"))
                        && coveringNaive(single.all, 11).isEmpty()
                        && replay(single, log(single)).status.equals("reproduced"),
                "a single-span trace must be consistent end to end");

        System.out.println("trace " + trace.all.size() + " spans over " + perService.size() + " services, root "
                + trace.root.duration + " ms, self time " + selfSum + " ms, critical path " + trace.criticalPath());
        System.out.println("schedule: root [" + trace.root.start + "," + (trace.root.start + trace.root.duration)
                + "), " + coveringNaive(trace.all, 52).size() + " spans live at 52, hottest service " + hottest);
        System.out.println("PASS");
    }
}
