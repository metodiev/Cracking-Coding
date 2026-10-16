/** Task 969 - Implement curriculum mapping to academic topics and pacing suggestions.
 *  (Tier: Master, CTCI Beyond -- Software Engineering & Tooling)
 *
 *  A curriculum mapper. Problems are tagged with topic codes, topics carry a study cost and may require
 *  other topics. The mapper (a) indexes problems per topic and reports problems nobody can place,
 *  (b) topologically orders topics so nothing is scheduled before its prerequisites (cycles rejected),
 *  (c) packs topics into weeks of bounded capacity, placing each topic in the earliest week that starts
 *  after every prerequisite and still has room, so week(topic) > week(prereq) always holds, and
 *  (d) emits pacing suggestions per week: practice problems for that week's topics plus one review
 *  problem drawn from topics already covered.
 *  Simplified: fixed per-topic cost instead of per-learner estimates and a first-fit packer instead of an
 *  optimiser; the prerequisite invariant, the capacity invariant and coverage accounting are the real ones.
 *  Verified: complete mapping, unmapped-problem detection, prerequisite respect, weekly load recomputed
 *  independently, cycle rejection, suggestion provenance (rogue problems never suggested), determinism
 *  under re-declaration and the empty-curriculum boundary.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final class Topic {
        final String code; final int minutes; final List<String> prereqs;
        Topic(String code, int minutes, String... prereqs) {
            this.code = code; this.minutes = minutes; this.prereqs = List.of(prereqs);
        }
    }
    static final class Problem {
        final String id; final int difficulty; final List<String> topics;
        Problem(String id, int difficulty, String... topics) {
            this.id = id; this.difficulty = difficulty; this.topics = List.of(topics);
        }
    }
    static final class Curriculum {
        final List<Topic> topics; final List<Problem> problems;
        Curriculum(List<Topic> topics, List<Problem> problems) { this.topics = topics; this.problems = problems; }
    }

    /** topic code -> problem ids in declaration order; ids whose topics are all unknown go to `unmapped`. */
    static Map<String, List<String>> index(Curriculum c, List<String> unmapped) {
        Map<String, List<String>> byTopic = new TreeMap<>();
        for (Topic t : c.topics) byTopic.put(t.code, new ArrayList<>());
        for (Problem p : c.problems) {
            boolean placed = false;
            for (String code : p.topics) {
                if (byTopic.containsKey(code)) { byTopic.get(code).add(p.id); placed = true; }
            }
            if (!placed) unmapped.add(p.id);
        }
        return byTopic;
    }

    /** Kahn topological order of the topics; an unknown prerequisite or a cycle is rejected. */
    static List<String> order(Curriculum c) {
        Map<String, List<String>> after = new TreeMap<>();
        Map<String, Integer> indegree = new TreeMap<>();
        for (Topic t : c.topics) { after.put(t.code, new ArrayList<>()); indegree.put(t.code, 0); }
        for (Topic t : c.topics) {
            for (String pre : t.prereqs) {
                if (!after.containsKey(pre)) throw new IllegalArgumentException("unknown prerequisite " + pre);
                after.get(pre).add(t.code);
                indegree.merge(t.code, 1, Integer::sum);
            }
        }
        PriorityQueue<String> ready = new PriorityQueue<>();                  // lexicographic => deterministic
        for (Map.Entry<String, Integer> e : indegree.entrySet()) if (e.getValue() == 0) ready.add(e.getKey());
        List<String> out = new ArrayList<>();
        while (!ready.isEmpty()) {
            String code = ready.poll();
            out.add(code);
            for (String next : after.get(code)) if (indegree.merge(next, -1, Integer::sum) == 0) ready.add(next);
        }
        if (out.size() != c.topics.size()) throw new IllegalStateException("prerequisite cycle in curriculum");
        return out;
    }

    /** topic code -> 1-based week: first week at or after (max prereq week + 1) that still has room. */
    static Map<String, Integer> pace(Curriculum c, List<String> order, int capacity) {
        Map<String, Topic> byCode = new TreeMap<>();
        for (Topic t : c.topics) byCode.put(t.code, t);
        Map<String, Integer> week = new TreeMap<>();
        List<Integer> load = new ArrayList<>();
        for (String code : order) {
            Topic t = byCode.get(code);
            int earliest = 1;
            for (String pre : t.prereqs) earliest = Math.max(earliest, week.get(pre) + 1);
            int w = earliest - 1;
            while (true) {
                if (w == load.size()) load.add(0);
                if (load.get(w) == 0 || load.get(w) + t.minutes <= capacity) break;
                w++;
            }
            load.set(w, load.get(w) + t.minutes);
            week.put(code, w + 1);
        }
        return week;
    }

    /** One week of pacing: practice problems for this week's topics plus one earlier-topic review. */
    static final class WeekPlan {
        final int week; final List<String> practice = new ArrayList<>(); final List<String> review = new ArrayList<>();
        WeekPlan(int week) { this.week = week; }
        List<String> all() { List<String> out = new ArrayList<>(practice); out.addAll(review); return out; }
    }

    static List<WeekPlan> suggestions(Curriculum c, List<String> order, Map<String, Integer> week,
                                      int slots, int weeks) {
        Map<Integer, List<String>> topicsOfWeek = new TreeMap<>();
        for (String code : order) topicsOfWeek.computeIfAbsent(week.get(code), k -> new ArrayList<>()).add(code);
        List<WeekPlan> out = new ArrayList<>();
        for (int w = 1; w <= weeks; w++) {
            WeekPlan plan = new WeekPlan(w);
            for (Problem p : c.problems) {
                if (plan.practice.size() == slots) break;
                for (String code : p.topics) if (topicsOfWeek.getOrDefault(w, List.of()).contains(code)) {
                    plan.practice.add(p.id);
                    break;
                }
            }
            for (Problem p : c.problems) {
                if (plan.practice.contains(p.id)) continue;
                for (String code : p.topics) {
                    Integer wk = week.get(code);                              // unknown topics are never scheduled
                    if (wk != null && wk < w) { plan.review.add(p.id); break; }
                }
                if (!plan.review.isEmpty()) break;
            }
            out.add(plan);
        }
        return out;
    }

    static Curriculum curriculum() {
        return new Curriculum(
                List.of(new Topic("arrays", 90), new Topic("sorting", 60, "arrays"),
                        new Topic("hashing", 60, "arrays"), new Topic("graphs", 120, "hashing", "sorting"),
                        new Topic("dp", 150, "graphs")),
                List.of(new Problem("two-sum", 1, "arrays", "hashing"), new Problem("sort-colors", 2, "sorting"),
                        new Problem("bfs-maze", 3, "graphs"), new Problem("coin-change", 4, "dp"),
                        new Problem("topo-sort", 3, "graphs", "sorting"), new Problem("rogue", 1, "unlisted")));
    }

    public static void main(String[] args) {
        Curriculum c = curriculum();
        List<String> unmapped = new ArrayList<>();
        Map<String, List<String>> byTopic = index(c, unmapped);
        Map<String, List<String>> expected = new TreeMap<>(Map.of(
                "arrays", List.of("two-sum"), "sorting", List.of("sort-colors", "topo-sort"),
                "hashing", List.of("two-sum"), "graphs", List.of("bfs-maze", "topo-sort"),
                "dp", List.of("coin-change")));
        check(byTopic.equals(expected) && unmapped.equals(List.of("rogue")),
                "problems must map under every known topic and unplaceable ones be reported: " + byTopic);

        List<String> order = order(c);
        boolean prereqsFirst = true;
        for (Topic t : c.topics) for (String pre : t.prereqs) prereqsFirst &= order.indexOf(pre) < order.indexOf(t.code);
        check(prereqsFirst && order.size() == c.topics.size() && order.get(0).equals("arrays"),
                "the topological order must place every prerequisite first: " + order);

        Curriculum cyclic = new Curriculum(List.of(new Topic("a", 10, "b"), new Topic("b", 10, "a")), List.of());
        try {
            order(cyclic);
            check(false, "a prerequisite cycle must be rejected");
        } catch (IllegalStateException e) {
            check(e.getMessage().contains("cycle"), "cycle rejection must explain itself: " + e.getMessage());
        }
        try {
            order(new Curriculum(List.of(new Topic("a", 10, "ghost")), List.of()));
            check(false, "an unknown prerequisite must be rejected");
        } catch (IllegalArgumentException e) {
            check(e.getMessage().contains("ghost"), "unknown prerequisite must be named: " + e.getMessage());
        }

        int capacity = 200;
        Map<String, Integer> week = pace(c, order, capacity);
        Map<Integer, Integer> load = new TreeMap<>();
        for (Topic t : c.topics) load.merge(week.get(t.code), t.minutes, Integer::sum);
        check(load.values().stream().allMatch(m -> m <= capacity)
                        && load.values().stream().mapToInt(Integer::intValue).sum()
                        == c.topics.stream().mapToInt(t -> t.minutes).sum()
                        && week.size() == c.topics.size() && load.size() == 4,
                "every topic is scheduled once and no week exceeds capacity: load=" + load + " week=" + week);
        boolean afterPrereqs = true;
        for (Topic t : c.topics) for (String pre : t.prereqs) afterPrereqs &= week.get(t.code) > week.get(pre);
        check(afterPrereqs && week.get("graphs") == 3 && week.get("dp") == 4,
                "a topic may only start in a week after all of its prerequisites: " + week);

        List<WeekPlan> plan = suggestions(c, order, week, 2, load.size());
        boolean provenance = plan.size() == load.size() && plan.stream().allMatch(p -> p.practice.size() <= 2);
        for (WeekPlan p : plan) {
            for (String id : p.practice) provenance &= practiceOfWeek(c, week, id, p.week);
            for (String id : p.review) provenance &= reviewOfEarlierWeek(c, week, id, p.week);
            provenance &= Collections.disjoint(p.practice, p.review);
        }
        check(provenance && plan.get(0).all().equals(List.of("two-sum"))
                        && plan.get(1).all().equals(List.of("two-sum", "sort-colors"))
                        && plan.get(2).all().equals(List.of("bfs-maze", "topo-sort", "two-sum"))
                        && plan.get(3).all().equals(List.of("coin-change", "two-sum")),
                "practice must match the week's topics and review an earlier week's topic: " + plan.get(3).all());

        List<String> allSuggested = new ArrayList<>();
        for (WeekPlan p : plan) allSuggested.addAll(p.all());
        check(!allSuggested.contains("rogue") && allSuggested.size() >= 6
                        && plan.get(2).review.size() == 1 && plan.get(3).review.size() == 1,
                "problems with only unknown topics must never be suggested: " + allSuggested);

        List<Topic> reversed = new ArrayList<>(c.topics);
        Collections.reverse(reversed);
        Curriculum shuffled = new Curriculum(reversed, c.problems);
        check(order(shuffled).equals(order) && pace(shuffled, order(shuffled), capacity).equals(week)
                        && suggestions(shuffled, order, week, 2, load.size()).get(3).all().equals(plan.get(3).all()),
                "declaring the same curriculum in another order must pace identically");
        Curriculum empty = new Curriculum(List.of(), List.of());
        check(order(empty).isEmpty() && pace(empty, List.of(), 100).isEmpty()
                        && suggestions(empty, List.of(), Map.of(), 2, 0).isEmpty(),
                "an empty curriculum must yield no weeks and no suggestions");

        System.out.println("topics=" + c.topics.size() + " problems=" + c.problems.size() + " weeks=" + load.size());
        System.out.println("load=" + load + " order=" + order);
        System.out.println("PASS");
    }

    /** True when `id` is tagged with a topic scheduled in exactly week `w`. */
    static boolean practiceOfWeek(Curriculum c, Map<String, Integer> week, String id, int w) {
        for (Problem p : c.problems)
            if (p.id.equals(id))
                for (String code : p.topics) if (Integer.valueOf(w).equals(week.get(code))) return true;
        return false;
    }

    /** True when `id` is tagged with a topic already covered strictly before week `w`. */
    static boolean reviewOfEarlierWeek(Curriculum c, Map<String, Integer> week, String id, int w) {
        for (Problem p : c.problems)
            if (p.id.equals(id))
                for (String code : p.topics) {
                    Integer wk = week.get(code);
                    if (wk != null && wk < w) return true;
                }
        return false;
    }
}
