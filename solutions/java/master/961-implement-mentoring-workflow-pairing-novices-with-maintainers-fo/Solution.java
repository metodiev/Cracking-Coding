/** Task 961 - Implement mentoring workflow pairing novices with maintainers for PR review.
 *  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 *  In-memory pairing service. Novices are ranked against maintainers by area affinity, then by the
 *  distance between their comfort levels, then by identifier so preferences are total and deterministic.
 *  Pairing uses capacitated deferred acceptance (Gale-Shapley, novices propose, maintainers hold their
 *  best `capacity` candidates and release the worst), which guarantees a stable assignment: no novice and
 *  maintainer both prefer each other over their current situation.
 *  Simplified: preferences are derived from the two fields above instead of review history, and all state
 *  is in memory; capacity, stability, determinism and the unmatched report are the real mechanics.
 *  Verified: ample capacity matches everyone; capacity is never exceeded and nobody appears twice; the
 *  assignment is stable against an independently written blocking-pair scan; shuffled input yields the
 *  same pairs; tight capacity matches only as many novices as there are slots while keeping area
 *  affinity; empty inputs and zero-capacity maintainers are handled; the matched count equals the
 *  feasible minimum of novices and total capacity.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final long FNV_OFFSET = 0xcbf29ce484222325L, FNV_PRIME = 0x100000001b3L;

    static long fnvText(long h, String s) {
        for (int i = 0; i < s.length(); i++) { h ^= s.charAt(i); h *= FNV_PRIME; }
        return h;
    }

    static String hex(long v) { return String.format("%016x", v); }

    record Novice(String id, String area, int level) {}
    record Maintainer(String id, String area, int capacity, int level) {}

    static int compareNoviceView(Novice novice, Maintainer a, Maintainer b) {
        boolean aMatch = novice.area().equals(a.area()), bMatch = novice.area().equals(b.area());
        if (aMatch != bMatch) return aMatch ? -1 : 1;
        int da = Math.abs(novice.level() - a.level()), db = Math.abs(novice.level() - b.level());
        if (da != db) return Integer.compare(da, db);
        return a.id().compareTo(b.id());
    }

    static int compareMaintainerView(Maintainer maintainer, Novice a, Novice b) {
        boolean aMatch = maintainer.area().equals(a.area()), bMatch = maintainer.area().equals(b.area());
        if (aMatch != bMatch) return aMatch ? -1 : 1;
        int da = Math.abs(maintainer.level() - a.level()), db = Math.abs(maintainer.level() - b.level());
        if (da != db) return Integer.compare(da, db);
        return a.id().compareTo(b.id());
    }

    static final class Mentoring {
        final List<Novice> novices;
        final List<Maintainer> maintainers;
        final Map<String, Maintainer> maintainersById = new TreeMap<>();
        final Map<String, Novice> novicesById = new TreeMap<>();
        final Map<String, List<String>> novicePrefs = new TreeMap<>();
        final Map<String, Map<String, Integer>> maintainerRanks = new TreeMap<>();

        Mentoring(List<Novice> novices, List<Maintainer> maintainers) {
            this.novices = new ArrayList<>(novices);
            this.maintainers = new ArrayList<>(maintainers);
            for (Maintainer m : maintainers) maintainersById.put(m.id(), m);
            for (Novice novice : novices) novicesById.put(novice.id(), novice);
            for (Novice novice : novices) {
                List<Maintainer> ordered = new ArrayList<>();
                for (Maintainer m : maintainers) if (m.capacity() > 0) ordered.add(m);
                ordered.sort((a, b) -> compareNoviceView(novice, a, b));
                List<String> ids = new ArrayList<>();
                for (Maintainer m : ordered) ids.add(m.id());
                novicePrefs.put(novice.id(), ids);
            }
            for (Maintainer maintainer : maintainers) {
                List<Novice> ordered = new ArrayList<>(novices);
                ordered.sort((a, b) -> compareMaintainerView(maintainer, a, b));
                Map<String, Integer> ranks = new TreeMap<>();
                for (int i = 0; i < ordered.size(); i++) ranks.put(ordered.get(i).id(), i);
                maintainerRanks.put(maintainer.id(), ranks);
            }
        }

        String worstHeld(List<String> assigned, String maintainerId) {
            String worst = null;
            for (String noviceId : assigned)
                if (worst == null || maintainerRanks.get(maintainerId).get(noviceId) > maintainerRanks.get(maintainerId).get(worst))
                    worst = noviceId;
            return worst;
        }

        /** Capacitated deferred acceptance; the result is a stable assignment by construction. */
        Map<String, String> match() {
            Map<String, String> assignment = new TreeMap<>();
            Map<String, List<String>> held = new TreeMap<>();
            Map<String, Integer> next = new TreeMap<>();
            Deque<String> free = new ArrayDeque<>();
            for (String noviceId : novicePrefs.keySet()) { free.add(noviceId); next.put(noviceId, 0); }
            while (!free.isEmpty()) {
                String noviceId = free.poll();
                List<String> prefs = novicePrefs.get(noviceId);
                int index = next.get(noviceId);
                while (index < prefs.size()) {
                    String maintainerId = prefs.get(index++);
                    Maintainer maintainer = maintainersById.get(maintainerId);
                    List<String> assigned = held.computeIfAbsent(maintainerId, key -> new ArrayList<>());
                    if (assigned.size() < maintainer.capacity()) {
                        assigned.add(noviceId);
                        assignment.put(noviceId, maintainerId);
                        break;
                    }
                    String worst = worstHeld(assigned, maintainerId);
                    if (maintainerRanks.get(maintainerId).get(noviceId) < maintainerRanks.get(maintainerId).get(worst)) {
                        assigned.remove(worst);
                        assignment.remove(worst);
                        assigned.add(noviceId);
                        assignment.put(noviceId, maintainerId);
                        free.add(worst);
                        break;
                    }
                }
                next.put(noviceId, index);
            }
            return assignment;
        }

        List<String> unmatched(Map<String, String> assignment) {
            List<String> out = new ArrayList<>();
            for (Novice novice : novices) if (!assignment.containsKey(novice.id())) out.add(novice.id());
            return out;
        }

        int totalCapacity() {
            int total = 0;
            for (Maintainer m : maintainers) total += Math.max(0, m.capacity());
            return total;
        }

        Map<String, Integer> load(Map<String, String> assignment) {
            Map<String, Integer> loads = new TreeMap<>();
            for (Maintainer m : maintainers) loads.put(m.id(), 0);
            for (String maintainerId : assignment.values()) loads.merge(maintainerId, 1, Integer::sum);
            return loads;
        }

        String digest(Map<String, String> assignment) {
            long h = fnvText(FNV_OFFSET, "mentoring");
            for (Map.Entry<String, String> entry : assignment.entrySet()) h = fnvText(h, entry.getKey() + "->" + entry.getValue() + "|");
            for (String noviceId : unmatched(assignment)) h = fnvText(h, "unmatched:" + noviceId + "|");
            return hex(h);
        }
    }

    /** Independent blocking-pair scan, written without reusing the matching code. */
    static List<String> blockingPairs(Mentoring mentoring, Map<String, String> assignment) {
        Map<String, List<String>> held = new TreeMap<>();
        for (Map.Entry<String, String> entry : assignment.entrySet())
            held.computeIfAbsent(entry.getValue(), key -> new ArrayList<>()).add(entry.getKey());
        List<String> blocking = new ArrayList<>();
        for (Novice novice : mentoring.novices) {
            String current = assignment.get(novice.id());
            List<String> prefs = mentoring.novicePrefs.get(novice.id());
            int currentRank = current == null ? Integer.MAX_VALUE : prefs.indexOf(current);
            for (Maintainer maintainer : mentoring.maintainers) {
                if (maintainer.capacity() <= 0 || maintainer.id().equals(current)) continue;
                int newRank = prefs.indexOf(maintainer.id());
                if (newRank < 0 || newRank >= currentRank) continue;
                List<String> assigned = held.getOrDefault(maintainer.id(), new ArrayList<>());
                if (assigned.size() < maintainer.capacity()) { blocking.add(novice.id() + "->" + maintainer.id()); continue; }
                int worst = -1;
                for (String noviceId : assigned) worst = Math.max(worst, mentoring.maintainerRanks.get(maintainer.id()).get(noviceId));
                if (mentoring.maintainerRanks.get(maintainer.id()).get(novice.id()) < worst)
                    blocking.add(novice.id() + "->" + maintainer.id());
            }
        }
        return blocking;
    }

    static List<Novice> novices() {
        return List.of(new Novice("n1", "math", 2), new Novice("n2", "math", 3), new Novice("n3", "bio", 1),
                new Novice("n4", "bio", 2), new Novice("n5", "art", 2));
    }

    static List<Maintainer> ampleMaintainers() {
        return List.of(new Maintainer("m1", "math", 2, 3), new Maintainer("m2", "bio", 2, 2),
                new Maintainer("m3", "art", 1, 2), new Maintainer("m4", "bio", 1, 1));
    }

    public static void main(String[] args) {
        Mentoring ample = new Mentoring(novices(), ampleMaintainers());
        Map<String, String> assignment = ample.match();

        // 1. ample capacity pairs everybody with a valid, in-preference maintainer
        boolean allMatched = assignment.size() == novices().size() && ample.unmatched(assignment).isEmpty();
        for (Map.Entry<String, String> entry : assignment.entrySet())
            allMatched &= ample.novicePrefs.get(entry.getKey()).contains(entry.getValue());
        check(allMatched, "every novice is paired when total capacity is sufficient");

        // 2. area affinity: each novice keeps a same-area maintainer when one exists
        boolean affinity = true;
        for (Novice novice : novices()) {
            boolean sameAreaExists = false;
            for (Maintainer m : ampleMaintainers())
                if (m.area().equals(novice.area()) && m.capacity() > 0) sameAreaExists = true;
            if (!sameAreaExists) continue;
            String assigned = assignment.get(novice.id());
            affinity &= ample.maintainersById.get(assigned).area().equals(novice.area());
        }
        check(affinity, "novices are paired inside their own area while such slots exist");

        // 3. capacity is never exceeded and each novice appears exactly once
        Map<String, Integer> loads = ample.load(assignment);
        boolean loadsOk = true;
        int assignedCount = 0;
        for (Maintainer m : ampleMaintainers()) loadsOk &= loads.get(m.id()) <= m.capacity();
        for (int load : loads.values()) assignedCount += load;
        check(loadsOk && assignedCount == assignment.size(),
                "no maintainer is over capacity and each pair is counted once");

        // 4. stability against the independent blocking-pair scan
        check(blockingPairs(ample, assignment).isEmpty(),
                "no blocking pair exists in the ample-capacity pairing");

        // 5. tight capacity: five novices, two slots
        Mentoring tight = new Mentoring(novices(),
                List.of(new Maintainer("m1", "math", 1, 3), new Maintainer("m2", "bio", 1, 2)));
        Map<String, String> tightAssignment = tight.match();
        String toM1 = null, toM2 = null;
        for (Map.Entry<String, String> entry : tightAssignment.entrySet()) {
            if (entry.getValue().equals("m1")) toM1 = entry.getKey();
            if (entry.getValue().equals("m2")) toM2 = entry.getKey();
        }
        boolean areaKept = toM1 != null && tight.novicesById.get(toM1).area().equals("math")
                && toM2 != null && tight.novicesById.get(toM2).area().equals("bio");
        check(tightAssignment.size() == 2 && tight.unmatched(tightAssignment).size() == 3 && areaKept
                        && blockingPairs(tight, tightAssignment).isEmpty(),
                "tight capacity pairs exactly two novices, keeps area affinity and stays stable");

        // 6. shuffled input yields the same pairing and digest
        List<Novice> shuffledNovices = new ArrayList<>(novices());
        List<Maintainer> shuffledMaintainers = new ArrayList<>(ampleMaintainers());
        Collections.reverse(shuffledNovices);
        Collections.reverse(shuffledMaintainers);
        Mentoring shuffled = new Mentoring(shuffledNovices, shuffledMaintainers);
        Map<String, String> shuffledAssignment = shuffled.match();
        check(shuffledAssignment.equals(assignment) && shuffled.digest(shuffledAssignment).equals(ample.digest(assignment)),
                "reversed input lists produce the identical pairing and digest");

        // 7. boundaries: no novices, no maintainers, zero-capacity maintainers
        Mentoring noNovices = new Mentoring(List.of(), ampleMaintainers());
        Mentoring noMaintainers = new Mentoring(novices(), List.of());
        Mentoring zeroCapacity = new Mentoring(novices(), List.of(new Maintainer("m0", "math", 0, 2)));
        check(noNovices.match().isEmpty() && noMaintainers.match().isEmpty()
                        && noMaintainers.unmatched(noMaintainers.match()).size() == 5
                        && zeroCapacity.match().isEmpty() && zeroCapacity.totalCapacity() == 0,
                "empty inputs and zero-capacity maintainers produce no pairing and a full unmatched list");

        // 8. independent oracle: matched count is the feasible maximum and never above total capacity
        boolean oracleOk = true;
        for (int i = 0; i < 20; i++) {
            List<Maintainer> scaled = new ArrayList<>();
            for (int k = 0; k < 1 + i % 5; k++)
                scaled.add(new Maintainer("m" + k, k % 2 == 0 ? "math" : "bio", 1 + (i + k) % 3, 1 + k % 3));
            Mentoring mentoring = new Mentoring(novices(), scaled);
            Map<String, String> pairs = mentoring.match();
            int feasible = Math.min(novices().size(), mentoring.totalCapacity());
            oracleOk &= pairs.size() == feasible && blockingPairs(mentoring, pairs).isEmpty()
                    && mentoring.unmatched(pairs).size() == novices().size() - pairs.size();
        }
        check(oracleOk, "matched count equals min(novices, total capacity) over 20 random capacities");

        System.out.println("ample pairing: " + assignment);
        System.out.println("loads: " + loads + ", unmatched: " + ample.unmatched(assignment));
        System.out.println("tight pairing: " + tightAssignment + ", unmatched: " + tight.unmatched(tightAssignment));
        System.out.println("pairing digest: " + ample.digest(assignment));
        System.out.println("PASS");
    }
}
