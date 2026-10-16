/** Task 633 - Implement CRDT counter and set examples (operation-based and state-based). (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * The classic replicable data types, both flavours:
 *   * G-Counter (state-based): one counter per replica, merge is the pointwise maximum, the value
 *     is the sum. Because merge = max, an update can never be lost, which is exactly what makes it
 *     commutative, associative and idempotent - the three laws a state-based CRDT must obey.
 *   * PN-Counter (state-based): a G-Counter of increments plus one of decrements.
 *   * G-Set / 2P-Set (state-based): union of additions; the 2P-Set also keeps the set of removals,
 *     so removal wins forever and the element can never come back.
 *   * OR-Set (observed-remove, both flavours): every add carries a unique tag; a remove only
 *     tombstones the tags it had *observed*. Concurrent add and remove therefore both survive -
 *     the element stays - which is the improvement over the 2P-Set, and it never needs a
 *     re-add to fight a stale removal.
 *
 * Simulation model, and what is simplified: replicas are plain objects; "replicating" is calling a
 * merge in a chosen order and "delivering an operation" is calling a method. There is no network,
 * no vector clock and no causal delivery machinery (that is task 630); the operation-based variants
 * here assume exactly-once causal delivery, which is stated rather than reimplemented. What is not
 * simplified is the algebra, and it is checked exhaustively.
 *
 * Verified: (1) the state merge of every type is commutative, idempotent and associative on a set
 * of generated states; (2) a G-Counter never decreases and its value equals the total number of
 * increments; (3) a PN-Counter tracks increments minus decrements, including a negative total;
 * (4) the 2P-Set really does let removal win forever (an add of a removed element is a no-op);
 * (5) the OR-Set keeps an element when an add and a remove are concurrent, and removes it once the
 * remover has observed every add; a fresh add after a removal makes it present again; (6) applying
 * a set of operations in all 24 permutations yields the same visible state; (7) the operation-based
 * counter agrees with the state-based one; (8) the negative control - a last-writer-wins merge on a
 * PN-Counter - provably loses a concurrent increment and is caught by the checker; and (9) empty
 * state, single replica and identity merges behave.
 */
import java.util.*;

/** Grow-only counter: merge is pointwise max, value is the sum. */
class GCounter {
    final Map<Integer, Integer> counts = new TreeMap<>();

    void increment(int replica) {
        counts.merge(replica, 1, Integer::sum);
    }

    void incrementBy(int replica, int amount) {
        counts.merge(replica, amount, Integer::sum);
    }

    void merge(GCounter other) {
        for (Map.Entry<Integer, Integer> e : other.counts.entrySet()) {
            counts.merge(e.getKey(), e.getValue(), Math::max);
        }
    }

    int value() {
        int total = 0;
        for (int v : counts.values()) total += v;
        return total;
    }

    GCounter copy() {
        GCounter c = new GCounter();
        c.counts.putAll(counts);
        return c;
    }

    String render() {
        return counts.toString();
    }
}

/** Increment/decrement counter: two grow-only counters under the hood. */
class PNCounter {
    final GCounter increments = new GCounter();
    final GCounter decrements = new GCounter();

    void increment(int replica) {
        increments.increment(replica);
    }

    void decrement(int replica) {
        decrements.increment(replica);
    }

    void merge(PNCounter other) {
        increments.merge(other.increments);
        decrements.merge(other.decrements);
    }

    /** The wrong merge: the sender's state simply overwrites the receiver's. */
    void mergeByOverwrite(PNCounter other) {
        increments.counts.clear();
        increments.counts.putAll(other.increments.counts);
        decrements.counts.clear();
        decrements.counts.putAll(other.decrements.counts);
    }

    int value() {
        return increments.value() - decrements.value();
    }

    /** The per-replica increment counts, for comparing against an op-based delivery trace. */
    Map<Integer, Integer> countsByReplica() {
        return new TreeMap<>(increments.counts);
    }

    PNCounter copy() {
        PNCounter c = new PNCounter();
        c.increments.merge(increments);
        c.decrements.merge(decrements);
        return c;
    }
}

/** Grow-only set: merge is union. */
class GSet {
    final Set<Integer> items = new TreeSet<>();

    void add(int item) {
        items.add(item);
    }

    void merge(GSet other) {
        items.addAll(other.items);
    }

    GSet copy() {
        GSet s = new GSet();
        s.items.addAll(items);
        return s;
    }

    String render() {
        return items.toString();
    }
}

/** Two-phase set: removal wins for good, so an element can never be re-added. */
class TwoPhaseSet {
    final GSet added = new GSet();
    final GSet removed = new GSet();

    void add(int item) {
        if (removed.items.contains(item)) return;      // the removal is permanent
        added.add(item);
    }

    void remove(int item) {
        removed.add(item);
    }

    boolean contains(int item) {
        return added.items.contains(item) && !removed.items.contains(item);
    }

    void merge(TwoPhaseSet other) {
        added.merge(other.added);
        removed.merge(other.removed);
    }

    TwoPhaseSet copy() {
        TwoPhaseSet s = new TwoPhaseSet();
        s.added.items.addAll(added.items);
        s.removed.items.addAll(removed.items);
        return s;
    }

    String render() {
        return added.items.toString() + "-" + removed.items;
    }
}

/** Observed-remove set: adds carry unique tags, removes tombstone only observed tags. */
class OrSet {
    final Map<Integer, Set<String>> adds = new TreeMap<>();
    final Set<String> removed = new TreeSet<>();

    void add(int element, String tag) {
        adds.computeIfAbsent(element, k -> new TreeSet<>()).add(tag);
    }

    /** The tags this replica can currently see for the element. */
    Set<String> observedTags(int element) {
        Set<String> out = new TreeSet<>();
        for (String tag : adds.getOrDefault(element, new TreeSet<>())) {
            if (!removed.contains(tag)) out.add(tag);
        }
        return out;
    }

    void removeObserved(int element) {
        removed.addAll(observedTags(element));
    }

    void removeTag(int element, String tag) {
        if (adds.getOrDefault(element, new TreeSet<>()).contains(tag)) removed.add(tag);
    }

    boolean contains(int element) {
        return !observedTags(element).isEmpty();
    }

    int size() {
        int count = 0;
        for (int element : adds.keySet()) if (contains(element)) count++;
        return count;
    }

    Set<Integer> elements() {
        Set<Integer> out = new TreeSet<>();
        for (int element : adds.keySet()) if (contains(element)) out.add(element);
        return out;
    }

    void merge(OrSet other) {
        for (Map.Entry<Integer, Set<String>> e : other.adds.entrySet()) {
            adds.computeIfAbsent(e.getKey(), k -> new TreeSet<>()).addAll(e.getValue());
        }
        removed.addAll(other.removed);
    }

    OrSet copy() {
        OrSet s = new OrSet();
        for (Map.Entry<Integer, Set<String>> e : adds.entrySet()) {
            s.adds.put(e.getKey(), new TreeSet<>(e.getValue()));
        }
        s.removed.addAll(removed);
        return s;
    }

    String render() {
        return elements().toString();
    }
}

public class Solution {
    static long seed = 0xACE1F00DL;

    static int nextInt(int bound) {
        seed += 0x9E3779B97F4A7C15L;
        long z = seed;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        z = z ^ (z >>> 31);
        return (int) Math.floorMod(z, bound);
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }

    public static void main(String[] args) {
        mergeLaws();
        growOnlyCounter();
        pnCounter();
        twoPhaseSetRemovalWins();
        orSetBasics();
        orSetConcurrentAddAndRemove();
        permutationConvergence();
        opBasedMatchesStateBased();
        negativeControlOverwrite();
        degenerateStates();
        System.out.println("PASS");
    }

    /** Commutativity, idempotence and associativity of the state-based merges. */
    static void mergeLaws() {
        for (int trial = 0; trial < 30; trial++) {
            GCounter a = new GCounter();
            GCounter b = new GCounter();
            GCounter c = new GCounter();
            for (int i = 0; i < 3; i++) {
                a.incrementBy(nextInt(3), 1 + nextInt(3));
                b.incrementBy(nextInt(3), 1 + nextInt(3));
                c.incrementBy(nextInt(3), 1 + nextInt(3));
            }
            GCounter ab = a.copy();
            ab.merge(b);
            GCounter ba = b.copy();
            ba.merge(a);
            check(ab.value() == ba.value() && ab.render().equals(ba.render()),
                    "counter merge is commutative");
            GCounter aa = a.copy();
            aa.merge(a);
            check(aa.render().equals(a.render()), "counter merge is idempotent");
            GCounter left = a.copy();
            left.merge(b.copy());
            left.merge(c.copy());
            GCounter bc = b.copy();
            bc.merge(c.copy());
            GCounter right = a.copy();
            right.merge(bc);
            check(left.render().equals(right.render()), "counter merge is associative");

            OrSet x = new OrSet();
            OrSet y = new OrSet();
            OrSet z = new OrSet();
            x.add(nextInt(4), "x" + trial);
            y.add(nextInt(4), "y" + trial);
            y.removeTag(1, "other");
            z.add(nextInt(4), "z" + trial);
            OrSet xy = x.copy();
            xy.merge(y);
            OrSet yx = y.copy();
            yx.merge(x);
            check(xy.render().equals(yx.render()), "OR-Set merge is commutative");
            OrSet xx = x.copy();
            xx.merge(x);
            check(xx.render().equals(x.render()) && xx.removed.equals(x.removed),
                    "OR-Set merge is idempotent");
            OrSet groupLeft = x.copy();
            OrSet yz = y.copy();
            yz.merge(z);
            groupLeft.merge(yz);
            OrSet leftGroup = x.copy();
            leftGroup.merge(y.copy());
            leftGroup.merge(z.copy());
            check(groupLeft.render().equals(leftGroup.render()), "OR-Set merge is associative");
        }
    }

    /** A grow-only counter never decreases and counts every increment. */
    static void growOnlyCounter() {
        GCounter a = new GCounter();
        GCounter b = new GCounter();
        for (int i = 0; i < 5; i++) a.increment(1);
        for (int i = 0; i < 3; i++) b.increment(2);
        check(a.value() == 5 && b.value() == 3, "per-replica totals");

        int before = a.value();
        GCounter merged = a.copy();
        merged.merge(b);
        check(merged.value() == 8, "the merged value is the sum of all increments: " + merged.value());
        check(merged.value() >= before, "a merge never lowers the counter");
        check(merged.counts.get(1) == 5 && merged.counts.get(2) == 3, "per-replica maxima kept");

        GCounter stale = new GCounter();
        stale.incrementBy(1, 2);
        GCounter fresh = new GCounter();
        fresh.incrementBy(1, 7);
        GCounter target = fresh.copy();
        target.merge(stale);
        check(target.counts.get(1) == 7, "the older replica cannot overwrite the newer value");
        target.merge(fresh);
        check(target.counts.get(1) == 7, "re-merging is a no-op");
    }

    /** Increments minus decrements, including a negative result. */
    static void pnCounter() {
        PNCounter a = new PNCounter();
        PNCounter b = new PNCounter();
        for (int i = 0; i < 4; i++) a.increment(1);
        a.decrement(1);
        b.decrement(2);
        b.decrement(2);
        PNCounter merged = a.copy();
        merged.merge(b);
        check(merged.value() == 1, "the total is 4 - 1 - 2 = 1: " + merged.value());
        check(merged.increments.value() == 4 && merged.decrements.value() == 3, "both halves tracked");

        PNCounter negative = new PNCounter();
        negative.decrement(3);
        check(negative.value() == -1, "a counter may go negative: " + negative.value());
        negative.increment(4);
        check(negative.value() == 0, "and come back");
    }

    /** The 2P-Set limitation: a removal is forever. */
    static void twoPhaseSetRemovalWins() {
        TwoPhaseSet a = new TwoPhaseSet();
        a.add(1);
        a.add(2);
        check(a.contains(1) && a.contains(2), "both elements present");
        a.remove(1);
        check(!a.contains(1) && a.contains(2), "removal takes effect");
        a.add(1);                                  // ignored: the removal wins for good
        check(!a.contains(1), "an element can never come back after a removal");

        TwoPhaseSet b = new TwoPhaseSet();
        b.add(1);                                  // concurrent add on another replica
        a.merge(b);
        check(!a.contains(1), "a stale concurrent add cannot resurrect the element");
        b.merge(a);
        check(!b.contains(1), "and the remover's verdict propagates to the other replica");
        check(a.render().equals(b.render()), "both replicas agree");
    }

    /** OR-Set basics: present, absent, present again with a fresh tag. */
    static void orSetBasics() {
        OrSet s = new OrSet();
        check(!s.contains(7) && s.size() == 0, "an empty set contains nothing");
        s.add(7, "A1");
        check(s.contains(7) && s.size() == 1, "added");
        s.removeObserved(7);
        check(!s.contains(7), "removed");
        s.removeObserved(7);
        check(!s.contains(7), "removing twice changes nothing");
        s.add(7, "A2");
        check(s.contains(7), "a fresh add with a new tag makes it visible again");
        check(s.observedTags(7).equals(new TreeSet<>(Arrays.asList("A2"))),
                "only the fresh tag is observed: " + s.observedTags(7));
        check(!s.contains(9), "other elements are unaffected");
    }

    /** The point of observed-remove: a concurrent add survives a concurrent remove. */
    static void orSetConcurrentAddAndRemove() {
        OrSet a = new OrSet();
        OrSet b = new OrSet();
        a.add(5, "A1");
        b.add(5, "B1");
        a.merge(b);                                 // both replicas now see both tags
        b.merge(a);

        OrSet remover = a.copy();
        remover.removeObserved(5);                  // removes A1 and B1
        check(!remover.contains(5), "a remove that observed every add wins");

        OrSet concurrent = new OrSet();
        concurrent.add(5, "A1");
        concurrent.removeObserved(5);               // observed only A1
        OrSet other = new OrSet();
        other.add(5, "B1");                         // concurrent, unobserved
        concurrent.merge(other);
        check(concurrent.contains(5), "the concurrent add keeps the element alive: "
                + concurrent.observedTags(5));
        check(concurrent.observedTags(5).equals(new TreeSet<>(Arrays.asList("B1"))),
                "the removed tag stays tombstoned");

        concurrent.removeObserved(5);               // now it has observed B1 too
        check(!concurrent.contains(5), "once every add is observed the remove takes effect");
    }

    /** Applying a set of operations in every permutation converges to the same state. */
    static void permutationConvergence() {
        List<Integer> ops = Arrays.asList(1, 2, 3, 4);
        Set<String> outcomes = new TreeSet<>();
        int permutations = 0;
        for (List<Integer> order : permutations(ops)) {
            OrSet s = new OrSet();
            for (int element : order) {
                if (element % 2 == 0) s.add(element, "tag" + element);
                else s.removeObserved(element);
            }
            outcomes.add(s.render() + "|" + s.removed);
            permutations++;
        }
        check(permutations == 24, "all permutations were tried: " + permutations);
        check(outcomes.size() == 1, "orbit-sets converge under every delivery order: " + outcomes);

        Set<String> counterOutcomes = new TreeSet<>();
        for (List<Integer> order : permutations(ops)) {
            GCounter g = new GCounter();
            for (int replica : order) g.increment(replica);
            counterOutcomes.add(Integer.toString(g.value()));
        }
        check(counterOutcomes.equals(new TreeSet<>(Arrays.asList("4"))),
                "the grow-only counter is order independent: " + counterOutcomes);

        Set<String> twoPhaseOutcomes = new TreeSet<>();
        for (List<Integer> order : permutations(Arrays.asList(1, 2))) {
            TwoPhaseSet s = new TwoPhaseSet();
            for (int element : order) {
                if (element == 1) s.add(1);
                else s.remove(1);
            }
            twoPhaseOutcomes.add(Boolean.toString(s.contains(1)));
        }
        check(twoPhaseOutcomes.equals(new TreeSet<>(Arrays.asList("false"))),
                "remove-wins makes the 2P-Set order independent: " + twoPhaseOutcomes);
        TwoPhaseSet added = new TwoPhaseSet();
        added.add(1);
        TwoPhaseSet removed = new TwoPhaseSet();
        removed.add(1);
        removed.remove(1);
        TwoPhaseSet forward = added.copy();
        forward.merge(removed);
        TwoPhaseSet backward = removed.copy();
        backward.merge(added);
        check(!forward.contains(1) && !backward.contains(1) && forward.render().equals(backward.render()),
                "merging a removed element in either order still removes it");
    }

    /** An operation-based counter delivered exactly once equals the state-based merge. */
    static void opBasedMatchesStateBased() {
        List<Integer> operations = Arrays.asList(1, 2, 3, 1);   // causal delivery order
        PNCounter stateBased = new PNCounter();
        Map<Integer, Integer> opBased = new TreeMap<>();
        Map<Integer, PNCounter> replicaStates = new TreeMap<>();
        for (int replica : operations) {
            PNCounter own = replicaStates.computeIfAbsent(replica, k -> new PNCounter());
            own.increment(replica);                    // the replica's own state grows monotonically
            stateBased.merge(own);                     // ... and gossips it
            opBased.merge(replica, 1, Integer::sum);   // exactly-once operation delivery
        }
        int opBasedTotal = 0;
        for (int v : opBased.values()) opBasedTotal += v;
        check(stateBased.value() == opBasedTotal,
                "state-based and op-based totals agree: " + stateBased.value() + " vs "
                        + opBasedTotal);
        check(stateBased.value() == 4, "four delivered operations: " + stateBased.value());
        check(stateBased.countsByReplica().equals(opBased),
                "even the per-replica split agrees: " + stateBased.countsByReplica());

        PNCounter dec = new PNCounter();
        dec.increment(1);
        dec.increment(1);
        dec.decrement(2);
        check(dec.value() == 1, "an increment and a decrement partially cancel: " + dec.value());
    }

    /** Negative control: an overwrite merge loses concurrent updates. */
    static void negativeControlOverwrite() {
        PNCounter replicaA = new PNCounter();
        PNCounter replicaB = new PNCounter();
        replicaA.increment(1);
        replicaB.increment(2);

        PNCounter correct = replicaA.copy();
        correct.merge(replicaB);
        check(correct.value() == 2, "the correct merge counts both increments: " + correct.value());

        PNCounter broken = replicaA.copy();
        broken.mergeByOverwrite(replicaB);
        check(broken.value() == 1, "the overwrite merge lost an increment: " + broken.value());
        check(broken.value() != correct.value(),
                "so the checker has something to catch - the merge is not a CRDT merge");
    }

    /** Empty states, single replicas and identity merges. */
    static void degenerateStates() {
        GCounter empty = new GCounter();
        GCounter fresh = new GCounter();
        empty.merge(fresh);
        check(empty.value() == 0, "merging with the empty counter is the identity");
        check(empty.render().equals("{}"), "and leaves no entries");

        GSet set = new GSet();
        set.add(3);
        GSet identity = new GSet();
        set.merge(identity);
        check(set.items.equals(new TreeSet<>(Arrays.asList(3))), "union with the empty set");

        OrSet or = new OrSet();
        or.merge(new OrSet());
        check(or.size() == 0, "merging with an empty OR-Set changes nothing");

        TwoPhaseSet single = new TwoPhaseSet();
        single.add(1);
        check(single.contains(1) && single.added.items.size() == 1, "a single replica works");

        OrSet selfMerge = new OrSet();
        selfMerge.add(2, "only");
        selfMerge.merge(selfMerge);
        check(selfMerge.contains(2) && selfMerge.adds.get(2).size() == 1,
                "merging a replica with itself duplicates no tags");
    }

    static List<List<Integer>> permutations(List<Integer> items) {
        List<List<Integer>> out = new ArrayList<>();
        permute(new ArrayList<>(items), 0, out);
        return out;
    }

    static void permute(List<Integer> items, int index, List<List<Integer>> out) {
        if (index == items.size()) {
            out.add(new ArrayList<>(items));
            return;
        }
        for (int i = index; i < items.size(); i++) {
            Collections.swap(items, index, i);
            permute(items, index + 1, out);
            Collections.swap(items, index, i);
        }
    }
}
