/**
 * Task 345 - RandomizedSet: insert/delete/getRandom in O(1). (Tier: Intermediate, CTCI Ch 7)
 *
 * Array + value -> index map; removal swaps the victim with the last element.  getRandom uses
 * the repo's deterministic LCG (seed = (seed*75+74) % 65537), so after the fixed operation
 * script every language produces the same draw sequence - asserted literally below.
 */
import java.util.*;

class RandomizedSet {
    private final List<Integer> items = new ArrayList<>();
    private final Map<Integer, Integer> index = new HashMap<>();
    private long rng;

    RandomizedSet(long seed) {
        this.rng = seed;
    }

    boolean insert(int value) {
        if (index.containsKey(value)) return false;
        index.put(value, items.size());
        items.add(value);
        return true;
    }

    boolean remove(int value) {
        Integer at = index.get(value);
        if (at == null) return false;
        int last = items.get(items.size() - 1);
        items.set(at, last);                      // move the last element into the hole
        index.put(last, at);
        items.remove(items.size() - 1);
        index.remove(value);
        return true;
    }

    int getRandom() {
        if (items.isEmpty()) throw new IllegalStateException("getRandom on an empty set");
        rng = (rng * 75 + 74) % 65537;
        return items.get((int) (rng % items.size()));
    }

    int size() {
        return items.size();
    }

    List<Integer> toList() {
        return new ArrayList<>(items);
    }

    boolean validate() {
        if (items.size() != index.size()) return false;
        for (Map.Entry<Integer, Integer> entry : index.entrySet()) {
            if (!items.get(entry.getValue()).equals(entry.getKey())) return false;
        }
        return true;
    }
}

public class Solution {
    public static void main(String[] args) {
        RandomizedSet set = new RandomizedSet(2024);
        check(set.size() == 0, "starts empty");

        int[] values = {5, 3, 9, 1, 7, 5};
        boolean[] expected = {true, true, true, true, true, false};
        for (int i = 0; i < values.length; i++) {
            check(set.insert(values[i]) == expected[i], "insert " + values[i]);
        }
        check(set.remove(3), "remove present value");
        check(!set.remove(42), "remove missing value returns false");
        check(set.insert(3), "re-insert after removal");
        check(set.toList().equals(Arrays.asList(5, 7, 9, 1, 3)), "swap-remove layout");
        check(set.size() == 5, "size 5");
        check(set.validate(), "array and map agree");

        // deterministic LCG draws - identical in every language
        int[] draws = {5, 1, 3, 3, 3, 1, 1, 1};
        for (int i = 0; i < draws.length; i++) {
            check(set.getRandom() == draws[i], "draw " + i + " expected " + draws[i]);
        }

        RandomizedSet empty = new RandomizedSet(1);
        try {
            empty.getRandom();
            throw new AssertionError("expected IllegalStateException from getRandom on an empty set");
        } catch (IllegalStateException expectedExc) {
            // documented behaviour
        }

        RandomizedSet bulk = new RandomizedSet(98765);
        for (int v = 1; v <= 50; v++) check(bulk.insert(v), "bulk insert " + v);
        for (int v = 1; v <= 50; v++) check(!bulk.insert(v), "bulk duplicate " + v);
        for (int v = 2; v <= 50; v += 2) check(bulk.remove(v), "bulk remove " + v);
        for (int v = 2; v <= 50; v += 2) check(!bulk.remove(v), "bulk re-remove " + v);
        check(bulk.size() == 25, "25 odds remain");
        List<Integer> remaining = bulk.toList();
        Collections.sort(remaining);
        List<Integer> odds = new ArrayList<>();
        for (int v = 1; v <= 50; v += 2) odds.add(v);
        check(remaining.equals(odds), "only odds remain");
        check(bulk.validate(), "bulk layout consistent");
        Set<Integer> oddsSet = new HashSet<>(odds);
        Set<Integer> drawn = new HashSet<>();
        for (int i = 0; i < 500; i++) {
            int value = bulk.getRandom();
            check(oddsSet.contains(value), "draw is one of the odds");
            drawn.add(value);
        }
        check(drawn.equals(oddsSet), "500 deterministic draws cover all odds");

        RandomizedSet small = new RandomizedSet(7);
        small.insert(10);
        small.insert(20);
        small.insert(30);
        check(small.remove(10), "swap victim removed");
        check(small.toList().equals(Arrays.asList(30, 20)), "layout after victim removal");
        check(small.remove(20), "last element removed");
        check(small.toList().equals(Arrays.asList(30)), "layout after last removal");
        check(small.remove(30), "final removal");
        check(small.toList().isEmpty(), "empty again");
        check(small.validate(), "small layout consistent");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
