/** Task 440 - Implement integer set with van Emde Boas (conceptual small universe). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Documented simplification of a van Emde Boas set on the small universe U = 256 = 16 x 16: the top
 * level keeps min/max, a 16-entry summary of cluster occupancy and 16 lazily created clusters, while
 * the summary and every cluster is the base case - a 16-bit member bitmask (BitSet16) whose bit scans
 * replace the recursion below u = 16. The decisive vEB trick is kept: the minimum is stored outside
 * the clusters, so delete(min) promotes the successor from the cluster/summary structure. Operations
 * MEMBER / SUCCESSOR / PREDECESSOR / INSERT / DELETE therefore match a full recursive vEB
 * (O(1) bit work at this two-level scale). Assertions cover hand sequences that cross cluster
 * boundaries, empty the structure and exercise successor/predecessor at both universe ends, plus LCG
 * operation scripts cross-checked against a sorted-set oracle with full 0..255 sweeps.
 */
import java.util.*;

class BitSet16 {
    int mask;

    void insert(int value) {
        mask |= 1 << value;
    }

    void remove(int value) {
        mask &= ~(1 << value);
    }

    boolean member(int value) {
        return ((mask >> value) & 1) == 1;
    }

    boolean empty() {
        return mask == 0;
    }

    static int lowestBit(int bits) {
        int index = 0;
        while ((bits & 1) == 0) {
            bits >>= 1;
            index++;
        }
        return index;
    }

    static int highestBit(int bits) {
        int index = -1;
        while (bits != 0) {
            bits >>= 1;
            index++;
        }
        return index;
    }

    Integer minimum() {
        return mask == 0 ? null : lowestBit(mask);
    }

    Integer maximum() {
        return mask == 0 ? null : highestBit(mask);
    }

    Integer successor(int value) {
        int higher = mask >> (value + 1);
        return higher == 0 ? null : value + 1 + lowestBit(higher);
    }

    Integer predecessor(int value) {
        int lower = mask & ((1 << value) - 1);
        return lower == 0 ? null : highestBit(lower);
    }
}

class VanEmdeBoas {
    static final int UNIVERSE = 256;
    static final int CLUSTERS = 16;

    Integer min;
    Integer max;
    int count;
    BitSet16 summary = new BitSet16();
    BitSet16[] clusters = new BitSet16[CLUSTERS];

    int size() {
        return count;
    }

    boolean member(int value) {
        if (min != null && value == min) return true;
        if (max != null && value == max) return true;
        BitSet16 cluster = clusters[value >> 4];
        return cluster != null && cluster.member(value & 15);
    }

    void insert(int value) {
        if (member(value)) return;
        if (min == null) {
            min = value;
            max = value;
            count = 1;
            return;
        }
        if (value < min) {
            int previousMin = min;
            min = value;
            value = previousMin;
        }
        if (value > max) max = value;
        int high = value >> 4;
        int low = value & 15;
        if (clusters[high] == null) {
            clusters[high] = new BitSet16();
            summary.insert(high);
        }
        clusters[high].insert(low);
        count++;
    }

    boolean delete(int value) {
        if (!member(value)) return false;
        if (count == 1) {
            min = null;
            max = null;
            summary = new BitSet16();
            clusters = new BitSet16[CLUSTERS];
            count = 0;
            return true;
        }
        if (value == min) {
            int successor = successorStored(value);
            min = successor;
            deleteStored(successor);
            return true;
        }
        deleteStored(value);
        return true;
    }

    private int successorStored(int value) {
        int high = value >> 4;
        int low = value & 15;
        BitSet16 cluster = clusters[high];
        if (cluster != null && low < 15) {
            Integer within = cluster.successor(low);
            if (within != null) return (high << 4) | within;
        }
        int nextHigh = summary.successor(high);
        return (nextHigh << 4) | clusters[nextHigh].minimum();
    }

    private void deleteStored(int value) {
        int high = value >> 4;
        int low = value & 15;
        clusters[high].remove(low);
        if (clusters[high].empty()) {
            summary.remove(high);
            clusters[high] = null;
        }
        count--;
        if (value == max) {
            Integer top = summary.maximum();
            max = top == null ? min : (top << 4) | clusters[top].maximum();
        }
    }

    Integer successor(int value) {
        if (min == null || value >= max) return null;
        if (value < min) return min;
        int high = value >> 4;
        int low = value & 15;
        BitSet16 cluster = clusters[high];
        if (cluster != null && low < 15) {
            Integer within = cluster.successor(low);
            if (within != null) return (high << 4) | within;
        }
        Integer nextHigh = summary.successor(high);
        if (nextHigh == null) return null;
        return (nextHigh << 4) | clusters[nextHigh].minimum();
    }

    Integer predecessor(int value) {
        if (min == null || value <= min) return null;
        if (value > max) return max;
        int high = value >> 4;
        int low = value & 15;
        BitSet16 cluster = clusters[high];
        if (cluster != null && low > 0) {
            Integer within = cluster.predecessor(low);
            if (within != null) return (high << 4) | within;
        }
        Integer previous = summary.predecessor(high);
        if (previous == null) return min;
        return (previous << 4) | clusters[previous].maximum();
    }
}

public class Solution {
    static void checkEquals(Integer actual, Integer expected, String label) {
        check(actual == null ? expected == null : actual.equals(expected), label + " got " + actual);
    }

    public static void main(String[] args) {
        VanEmdeBoas veb = new VanEmdeBoas();
        check(veb.size() == 0, "empty size");
        check(!veb.member(0), "empty member");
        checkEquals(veb.successor(0), null, "empty successor");
        checkEquals(veb.predecessor(255), null, "empty predecessor");
        check(!veb.delete(0), "empty delete");

        veb.insert(200);
        check(veb.size() == 1, "single size");
        check(veb.member(200) && !veb.member(199), "single member");
        checkEquals(veb.successor(200), null, "single successor of max");
        checkEquals(veb.successor(5), 200, "single successor below");
        checkEquals(veb.predecessor(200), null, "single predecessor of min");
        checkEquals(veb.predecessor(255), 200, "single predecessor above");
        check(veb.delete(200), "single delete");
        check(!veb.delete(200), "double delete");
        check(veb.size() == 0, "single drained");

        veb.insert(5);
        veb.insert(100);
        veb.insert(200);
        check(veb.size() == 3, "hand size");
        checkEquals(veb.successor(5), 100, "hand successor middle");
        checkEquals(veb.successor(4), 5, "hand successor from below");
        checkEquals(veb.predecessor(100), 5, "hand predecessor middle");
        checkEquals(veb.predecessor(101), 100, "hand predecessor close");
        checkEquals(veb.successor(199), 200, "hand successor near max");
        check(veb.delete(100), "hand delete middle");
        check(!veb.member(100), "hand absent after delete");
        checkEquals(veb.successor(5), 200, "successor skips hole");
        checkEquals(veb.predecessor(200), 5, "predecessor skips hole");
        check(veb.delete(5), "delete min promotion");
        checkEquals(veb.successor(199), 200, "min promotion successor");
        checkEquals(veb.predecessor(199), null, "min promotion predecessor");
        check(veb.delete(200), "delete last");
        checkEquals(veb.successor(0), null, "empty again");

        veb.insert(0);
        veb.insert(15);
        veb.insert(16);
        veb.insert(17);
        veb.insert(255);
        checkEquals(veb.successor(0), 15, "boundary 0->15");
        checkEquals(veb.successor(15), 16, "boundary cluster 0/1");
        checkEquals(veb.successor(16), 17, "boundary 16->17");
        checkEquals(veb.successor(17), 255, "boundary to 255");
        checkEquals(veb.predecessor(255), 17, "boundary from 255");
        checkEquals(veb.predecessor(17), 16, "boundary 17->16");
        checkEquals(veb.predecessor(16), 15, "boundary cluster 1/0");
        checkEquals(veb.predecessor(0), null, "boundary below 0");
        check(veb.delete(16), "boundary delete");
        checkEquals(veb.successor(15), 17, "after boundary delete");
        check(veb.delete(0), "delete min 0");
        check(veb.min == 15, "min promoted to 15");
        check(veb.delete(15), "delete min 15");
        check(veb.min == 17, "min promoted to 17");
        check(veb.delete(17), "delete min 17");
        check(veb.min == 255 && veb.max == 255, "last element state");
        check(veb.delete(255), "delete 255");
        check(veb.size() == 0 && veb.min == null, "final empty");

        veb = new VanEmdeBoas();
        List<Integer> oracle = new ArrayList<>();
        int seed = 4400;
        for (int step = 0; step < 600; step++) {
            seed = (seed * 75 + 74) % 65537;
            int op = seed % 5;
            seed = (seed * 75 + 74) % 65537;
            int value = seed % 256;
            if (op <= 1) {
                veb.insert(value);
                if (!oracle.contains(value)) {
                    int index = Collections.binarySearch(oracle, value);
                    oracle.add(index < 0 ? -(index + 1) : index, value);
                }
            } else if (op == 2) {
                boolean expected = oracle.contains(value);
                if (expected) oracle.remove(Integer.valueOf(value));
                check(veb.delete(value) == expected, "random delete step " + step);
            } else if (op == 3) {
                check(veb.member(value) == oracle.contains(value), "random member step " + step);
            } else {
                Integer expectedSuccessor = null;
                Integer expectedPredecessor = null;
                for (int candidate : oracle) {
                    if (candidate > value && expectedSuccessor == null) expectedSuccessor = candidate;
                    if (candidate < value) expectedPredecessor = candidate;
                }
                checkEquals(veb.successor(value), expectedSuccessor, "random successor step " + step);
                checkEquals(veb.predecessor(value), expectedPredecessor, "random predecessor step " + step);
            }
            check(veb.size() == oracle.size(), "random size step " + step);
            if (step % 50 == 0) {
                for (int probe = 0; probe < 256; probe++) {
                    check(veb.member(probe) == oracle.contains(probe), "sweep member step " + step + " probe " + probe);
                }
            }
        }
        check(!oracle.isEmpty(), "workload left values");
        for (int probe = 0; probe < 256; probe++) {
            check(veb.member(probe) == oracle.contains(probe), "final member probe " + probe);
            Integer expectedSuccessor = null;
            Integer expectedPredecessor = null;
            for (int candidate : oracle) {
                if (candidate > probe && expectedSuccessor == null) expectedSuccessor = candidate;
                if (candidate < probe) expectedPredecessor = candidate;
            }
            checkEquals(veb.successor(probe), expectedSuccessor, "final successor probe " + probe);
            checkEquals(veb.predecessor(probe), expectedPredecessor, "final predecessor probe " + probe);
        }

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
