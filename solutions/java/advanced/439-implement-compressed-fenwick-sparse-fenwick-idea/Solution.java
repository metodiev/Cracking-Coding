/** Task 439 - Implement compressed Fenwick / sparse Fenwick idea. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A Fenwick tree that is touched at only a few coordinates does not need a dense array: the sparse
 * version keeps a map of just the cells an update really visits (i, i + lowbit(i), ...) and treats
 * every untouched cell as 0 while walking a prefix query (i, i - lowbit(i), ...). Documented setup:
 * SIZE = 2^20 = 1048576, coordinates 1..SIZE, updates propagate while i <= SIZE. A dense Fenwick of
 * the same size is the oracle: LCG updates are applied to both and every prefix query must agree,
 * with a brute-force sweep over the recorded updates as a third opinion. The touched-cell count is
 * asserted to stay within 21 * (#updates) - at most log2(SIZE) = 20 cells are visited per update -
 * and far below SIZE, which is the evidence that the representation really is compressed.
 */
import java.util.*;

class SparseFenwick {
    final int size;
    final Map<Integer, Long> tree = new HashMap<>();

    SparseFenwick(int size) {
        this.size = size;
    }

    void add(int index, long delta) {
        int i = index;
        while (i <= size) {
            tree.merge(i, delta, Long::sum);
            i += i & -i;
        }
    }

    long prefix(int index) {
        long total = 0;
        int i = index;
        while (i > 0) {
            Long value = tree.get(i);
            if (value != null) total += value;
            i -= i & -i;
        }
        return total;
    }

    int touched() {
        return tree.size();
    }
}

class DenseFenwick {
    final int size;
    final long[] tree;

    DenseFenwick(int size) {
        this.size = size;
        this.tree = new long[size + 1];
    }

    void add(int index, long delta) {
        int i = index;
        while (i <= size) {
            tree[i] += delta;
            i += i & -i;
        }
    }

    long prefix(int index) {
        long total = 0;
        int i = index;
        while (i > 0) {
            total += tree[i];
            i -= i & -i;
        }
        return total;
    }
}

public class Solution {
    static final int SIZE = 1 << 20;

    public static void main(String[] args) {
        SparseFenwick sparse = new SparseFenwick(16);
        DenseFenwick dense = new DenseFenwick(16);
        sparse.add(1, 5);
        sparse.add(3, 2);
        sparse.add(16, 4);
        dense.add(1, 5);
        dense.add(3, 2);
        dense.add(16, 4);
        check(sparse.prefix(0) == 0, "empty prefix");
        check(sparse.prefix(1) == 5, "prefix 1");
        check(sparse.prefix(2) == 5, "prefix 2");
        check(sparse.prefix(3) == 7, "prefix 3");
        check(sparse.prefix(15) == 7, "prefix 15");
        check(sparse.prefix(16) == 11, "prefix 16");
        check(sparse.touched() == 6, "touched cells " + sparse.touched());
        for (int index = 0; index <= 16; index++) {
            check(sparse.prefix(index) == dense.prefix(index), "small agreement " + index);
        }

        sparse = new SparseFenwick(SIZE);
        dense = new DenseFenwick(SIZE);
        check(sparse.prefix(0) == 0 && sparse.prefix(SIZE) == 0, "large empty");
        check(sparse.touched() == 0, "large empty touched");

        List<int[]> recorded = new ArrayList<>();
        int seed = 4390;
        int queries = 0;
        int updates = 0;
        for (int step = 0; step < 800; step++) {
            seed = (seed * 75 + 74) % 65537;
            seed = (seed * 75 + 74) % 65537;
            int coord = 1 + (seed * 1024 + (seed >> 4)) % (SIZE - 1);
            seed = (seed * 75 + 74) % 65537;
            if (step % 2 == 0) {
                long delta = seed % 21 - 10;
                sparse.add(coord, delta);
                dense.add(coord, delta);
                recorded.add(new int[]{coord, (int) delta});
                updates++;
            } else {
                check(sparse.prefix(coord) == dense.prefix(coord), "random agreement step " + step);
                queries++;
            }
        }
        check(updates == 400 && queries == 400, "operation counts");

        for (int coord : new int[]{1, 12345, SIZE / 2, SIZE - 1, SIZE}) {
            long expected = 0;
            for (int[] pair : recorded) {
                if (pair[0] <= coord) expected += pair[1];
            }
            check(sparse.prefix(coord) == dense.prefix(coord) && sparse.prefix(coord) == expected,
                    "brute agreement " + coord);
        }
        check(sparse.touched() >= 1, "some cells touched");
        check(sparse.touched() <= 21 * updates, "compressed within bound " + sparse.touched());
        check(sparse.touched() < SIZE / 100, "compressed far below dense " + sparse.touched());
        long total = 0;
        for (int[] pair : recorded) total += pair[1];
        check(sparse.prefix(SIZE) == total, "total sum");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
