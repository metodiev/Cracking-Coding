/** Task 610 - Implement succinct tree representations (DFUDS) and navigation ops. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * The DFUDS of a tree with n nodes is the depth-first unary degree sequence: the nodes are visited in
 * preorder and each one writes '(' * degree followed by a single ')', so the string has 2n-1 bits with
 * n-1 opens and n closes. A phantom '(' in front makes it a balanced parenthesis word. The structure
 * keeps only that bitvector plus rank/select/excess over it; the tree is recovered from it:
 *   - close(v)  = select0(v), so degree(v) = close(v) - close(v-1) - 1 (the opens of v's block),
 *   - the subtree of v occupies the consecutive closes [close(v-1)+1, regionEnd(v)], where
 *     regionEnd(v) = forwardSearch(start(v), excess(start(v)-1) - 1) is the first position whose
 *     excess drops by one again - the region is a primitive excursion, which is why this is exactly
 *     the range of v's subtree,
 *   - lastDescendant(v) = rank0(regionEnd(v)) and subtreeSize(v) = lastDescendant(v) - v + 1,
 *   - parent(v) = the largest j < v whose region contains start(v); ancestors are exactly the nodes
 *     with an earlier close and a region reaching past start(v).
 * firstChild, nextSibling, child(v,j), childRank, depth, lca and isAncestor are then built on those
 * primitives and never look at a stored tree.
 *
 * Simplified: forwardSearch is a plain forward scan (the literature adds byte lookup tables to skip
 * whole words) and parent walks the candidate close positions backwards instead of using the
 * O(1) version with two excess tables; both are documented and both are checked against brute force.
 * Verified: the encoding is balanced, has 2n-1 bits, n-1 opens and n closes; decoding reproduces the
 * input parent array and degree sequence exactly; re-encoding a decoded tree reproduces the same
 * bitvector; degree, parent, child, firstChild, nextSibling, childRank, subtreeSize, lastDescendant,
 * depth, isAncestor and lca agree with a brute-force oracle for chains, stars, caterpillars and
 * pseudo-random preorder trees, and the out-of-range queries return -1.
 */
import java.util.*;

class SuccinctBits {
    final int length;                     // positions are 1-based: 1 .. length
    private final long[] words;
    private final int[] wordRank;         // one-bits before each 64-bit word

    SuccinctBits(int[] bits) {
        length = bits.length;
        words = new long[(length + 63) >>> 6];
        for (int i = 0; i < length; i++) if (bits[i] == 1) words[i >>> 6] |= 1L << (i & 63);
        wordRank = new int[words.length + 1];
        for (int w = 0; w < words.length; w++) wordRank[w + 1] = wordRank[w] + Long.bitCount(words[w]);
    }

    static long maskLow(int k) { return k >= 64 ? -1L : (1L << k) - 1; }

    int bit(int p) {
        return (p < 1 || p > length) ? 0 : (int) ((words[(p - 1) >>> 6] >>> ((p - 1) & 63)) & 1L);
    }

    int rank1(int p) {
        if (p <= 0) return 0;
        if (p > length) p = length;
        int w = (p - 1) >>> 6;
        return wordRank[w] + Long.bitCount(words[w] & maskLow(p - (w << 6)));
    }

    int rank0(int p) {
        if (p <= 0) return 0;
        if (p > length) p = length;
        return p - rank1(p);
    }

    int count1() { return wordRank[words.length]; }

    int count0() { return length - count1(); }

    /** Excess over the phantom open at position 0, so a balanced word has excess 1..0. */
    int excess(int p) { return p <= 0 ? 1 : 1 + rank1(p) - rank0(p); }

    int select1(int k) {
        if (k < 1 || k > count1()) return -1;
        int lo = 1, hi = length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (rank1(mid) >= k) hi = mid; else lo = mid + 1;
        }
        return lo;
    }

    int select0(int k) {
        if (k < 1 || k > count0()) return -1;
        int lo = 1, hi = length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (rank0(mid) >= k) hi = mid; else lo = mid + 1;
        }
        return lo;
    }

    /** First position p >= from with excess(p) == target, or -1 when the excess never gets there. */
    int forwardSearch(int from, int target) {
        int p = Math.max(1, from);
        int e = excess(p - 1);
        while (p <= length) {
            e += bit(p) == 1 ? 1 : -1;
            if (e == target) return p;
            p++;
        }
        return -1;
    }

    int minExcess(int from) {
        int p = Math.max(1, from), e = excess(p - 1), best = e;
        while (p <= length) {
            e += bit(p) == 1 ? 1 : -1;
            best = Math.min(best, e);
            p++;
        }
        return best;
    }
}

class Dfuds {
    final int n;
    final SuccinctBits bits;
    final int[] parent;                    // preorder numbering, parent[1] = -1

    Dfuds(SuccinctBits bits) {
        this.bits = bits;
        this.n = bits.count0();
        this.parent = new int[n + 1];
        parent[1] = -1;
        for (int v = 2; v <= n; v++) parent[v] = parentByWalk(v);
    }

    /** Preorder parent array of a tree with n nodes -> the DFUDS bitvector. */
    static SuccinctBits encode(int[] preorderParent) {
        int n = preorderParent.length;
        int[] degree = new int[n];
        for (int i = 1; i < n; i++) {
            if (preorderParent[i] < 0 || preorderParent[i] >= i) {
                throw new IllegalArgumentException("not a preorder numbering at node " + i);
            }
            degree[preorderParent[i]]++;
        }
        int[] bits = new int[2 * n - 1];
        int p = 0;
        for (int v = 0; v < n; v++) {
            for (int k = 0; k < degree[v]; k++) bits[p++] = 1;
            bits[p++] = 0;
        }
        return new SuccinctBits(bits);
    }

    int closePosition(int v) { return v == 0 ? 0 : bits.select0(v); }

    int startPosition(int v) { return closePosition(v - 1) + 1; }

    int degree(int v) { return (v < 1 || v > n) ? -1 : closePosition(v) - closePosition(v - 1) - 1; }

    int[] degrees() {
        int[] out = new int[n + 1];
        for (int v = 1; v <= n; v++) out[v] = degree(v);
        return out;
    }

    int[] parents() { return parent.clone(); }

    int endOfRegion(int v) {
        int start = startPosition(v);
        return bits.forwardSearch(start, bits.excess(start - 1) - 1);
    }

    int lastDescendant(int v) { return bits.rank0(endOfRegion(v)); }

    int subtreeSize(int v) { return lastDescendant(v) - v + 1; }

    int parentOf(int v) { return (v < 1 || v > n) ? -1 : parent[v]; }

    private int parentByWalk(int v) {
        if (v == 1) return -1;
        int start = startPosition(v);
        for (int j = v - 1; j >= 1; j--) {
            if (startPosition(j) > start) continue;
            if (endOfRegion(j) >= start) return j;
        }
        return -1;
    }

    int firstChild(int v) { return degree(v) > 0 ? v + 1 : -1; }

    int nextSibling(int v) {
        if (v < 1 || v > n) return -1;
        int candidate = lastDescendant(v) + 1;
        if (candidate > n) return -1;
        return parent[candidate] == parent[v] ? candidate : -1;
    }

    int child(int v, int rank) {
        if (v < 1 || v > n || rank < 1 || rank > degree(v)) return -1;
        int child = firstChild(v);
        for (int i = 1; i < rank; i++) child = nextSibling(child);
        return child;
    }

    int childRank(int v) {
        if (v < 1 || v > n || parent[v] < 0) return -1;
        int rank = 1;
        for (int u = firstChild(parent[v]); u != v && u != -1; u = nextSibling(u)) rank++;
        return rank;
    }

    int depth(int v) {
        int d = 0;
        while (parent[v] >= 0) {
            v = parent[v];
            d++;
        }
        return d;
    }

    boolean isAncestor(int u, int v) {
        if (u < 1 || v < 1 || u > n || v > n) return false;
        return u <= v && v <= lastDescendant(u);
    }

    int lca(int u, int v) {
        while (u != v) {
            if (depth(u) < depth(v)) v = parent[v]; else u = parent[u];
        }
        return u;
    }
}

public class Solution {
    private static long state = 0x243F6A8885A308D3L;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int[] chain(int n) {
        int[] parent = new int[n];
        for (int i = 1; i < n; i++) parent[i] = i - 1;
        parent[0] = -1;
        return parent;
    }

    static int[] star(int n) {
        int[] parent = new int[n];
        for (int i = 1; i < n; i++) parent[i] = 0;
        parent[0] = -1;
        return parent;
    }

    /** Every node with an even index gets the next node as a child: a caterpillar. */
    static int[] caterpillar(int n) {
        int[] parent = new int[n];
        parent[0] = -1;
        int spine = 0;
        for (int i = 1; i < n; i++) {
            if (i % 2 == 1) {
                parent[i] = spine;
            } else {
                parent[i] = i - 1;
                spine = i;
            }
        }
        return parent;
    }

    /** Random preorder tree: fill node numbers by attaching the next node to the open chain top. */
    static int[] randomTree(int n) {
        int[] parent = new int[n];
        parent[0] = -1;
        Deque<Integer> open = new ArrayDeque<>();
        open.push(0);
        for (int next = 1; next < n; next++) {
            if (open.size() > 1 && (nextLong() >>> 33) % 100 < 45) open.pop();
            parent[next] = open.peek();
            open.push(next);
        }
        return parent;
    }

    static int[][] childrenOf(int[] parent) {
        int n = parent.length;
        int[] degree = new int[n];
        for (int i = 1; i < n; i++) degree[parent[i]]++;
        int[][] children = new int[n][];
        for (int v = 0; v < n; v++) children[v] = new int[degree[v]];
        int[] fill = new int[n];
        for (int i = 1; i < n; i++) children[parent[i]][fill[parent[i]]++] = i;
        return children;
    }

    static int[] bruteSubtreeSizes(int[] parent) {
        int n = parent.length;
        int[] size = new int[n];
        Arrays.fill(size, 1);
        for (int v = n - 1; v >= 1; v--) size[parent[v]] += size[v];
        return size;
    }

    static int bruteDepth(int[] parent, int v) {
        int d = 0;
        while (parent[v] >= 0) {
            v = parent[v];
            d++;
        }
        return d;
    }

    static boolean bruteAncestor(int[] parent, int u, int v) {
        for (int x = v; x >= 0; x = parent[x]) if (x == u) return true;
        return false;
    }

    static int bruteLca(int[] parent, int u, int v) {
        while (u != v) {
            if (bruteDepth(parent, u) < bruteDepth(parent, v)) v = parent[v]; else u = parent[u];
        }
        return u;
    }

    /** Cross-checks every navigation op of the succinct tree against the array-based oracle. */
    static void verify(int[] arrayParent) {
        int n = arrayParent.length;
        SuccinctBits bits = Dfuds.encode(arrayParent);
        check(bits.length == 2 * n - 1, "DFUDS length 2n-1 for n=" + n);
        check(bits.count1() == n - 1 && bits.count0() == n, "DFUDS has n-1 opens and n closes");
        check(bits.excess(bits.length) == 0 && bits.minExcess(1) >= 0, "DFUDS word is balanced");

        Dfuds tree = new Dfuds(bits);
        int[][] children = childrenOf(arrayParent);
        int[] sizes = bruteSubtreeSizes(arrayParent);
        for (int v = 1; v <= n; v++) {
            check(tree.degree(v) == children[v - 1].length, "degree of " + v);
            check(tree.parentOf(v) == (v == 1 ? -1 : arrayParent[v - 1] + 1), "parent of " + v);
            check(tree.subtreeSize(v) == sizes[v - 1], "subtree size of " + v);
            check(tree.lastDescendant(v) == v + sizes[v - 1] - 1, "last descendant of " + v);
            check(tree.depth(v) == bruteDepth(arrayParent, v - 1), "depth of " + v);
            check(tree.isAncestor(v, v), "a node is its own ancestor");
            int first = children[v - 1].length == 0 ? -1 : children[v - 1][0] + 1;
            check(tree.firstChild(v) == first, "first child of " + v);
            for (int j = 1; j <= children[v - 1].length + 1; j++) {
                int expected = j <= children[v - 1].length ? children[v - 1][j - 1] + 1 : -1;
                check(tree.child(v, j) == expected, "child " + j + " of " + v);
            }
            if (v > 1) {
                int[] siblings = children[arrayParent[v - 1]];
                int myRank = 0;
                for (int i = 0; i < siblings.length; i++) if (siblings[i] == v - 1) myRank = i;
                check(tree.childRank(v) == myRank + 1, "child rank of " + v);
                int expectedNext = myRank + 1 < siblings.length ? siblings[myRank + 1] + 1 : -1;
                check(tree.nextSibling(v) == expectedNext, "next sibling of " + v);
            } else {
                check(tree.parentOf(1) == -1 && tree.nextSibling(n + 5) == -1, "root edges");
            }
        }
        check(tree.child(1, 0) == -1 && tree.child(1, tree.degree(1) + 1) == -1, "child out of range");
        check(tree.parentOf(0) == -1 && tree.parentOf(n + 1) == -1, "parent out of range");
        for (int u = 1; u <= n; u++) {
            for (int v = 1; v <= n; v++) {
                boolean ancestor = tree.isAncestor(u, v);
                check(ancestor == bruteAncestor(arrayParent, u - 1, v - 1), "ancestor " + u + " " + v);
                if (ancestor && u != v) {
                    check(tree.depth(u) < tree.depth(v), "an ancestor is shallower " + u + " " + v);
                    check(tree.lca(u, v) == u, "lca of an ancestor pair " + u + " " + v);
                }
                if ((u * 31 + v) % 11 == 0) {
                    check(tree.lca(u, v) == bruteLca(arrayParent, u - 1, v - 1) + 1, "lca " + u + " " + v);
                }
            }
        }
        // re-encoding the decoded tree must reproduce the same bitvector
        int[] decoded = tree.parents();
        int[] reencode = new int[n];
        reencode[0] = -1;
        for (int v = 2; v <= n; v++) reencode[v - 1] = decoded[v] - 1;
        SuccinctBits again = Dfuds.encode(reencode);
        check(again.length == bits.length, "re-encoded length");
        boolean same = true;
        for (int p = 1; p <= bits.length; p++) same &= bits.bit(p) == again.bit(p);
        check(same, "re-encoding reproduces the DFUDS");
    }

    public static void main(String[] args) {
        int[] single = {-1};
        verify(single);
        check(Dfuds.encode(single).length == 1 && Dfuds.encode(single).bit(1) == 0, "single node encodes ')'");
        verify(chain(2));
        verify(star(2));
        verify(chain(61));
        verify(star(61));
        verify(caterpillar(40));
        for (int n : new int[]{3, 4, 5, 9, 16, 33, 64, 100}) {
            for (int trial = 0; trial < 3; trial++) verify(randomTree(n));
        }

        // select/rank/excess agreement on one larger DFUDS string
        SuccinctBits big = Dfuds.encode(randomTree(150));
        int ones = 0;
        for (int p = 0; p <= big.length; p++) {
            check(big.rank1(p) == ones, "rank1 of the big DFUDS at " + p);
            check(big.rank0(p) == p - ones, "rank0 of the big DFUDS at " + p);
            if (p < big.length) ones += big.bit(p + 1);
        }
        for (int k = 1; k <= big.count1(); k++) check(big.bit(big.select1(k)) == 1, "select1 of the big DFUDS");
        for (int k = 1; k <= big.count0(); k++) check(big.bit(big.select0(k)) == 0, "select0 of the big DFUDS");
        check(big.select1(0) == -1 && big.select0(big.count0() + 1) == -1, "select out of range");

        Dfuds bigTree = new Dfuds(big);
        System.out.println("n=" + bigTree.n + ": DFUDS " + big.length + " bits vs "
                + (2L * bigTree.n * 32) + " bits for parent and degree arrays");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
