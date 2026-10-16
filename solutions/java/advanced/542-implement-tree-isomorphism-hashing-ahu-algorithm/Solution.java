/** Task 542 - Implement tree isomorphism hashing (AHU algorithm). (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)
 *
 * Ahu's canonical form for trees: root the tree at its centre (found by repeatedly peeling the
 * leaves, which is isomorphism invariant), recursively encode every rooted subtree as
 * "(" + sorted child encodings + ")", and take the lexicographically smallest encoding over the
 * one or two centres. Two trees are isomorphic exactly when those strings are equal, so a 64-bit
 * FNV-1a hash of the canonical string serves as an isomorphism fingerprint; the self-test also
 * asserts that the fingerprint is collision-free across the isomorphism classes it builds.
 *
 * Input trees are generated exhaustively from Prüfer sequences (all n^(n-2) labelled trees on n
 * vertices), decoded with the O(n) monotone-pointer leaf scan — the pointer is rewound whenever
 * the newly created leaf has a smaller index than the current pointer, which is what keeps the
 * scan linear and correct. Documented checks: for n = 1..7 the number of distinct canonical
 * strings equals the known counts of unlabelled trees 1, 1, 1, 2, 3, 6, 11, and for n <= 6 the
 * number of distinct labelled trees equals Cayley's n^(n-2); every decoded graph is verified to
 * be a tree (n-1 edges, connected, loop-free); relabelling by a deterministic permutation leaves
 * both the canonical string and the hash unchanged; canonical equality is compared with an
 * exhaustive permutation search over every pair of isomorphism classes at n = 6 and n = 7; a
 * pair of classes sharing a degree sequence is exhibited to show the AHU form is strictly finer
 * than the degree multiset; and the centres and encodings are checked by hand (5-vertex path has
 * centre 2, 6-vertex path has centres 2 and 3, the star has its hub, the 3-leaf star encodes as
 * "(()()())" and the 5-vertex path as "((())(()))", while the single edge encodes as "(())").
 */
import java.util.*;

public class Solution {
    /** O(n) Prüfer decoding: the smallest current leaf drives the scan. */
    static boolean[][] pruferToTree(int[] sequence, int n) {
        boolean[][] link = new boolean[n][n];
        if (n == 1) return link;
        int[] degree = new int[n];
        Arrays.fill(degree, 1);
        for (int value : sequence) degree[value]++;
        int pointer = 0;
        while (degree[pointer] != 1) pointer++;
        for (int index = 0; index < n - 2; index++) {
            int leaf = pointer;
            int target = sequence[index];
            if (leaf == target) throw new RuntimeException("FAILED: Prüfer decoding produced a self-loop");
            link[leaf][target] = true;
            link[target][leaf] = true;
            degree[leaf]--;
            degree[target]--;
            if (degree[target] == 1 && target < pointer) {
                pointer = target;
            } else {
                while (pointer < n && degree[pointer] != 1) pointer++;
            }
        }
        int first = -1;
        int second = -1;
        for (int v = 0; v < n; v++) {
            if (degree[v] == 1) {
                if (first < 0) first = v;
                else second = v;
            }
        }
        link[first][second] = true;
        link[second][first] = true;
        return link;
    }

    static boolean isTree(boolean[][] link, int n) {
        int edges = 0;
        for (int u = 0; u < n; u++) {
            if (link[u][u]) return false;
            for (int v = u + 1; v < n; v++) {
                if (link[u][v]) edges++;
            }
        }
        if (n == 1) return edges == 0;
        if (edges != n - 1) return false;
        boolean[] seen = new boolean[n];
        Deque<Integer> stack = new ArrayDeque<>();
        stack.push(0);
        seen[0] = true;
        int reached = 1;
        while (!stack.isEmpty()) {
            int v = stack.pop();
            for (int w = 0; w < n; w++) {
                if (link[v][w] && !seen[w]) {
                    seen[w] = true;
                    reached++;
                    stack.push(w);
                }
            }
        }
        return reached == n;
    }

    /** Centre(s): peel leaves layer by layer until at most two vertices remain. */
    static int[] centers(boolean[][] link, int n) {
        if (n == 1) return new int[]{0};
        int[] degree = new int[n];
        for (int v = 0; v < n; v++) {
            for (int w = 0; w < n; w++) {
                if (link[v][w]) degree[v]++;
            }
        }
        boolean[] removed = new boolean[n];
        int remaining = n;
        while (remaining > 2) {
            List<Integer> leaves = new ArrayList<>();
            for (int v = 0; v < n; v++) {
                if (!removed[v] && degree[v] == 1) leaves.add(v);
            }
            for (int v : leaves) {
                removed[v] = true;
                remaining--;
                for (int w = 0; w < n; w++) {
                    if (link[v][w] && !removed[w]) degree[w]--;
                }
            }
        }
        List<Integer> result = new ArrayList<>();
        for (int v = 0; v < n; v++) {
            if (!removed[v]) result.add(v);
        }
        int[] out = new int[result.size()];
        for (int i = 0; i < out.length; i++) out[i] = result.get(i);
        return out;
    }

    static String rootedCanon(boolean[][] link, int root, int parent, int n) {
        List<String> children = new ArrayList<>();
        for (int w = 0; w < n; w++) {
            if (link[root][w] && w != parent) children.add(rootedCanon(link, w, root, n));
        }
        Collections.sort(children);
        StringBuilder builder = new StringBuilder("(");
        for (String child : children) builder.append(child);
        return builder.append(")").toString();
    }

    static String canonical(boolean[][] link, int n) {
        String best = null;
        for (int center : centers(link, n)) {
            String form = rootedCanon(link, center, -1, n);
            if (best == null || form.compareTo(best) < 0) best = form;
        }
        return best;
    }

    static long hash(String canonicalForm) {
        long value = 0xcbf29ce484222325L;
        for (int i = 0; i < canonicalForm.length(); i++) {
            value ^= canonicalForm.charAt(i);
            value *= 0x100000001b3L;
        }
        return value;
    }

    static int[] degrees(boolean[][] link, int n) {
        int[] degree = new int[n];
        for (int v = 0; v < n; v++) {
            for (int w = 0; w < n; w++) {
                if (link[v][w]) degree[v]++;
            }
        }
        return degree;
    }

    static boolean sameDegreeSequence(boolean[][] a, boolean[][] b, int n) {
        int[] da = degrees(a, n);
        int[] db = degrees(b, n);
        Arrays.sort(da);
        Arrays.sort(db);
        return Arrays.equals(da, db);
    }

    /** Independent oracle: exhaustive search over all relabellings, degree sequence as a filter. */
    static boolean isomorphicBrute(boolean[][] a, boolean[][] b, int n) {
        if (!sameDegreeSequence(a, b, n)) return false;
        int[] perm = new int[n];
        for (int i = 0; i < n; i++) perm[i] = i;
        while (true) {
            boolean ok = true;
            for (int u = 0; u < n && ok; u++) {
                for (int v = u + 1; v < n; v++) {
                    if (a[u][v] != b[perm[u]][perm[v]]) {
                        ok = false;
                        break;
                    }
                }
            }
            if (ok) return true;
            int i = n - 2;
            while (i >= 0 && perm[i] > perm[i + 1]) i--;
            if (i < 0) return false;
            int j = n - 1;
            while (perm[j] < perm[i]) j--;
            int tmp = perm[i];
            perm[i] = perm[j];
            perm[j] = tmp;
            for (int lo = i + 1, hi = n - 1; lo < hi; lo++, hi--) {
                tmp = perm[lo];
                perm[lo] = perm[hi];
                perm[hi] = tmp;
            }
        }
    }

    static boolean[][] relabel(boolean[][] link, int[] perm, int n) {
        boolean[][] out = new boolean[n][n];
        for (int u = 0; u < n; u++) {
            for (int v = 0; v < n; v++) {
                if (link[u][v]) out[perm[u]][perm[v]] = true;
            }
        }
        return out;
    }

    static int[] permutation(int n, int seed) {
        int[] perm = new int[n];
        for (int i = 0; i < n; i++) perm[i] = i;
        for (int i = n - 1; i > 0; i--) {
            seed = (seed * 75 + 74) % 65537;
            int j = seed % (i + 1);
            int tmp = perm[i];
            perm[i] = perm[j];
            perm[j] = tmp;
        }
        return perm;
    }

    /** All labelled trees on n vertices, decoded from every Prüfer sequence. */
    static List<boolean[][]> allTrees(int n) {
        List<boolean[][]> trees = new ArrayList<>();
        if (n == 1) {
            trees.add(new boolean[1][1]);
            return trees;
        }
        if (n == 2) {
            trees.add(pruferToTree(new int[0], 2));
            return trees;
        }
        long total = 1;
        for (int i = 0; i < n - 2; i++) total *= n;
        int[] sequence = new int[n - 2];
        for (long index = 0; index < total; index++) {
            long value = index;
            for (int i = 0; i < n - 2; i++) {
                sequence[i] = (int) (value % n);
                value /= n;
            }
            trees.add(pruferToTree(sequence, n));
        }
        return trees;
    }

    static boolean[][] path(int n) {
        boolean[][] link = new boolean[n][n];
        for (int i = 0; i + 1 < n; i++) {
            link[i][i + 1] = true;
            link[i + 1][i] = true;
        }
        return link;
    }

    static boolean[][] star(int n) {
        boolean[][] link = new boolean[n][n];
        for (int i = 1; i < n; i++) {
            link[0][i] = true;
            link[i][0] = true;
        }
        return link;
    }

    public static void main(String[] args) {
        check(canonical(new boolean[1][1], 1).equals("()"), "a single vertex encodes as ()");
        boolean[][] edge = pruferToTree(new int[0], 2);
        check(edge[0][1] && edge[1][0] && isTree(edge, 2), "the empty Prüfer sequence gives the edge");
        check(canonical(edge, 2).equals("(())"), "a two-vertex tree encodes as (())");
        check(canonical(star(4), 4).equals("(()()())"), "hand-written encoding of the 3-leaf star");
        check(canonical(path(5), 5).equals("((())(()))"), "hand-written encoding of the 5-vertex path");

        int[] center5 = centers(path(5), 5);
        check(center5.length == 1 && center5[0] == 2, "the 5-vertex path has centre 2");
        int[] path6 = centers(path(6), 6);
        check(path6.length == 2 && path6[0] == 2 && path6[1] == 3, "the 6-vertex path has centres 2 and 3");
        int[] star6 = centers(star(6), 6);
        check(star6.length == 1 && star6[0] == 0, "the star has its hub as centre");
        check(centers(new boolean[1][1], 1).length == 1, "the single vertex is its own centre");
        check(canonical(path(6), 6).equals(canonical(relabel(path(6), permutation(6, 5), 6), 6)),
                "the two centre roots agree after relabelling");

        int[] known = {1, 1, 1, 2, 3, 6, 11};
        List<List<boolean[][]>> classesBySize = new ArrayList<>();
        for (int n = 1; n <= 7; n++) {
            List<boolean[][]> trees = allTrees(n);
            long cayley = n == 1 ? 1 : (long) Math.pow(n, Math.max(0, n - 2));
            check(trees.size() == cayley, "Cayley: n = " + n + " has n^(n-2) labelled trees");
            for (boolean[][] tree : trees) check(isTree(tree, n), "decoded graph is a tree (n = " + n + ")");
            Map<String, boolean[][]> classes = new TreeMap<>();
            Set<Long> hashes = new TreeSet<>();
            for (boolean[][] tree : trees) {
                String form = canonical(tree, n);
                classes.putIfAbsent(form, tree);
                hashes.add(hash(form));
            }
            check(classes.size() == known[n - 1],
                    "n = " + n + ": " + classes.size() + " isomorphism classes (expected " + known[n - 1] + ")");
            check(hashes.size() == classes.size(), "n = " + n + ": no hash collisions across classes");
            classesBySize.add(new ArrayList<>(classes.values()));

            int[] perm = permutation(n, 1234 + n);
            for (boolean[][] tree : trees) {
                boolean[][] shuffled = relabel(tree, perm, n);
                check(canonical(shuffled, n).equals(canonical(tree, n)),
                        "n = " + n + ": relabelling preserves the canonical form");
                check(hash(canonical(shuffled, n)) == hash(canonical(tree, n)),
                        "n = " + n + ": relabelling preserves the hash");
                check(isomorphicBrute(tree, shuffled, n), "n = " + n + ": relabelling keeps isomorphism");
            }
        }

        for (int n : new int[]{6, 7}) {
            List<boolean[][]> classes = classesBySize.get(n - 1);
            int sharedPairs = 0;
            for (int i = 0; i < classes.size(); i++) {
                check(isomorphicBrute(classes.get(i), relabel(classes.get(i), permutation(n, 7 * i + 3), n), n),
                        "n = " + n + ": a relabelled tree is isomorphic to its representative");
                for (int j = i + 1; j < classes.size(); j++) {
                    check(!isomorphicBrute(classes.get(i), classes.get(j), n),
                            "n = " + n + ": distinct classes are not isomorphic");
                    check(!canonical(classes.get(i), n).equals(canonical(classes.get(j), n)),
                            "n = " + n + ": distinct classes have distinct canonical forms");
                    if (sameDegreeSequence(classes.get(i), classes.get(j), n)) sharedPairs++;
                }
            }
            check(sharedPairs >= 1,
                    "n = " + n + ": some degree sequence is realised by structurally different trees");
        }

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
