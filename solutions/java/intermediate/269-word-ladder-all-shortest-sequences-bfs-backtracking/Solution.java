/** Task 269 - Word ladder all shortest sequences (BFS + backtracking). (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

public class Solution {
    /**
     * All shortest transformation sequences (each step changes one letter; every
     * intermediate word must be in the word list). BFS over the wildcard bucket index
     * assigns minimum distances; the DFS then walks the LAYERED DAG, following only
     * edges u -> v with dist[v] == dist[u] + 1. That keeps every shortest path exactly
     * once (dist strictly increases, so no word repeats within a path) and prunes all
     * non-shortest continuations. Returned sequences are sorted element-wise
     * lexicographically so hash-iteration order never leaks into the result.
     * [] when end is not in the list, the lengths differ, or end is unreachable;
     * [[begin]] when begin == end (and end is in the list).
     */
    static List<List<String>> findLadders(String begin, String end, List<String> wordList) {
        Set<String> words = new HashSet<>(wordList);
        List<List<String>> result = new ArrayList<>();
        if (!words.contains(end) || begin.length() != end.length()) return result;
        if (begin.equals(end)) {
            result.add(new ArrayList<>(Arrays.asList(begin)));
            return result;
        }
        int length = begin.length();
        Map<String, List<String>> buckets = new HashMap<>();
        for (String word : words) {
            if (word.length() != length) continue;
            for (int i = 0; i < length; i++) {
                String pattern = word.substring(0, i) + "*" + word.substring(i + 1);
                buckets.computeIfAbsent(pattern, k -> new ArrayList<>()).add(word);
            }
        }
        Map<String, Integer> dist = new HashMap<>();        // BFS: minimum distance per word
        dist.put(begin, 0);
        Deque<String> queue = new ArrayDeque<>();
        queue.add(begin);
        while (!queue.isEmpty()) {
            String word = queue.poll();
            for (int i = 0; i < length; i++) {
                List<String> neighbours = buckets.get(word.substring(0, i) + "*" + word.substring(i + 1));
                if (neighbours == null) continue;
                for (String nxt : neighbours)
                    if (!dist.containsKey(nxt)) {
                        dist.put(nxt, dist.get(word) + 1);
                        queue.add(nxt);
                    }
            }
        }
        if (!dist.containsKey(end)) return result;
        int target = dist.get(end);
        List<String> path = new ArrayList<>();
        path.add(begin);
        dfs(begin, end, target, length, buckets, dist, path, result);
        sortPaths(result);
        return result;
    }

    private static void dfs(String word, String end, int target, int length,
                            Map<String, List<String>> buckets, Map<String, Integer> dist,
                            List<String> path, List<List<String>> result) {
        if (word.equals(end)) {
            result.add(new ArrayList<>(path));
            return;
        }
        if (dist.get(word) >= target) return;
        for (int i = 0; i < length; i++) {
            List<String> neighbours = buckets.get(word.substring(0, i) + "*" + word.substring(i + 1));
            if (neighbours == null) continue;
            for (String nxt : neighbours)
                if (dist.get(nxt) != null && dist.get(nxt) == dist.get(word) + 1) {
                    path.add(nxt);
                    dfs(nxt, end, target, length, buckets, dist, path, result);
                    path.remove(path.size() - 1);
                }
        }
    }

    private static void sortPaths(List<List<String>> paths) {
        paths.sort((p, q) -> {
            int common = Math.min(p.size(), q.size());
            for (int i = 0; i < common; i++) {
                int c = p.get(i).compareTo(q.get(i));
                if (c != 0) return c;
            }
            return Integer.compare(p.size(), q.size());
        });
    }

    /** Enumerate all simple paths, keep the shortest; cross-check only. */
    static List<List<String>> findLaddersBrute(String begin, String end, List<String> wordList) {
        Set<String> words = new HashSet<>(wordList);
        List<List<String>> found = new ArrayList<>();
        if (!words.contains(end) || begin.length() != end.length()) return found;
        if (begin.equals(end)) {
            found.add(new ArrayList<>(Arrays.asList(begin)));
            return found;
        }
        List<String> path = new ArrayList<>();
        path.add(begin);
        Set<String> seen = new HashSet<>();
        seen.add(begin);
        dfsBrute(begin, end, words, path, seen, found);
        if (found.isEmpty()) return found;
        int best = Integer.MAX_VALUE;
        for (List<String> p : found) best = Math.min(best, p.size());
        List<List<String>> result = new ArrayList<>();
        for (List<String> p : found) if (p.size() == best) result.add(p);
        sortPaths(result);
        return result;
    }

    private static void dfsBrute(String word, String end, Set<String> words,
                                 List<String> path, Set<String> seen, List<List<String>> found) {
        if (word.equals(end)) {
            found.add(new ArrayList<>(path));
            return;
        }
        if (path.size() > words.size() + 1) return;         // simple paths cannot exceed this
        for (String nxt : words) {
            if (seen.contains(nxt) || nxt.length() != word.length()) continue;
            int diff = 0;
            for (int i = 0; i < word.length() && diff <= 1; i++) if (word.charAt(i) != nxt.charAt(i)) diff++;
            if (diff == 1) {
                seen.add(nxt);
                path.add(nxt);
                dfsBrute(nxt, end, words, path, seen, found);
                path.remove(path.size() - 1);
                seen.remove(nxt);
            }
        }
    }

    public static void main(String[] args) {
        checkPaths(findLadders("hit", "cog", Arrays.asList("hot", "dot", "dog", "lot", "log", "cog")),
                new String[][]{{"hit", "hot", "dot", "dog", "cog"}, {"hit", "hot", "lot", "log", "cog"}},
                "classic two sequences");
        checkPaths(findLadders("red", "tax", Arrays.asList("ted", "tex", "red", "tax", "tad", "den", "rex", "pee")),
                new String[][]{{"red", "rex", "tex", "tax"}, {"red", "ted", "tad", "tax"}, {"red", "ted", "tex", "tax"}},
                "red -> tax three sequences");
        checkPaths(findLadders("ab", "ba", Arrays.asList("bb", "aa", "ba")),
                new String[][]{{"ab", "aa", "ba"}, {"ab", "bb", "ba"}}, "two short paths");
        checkPaths(findLadders("hit", "cog", Arrays.asList("hot", "dot", "dog", "lot", "log")),
                new String[][]{}, "unreachable");
        checkPaths(findLadders("hot", "hot", Arrays.asList("hot")),
                new String[][]{{"hot"}}, "same word");
        checkPaths(findLadders("a", "c", Arrays.asList("a", "b", "c")),
                new String[][]{{"a", "c"}}, "one-letter direct hop");
        checkPaths(findLadders("ab", "abc", Arrays.asList("abc")), new String[][]{}, "mismatched lengths");
        checkPaths(findLadders("hot", "hot", Arrays.asList("dog")), new String[][]{}, "end not in list wins");
        checkPaths(findLadders("lost", "cost", Arrays.asList("most", "fost", "cost", "cast")),
                new String[][]{{"lost", "cost"}}, "direct hop");

        Random rng = new Random(269);
        for (int test = 0; test < 200; test++) {            // cross-check vs the path-enumeration brute
            Set<String> pool = new TreeSet<>();
            int poolSize = 1 + rng.nextInt(6);
            while (pool.size() < poolSize)
                pool.add("" + (char) ('a' + rng.nextInt(3)) + (char) ('a' + rng.nextInt(3)) + (char) ('a' + rng.nextInt(3)));
            List<String> wordList = new ArrayList<>(pool);
            String begin = "" + (char) ('a' + rng.nextInt(3)) + (char) ('a' + rng.nextInt(3)) + (char) ('a' + rng.nextInt(3));
            String end = rng.nextDouble() < 0.5 ? wordList.get(rng.nextInt(wordList.size()))
                    : "" + (char) ('a' + rng.nextInt(3)) + (char) ('a' + rng.nextInt(3)) + (char) ('a' + rng.nextInt(3));
            check(findLadders(begin, end, wordList).equals(findLaddersBrute(begin, end, wordList)),
                    "random agreement begin=" + begin + " end=" + end);
        }
        System.out.println("PASS");
    }

    static void checkPaths(List<List<String>> got, String[][] expected, String label) {
        boolean ok = got.size() == expected.length;
        for (int i = 0; ok && i < expected.length; i++) {
            ok = got.get(i).equals(Arrays.asList(expected[i]));
            if (!ok) label = label + " (row " + i + " got=" + got.get(i) + ")";
        }
        check(ok, label);
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
