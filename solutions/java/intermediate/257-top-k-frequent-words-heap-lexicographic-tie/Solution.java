/** Task 257 - Top K frequent words (heap + lexicographic tie). (Tier: Intermediate, CTCI Ch 10) */
import java.util.*;

public class Solution {
    /**
     * LeetCode 692 ordering: more frequent first, ties broken lexicographically ascending.
     * The heap root is the worst kept word (lower count, or on a tie the lexicographically
     * larger word) so eviction is a single heap poll. O(n + d log k) for d distinct words.
     */
    static List<String> topKFrequentWords(String[] words, int k) {
        Map<String, Integer> freq = new HashMap<>();
        for (String w : words) freq.merge(w, 1, Integer::sum);
        List<String> out = new ArrayList<>();
        if (k <= 0) return out;
        PriorityQueue<String> heap = new PriorityQueue<>((a, b) -> {
            int cmp = Integer.compare(freq.get(a), freq.get(b));
            if (cmp != 0) return cmp;                   // lower count = worse = first
            return b.compareTo(a);                       // on a tie the larger word is worse
        });
        for (String w : freq.keySet()) {
            if (heap.size() < k) {
                heap.add(w);
            } else if (isBetter(w, heap.peek(), freq)) {
                heap.poll();
                heap.add(w);
            }
        }
        out.addAll(heap);
        out.sort((a, b) -> freq.get(a).equals(freq.get(b))
                ? a.compareTo(b) : freq.get(b) - freq.get(a));
        return out;
    }

    static boolean isBetter(String candidate, String worst, Map<String, Integer> freq) {
        int cand = freq.get(candidate);
        int w = freq.get(worst);
        if (cand != w) return cand > w;
        return candidate.compareTo(worst) < 0;
    }

    public static void main(String[] args) {
        String[] classic = {"i", "love", "leetcode", "i", "love", "coding"};
        check(topKFrequentWords(classic, 1).toString().equals("[i]"), "classic k=1");
        check(topKFrequentWords(classic, 2).toString().equals("[i, love]"), "classic k=2");
        check(topKFrequentWords(classic, 3).toString().equals("[i, love, coding]"), "classic k=3");
        check(topKFrequentWords(classic, 4).toString().equals("[i, love, coding, leetcode]"), "classic k=4");
        check(topKFrequentWords(classic, 9).toString().equals("[i, love, coding, leetcode]"), "k > distinct");
        check(topKFrequentWords(classic, 0).isEmpty(), "k=0");

        String[] other = {"the", "day", "is", "sunny", "the", "the", "sunny", "is", "is"};
        check(topKFrequentWords(other, 1).toString().equals("[is]"), "is beats the on ties");
        check(topKFrequentWords(other, 2).toString().equals("[is, the]"), "is, the");
        check(topKFrequentWords(other, 4).toString().equals("[is, the, sunny, day]"), "full order");

        check(topKFrequentWords(new String[]{"a", "aa", "aaa"}, 2).toString().equals("[a, aa]"), "lex ties");
        check(topKFrequentWords(new String[]{"a", "aa", "aaa"}, 3).toString().equals("[a, aa, aaa]"), "all three");
        String[] mixed = {"b", "b", "a", "a", "c", "c", "d"};       // a:2 b:2 c:2 d:1
        check(topKFrequentWords(mixed, 1).toString().equals("[a]"), "eviction tie k=1");
        check(topKFrequentWords(mixed, 2).toString().equals("[a, b]"), "eviction tie k=2");
        check(topKFrequentWords(mixed, 3).toString().equals("[a, b, c]"), "eviction tie k=3");
        check(topKFrequentWords(new String[]{"only"}, 1).toString().equals("[only]"), "single word");
        check(topKFrequentWords(new String[]{}, 3).isEmpty(), "empty input");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
