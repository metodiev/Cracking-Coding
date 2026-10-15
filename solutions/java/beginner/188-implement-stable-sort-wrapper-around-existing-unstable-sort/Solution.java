/** Task 188 - Implement stable sort wrapper around existing unstable sort. (Tier: Beginner, CTCI Beyond — Software Engineering & Tooling) */
import java.util.*;

public class Solution {
    static class Record {
        final int key;
        final String tag;
        final int order;      // original position, -1 when not decorated

        Record(int key, String tag) {
            this(key, tag, -1);
        }

        Record(int key, String tag, int order) {
            this.key = key;
            this.tag = tag;
            this.order = order;
        }
    }

    /**
     * JDK sorting of objects is TimSort (already stable) and Arrays.sort for primitives
     * cannot carry a payload, so the *unstable* black box used here is a hand-rolled Lomuto
     * quicksort: its strict "< 0" comparison lets records with equal keys swap places.
     */
    static List<Record> unstableSort(List<Record> records) {
        List<Record> sorted = new ArrayList<>(records);
        quick(sorted, 0, sorted.size() - 1, (a, b) -> Integer.compare(a.key, b.key));
        return sorted;
    }

    /** Stable wrapper: decorate with the original index, sort (key, order) pairs, undecorate. */
    static List<Record> stableSort(List<Record> records) {
        List<Record> decorated = new ArrayList<>();
        for (int i = 0; i < records.size(); i++) {
            Record record = records.get(i);
            decorated.add(new Record(record.key, record.tag, i));
        }
        quick(decorated, 0, decorated.size() - 1,
                (a, b) -> a.key != b.key ? Integer.compare(a.key, b.key) : Integer.compare(a.order, b.order));
        List<Record> result = new ArrayList<>();
        for (Record record : decorated) result.add(new Record(record.key, record.tag));
        return result;
    }

    private static void quick(List<Record> records, int lo, int hi, Comparator<Record> cmp) {
        while (lo < hi) {
            Record pivot = records.get(hi);
            int store = lo;
            for (int j = lo; j < hi; j++) {
                if (cmp.compare(records.get(j), pivot) < 0) {
                    Collections.swap(records, store, j);
                    store++;
                }
            }
            Collections.swap(records, store, hi);
            quick(records, lo, store - 1, cmp);
            lo = store + 1;
        }
    }

    static String tags(List<Record> records) {
        StringBuilder sb = new StringBuilder();
        for (Record record : records) {
            if (sb.length() > 0) sb.append(',');
            sb.append(record.tag);
        }
        return sb.toString();
    }

    static String groupTags(List<Record> records, int key) {
        StringBuilder sb = new StringBuilder();
        for (Record record : records) {
            if (record.key == key) {
                if (sb.length() > 0) sb.append(',');
                sb.append(record.tag);
            }
        }
        return sb.toString();
    }

    static boolean keysSorted(List<Record> records) {
        for (int i = 1; i < records.size(); i++) {
            if (records.get(i - 1).key > records.get(i).key) return false;
        }
        return true;
    }

    static boolean isPermutation(List<Record> left, List<Record> right) {
        List<String> a = new ArrayList<>();
        List<String> b = new ArrayList<>();
        for (Record record : left) a.add(record.key + ":" + record.tag);
        for (Record record : right) b.add(record.key + ":" + record.tag);
        Collections.sort(a);
        Collections.sort(b);
        return a.equals(b);
    }

    public static void main(String[] args) {
        List<Record> records = Arrays.asList(
                new Record(3, "a"), new Record(1, "x"), new Record(3, "b"), new Record(2, "y"),
                new Record(3, "c"), new Record(1, "w"), new Record(2, "z"));

        check(tags(stableSort(records)).equals("x,w,y,z,a,b,c"), "ties keep original order");
        check(tags(records).equals("a,x,b,y,c,w,z"), "input untouched");

        // the black box itself is only guaranteed to order by key, not to keep ties in order
        List<Record> raw = unstableSort(records);
        check(keysSorted(raw), "raw sort ordered by key");
        check(isPermutation(raw, records), "raw sort is a permutation");

        check(tags(stableSort(Arrays.asList(new Record(1, "p"), new Record(1, "q"),
                new Record(1, "r"), new Record(1, "s")))).equals("p,q,r,s"), "all keys equal");
        check(stableSort(new ArrayList<Record>()).isEmpty(), "empty input");
        check(tags(stableSort(Arrays.asList(new Record(9, "only")))).equals("only"), "single record");
        check(tags(stableSort(Arrays.asList(new Record(0, "p"), new Record(-1, "q"),
                new Record(0, "r"), new Record(-1, "s")))).equals("q,s,p,r"), "negative keys");

        // larger batch: equal keys must stay in their original order
        Random rng = new Random(41);
        List<Record> batch = new ArrayList<>();
        for (int i = 0; i < 120; i++) batch.add(new Record(rng.nextInt(5), String.format("tag-%03d", i)));
        List<Record> ordered = stableSort(batch);
        check(keysSorted(ordered), "batch keys sorted");
        check(isPermutation(ordered, batch), "batch is a permutation");
        for (int key = 0; key < 5; key++) {
            check(groupTags(ordered, key).equals(groupTags(batch, key)), "batch group " + key);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
