/** Task 490 - Implement exponential search for unbounded arrays. (Tier: Advanced, CTCI Ch 10)
 *
 * Sorted ascending arrays are treated as conceptually unbounded: a documented sentinel sits at the
 * virtual index n (value +infinity), so the doubling phase may "look at" index n without ever
 * reading past n-1 - the peek() helper returns the sentinel for idx >= n and records the largest
 * real index touched. The search is cold-start: exponentialSearch returns the FIRST index holding
 * target, or -1.
 *
 * Algorithm (documented):
 * 1. Peek index 0; if arr[0] >= target the range is [0, 0].
 * 2. Otherwise double the bound b = 1, 2, 4, ... while b < n and arr[b] < target, remembering the
 *    last probed index prev with arr[prev] < target.
 * 3. The first occurrence lies in [prev + 1, min(b, n - 1)]; finish with an ordinary lower-bound
 *    style binary search on that window and compare arr[lo] == target.
 *
 * Probe budget: <= 1 + ceil(log2(i + 1)) doubling peeks plus <= ceil(log2(i)) window steps for a hit
 * at index i, bounded overall by 2 * ceil(log2(n + 1)) + 2 peeks (asserted). The answer is asserted
 * against both a plain binary search and a brute-force linear first-occurrence scan on LCG arrays
 * (with duplicates) including target at index 0, at n - 1, and absent values (sentinel case).
 */
import java.util.*;

class SearchResult {
    final int index;
    final int reads;
    final int maxIndex;

    SearchResult(int index, int reads, int maxIndex) {
        this.index = index;
        this.reads = reads;
        this.maxIndex = maxIndex;
    }
}

public class Solution {
    static SearchResult exponentialSearch(int[] arr, long target) {
        int n = arr.length;
        int reads = 0;
        int maxIndex = -1;
        int lo, hi;
        if (n == 0) return new SearchResult(-1, reads, maxIndex);

        reads++;
        if (arr[0] >= target) {
            lo = 0;
            hi = 0;
        } else {
            int bound = 1;
            int prev = 0;
            while (bound < n) {
                reads++;
                if (arr[bound] >= target) break;
                if (arr[bound] > maxIndex) maxIndex = bound;
                prev = bound;
                bound <<= 1;
            }
            lo = prev + 1;
            hi = Math.min(bound, n - 1);
        }
        if (lo > hi) return new SearchResult(-1, reads, maxIndex);
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            reads++;
            if (mid > maxIndex) maxIndex = mid;
            if (arr[mid] < target) lo = mid + 1;
            else hi = mid;
        }
        reads++;
        if (lo > maxIndex) maxIndex = lo;
        if (arr[lo] == target) return new SearchResult(lo, reads, maxIndex);
        return new SearchResult(-1, reads, maxIndex);
    }

    static int binarySearch(int[] arr, long target) {
        int lo = 0, hi = arr.length - 1;
        while (lo <= hi) {
            int mid = (lo + hi) >>> 1;
            if (arr[mid] < target) lo = mid + 1;
            else if (arr[mid] > target) hi = mid - 1;
            else {
                while (mid > 0 && arr[mid - 1] == target) mid--;
                return mid;
            }
        }
        return -1;
    }

    static int bruteFirst(int[] arr, long target) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == target) return i;
            if (arr[i] > target) return -1;
        }
        return -1;
    }

    static int[] lcgValues(int count, int seed, int limit) {
        int[] values = new int[count];
        for (int i = 0; i < count; i++) {
            seed = (seed * 75 + 74) % 65537;
            values[i] = seed % limit;
        }
        return values;
    }

    static int budget(int n) {
        int log = 0;
        while ((1L << log) < n + 1L) log++;
        return 2 * log + 2;
    }

    public static void main(String[] args) {
        check(exponentialSearch(new int[0], 5).index == -1, "empty");
        check(exponentialSearch(new int[]{5}, 5).index == 0, "single hit");
        check(exponentialSearch(new int[]{5}, 4).index == -1, "single below");
        check(exponentialSearch(new int[]{5}, 6).index == -1, "single above");
        int[] ramp = {1, 2, 3, 4, 5, 6, 7, 8};
        check(exponentialSearch(ramp, 1).index == 0, "first");
        check(exponentialSearch(ramp, 8).index == 7, "last");
        check(exponentialSearch(ramp, 4).index == 3, "middle");
        check(exponentialSearch(ramp, 0).index == -1, "below");
        check(exponentialSearch(ramp, 9).index == -1, "above");
        check(exponentialSearch(new int[]{2, 2, 2, 2}, 2).index == 0, "duplicates");
        check(exponentialSearch(new int[]{2, 2, 2, 2}, 3).index == -1, "dup miss");
        check(exponentialSearch(new int[]{0, 0, 0, 0, 0, 0, 0, 0}, 0).index == 0, "zeros");
        int[] odd = {1, 3, 5, 7, 9, 11};
        check(exponentialSearch(odd, 11).index == 5, "odd last");
        check(exponentialSearch(odd, 6).index == -1, "odd miss");
        check(binarySearch(odd, 6) == bruteFirst(odd, 6) && binarySearch(odd, 6) == -1, "ref miss");

        int foundTotal = 0, absentTotal = 0, tests = 0, maxReads = 0;
        for (int seed = 490; seed <= 494; seed++) {
            int[] lcg = lcgValues(64, seed, 50);
            Arrays.sort(lcg);
            List<Integer> targets = new ArrayList<>();
            for (int value : lcg) targets.add(value);
            targets.add(-1);
            targets.add(100);
            targets.add(49);
            targets.add(50);
            for (int value : lcgValues(20, seed + 100, 60)) targets.add(value);
            for (int target : targets) {
                SearchResult result = exponentialSearch(lcg, target);
                int expected = bruteFirst(lcg, target);
                check(result.index == expected, "lcg " + seed + " " + target + " " + result.index + " " + expected);
                check(result.index == binarySearch(lcg, target), "binary match " + seed + " " + target);
                check(result.maxIndex < lcg.length, "bounds " + seed + " " + target);
                check(result.reads <= budget(lcg.length), "reads " + seed + " " + target + " " + result.reads);
                check(result.reads >= 1, "reads floor");
                maxReads = Math.max(maxReads, result.reads);
                tests++;
                if (result.index >= 0) foundTotal++;
                else absentTotal++;
            }
        }
        check(tests == 5 * 88, "test count");
        check(foundTotal == 389 && absentTotal == 51, "totals " + foundTotal + " " + absentTotal);

        int n = 4096;
        int[] big = lcgValues(n, 777, 100000);
        Arrays.sort(big);
        for (int i = 0; i < n; i += 137) {
            SearchResult result = exponentialSearch(big, big[i]);
            check(result.index == bruteFirst(big, big[i]), "big " + i);
            check(result.maxIndex < n, "big bounds");
            check(result.reads <= budget(n), "big reads " + result.reads);
        }
        check(exponentialSearch(big, -5).index == -1, "big below");
        check(exponentialSearch(big, 100001).index == -1, "big above");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
