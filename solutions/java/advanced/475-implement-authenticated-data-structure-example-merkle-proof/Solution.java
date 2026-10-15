/** Task 475 - Implement authenticated data structure example (Merkle+proof). (Tier: Advanced, CTCI Beyond — Security & Privacy) */
import java.util.*;

class AuthenticatedLog {
    static final long MASK = 0xFFFFFFFFL;

    final List<String> records;
    final List<long[]> levels;

    AuthenticatedLog(List<String> values) {
        records = new ArrayList<>(values);
        levels = new ArrayList<>();
        List<Long> current = new ArrayList<>();
        for (String v : records) current.add(hashLeaf(v));
        List<List<Long>> all = new ArrayList<>();
        all.add(current);
        while (current.size() > 1) {
            List<Long> next = new ArrayList<>();
            for (int i = 0; i + 1 < current.size(); i += 2) next.add(hashNode(current.get(i), current.get(i + 1)));
            if (current.size() % 2 == 1) next.add(current.get(current.size() - 1)); // promote
            all.add(next);
            current = next;
        }
        for (List<Long> lvl : all) {
            long[] arr = new long[lvl.size()];
            for (int i = 0; i < lvl.size(); i++) arr[i] = lvl.get(i);
            levels.add(arr);
        }
    }

    int size() {
        return records.size();
    }

    long root() {
        return levels.get(levels.size() - 1)[0];
    }

    List<long[]> proof(int index) {
        List<long[]> path = new ArrayList<>();
        int idx = index;
        for (int li = 0; li < levels.size() - 1; li++) {
            long[] level = levels.get(li);
            if (idx % 2 == 1) path.add(new long[]{level[idx - 1], 0});
            else if (idx + 1 < level.length) path.add(new long[]{level[idx + 1], 1});
            idx /= 2;
        }
        return path;
    }

    List<List<long[]>> multiProof(int[] targets) {
        long[] level = levels.get(0);
        Set<Integer> needed = new TreeSet<>();
        for (int t : targets) needed.add(t);
        List<List<long[]>> out = new ArrayList<>();
        while (level.length > 1) {
            List<long[]> siblings = new ArrayList<>();
            for (int i : needed) {
                if (i % 2 == 1 && !needed.contains(i - 1)) {
                    siblings.add(new long[]{i - 1, level[i - 1]});
                } else if (i % 2 == 0 && i + 1 < level.length && !needed.contains(i + 1)) {
                    siblings.add(new long[]{i + 1, level[i + 1]});
                }
            }
            out.add(siblings);
            int nxtLen = (level.length + 1) / 2;
            long[] nxt = new long[nxtLen];
            for (int i = 0; i + 1 < level.length; i += 2) nxt[i / 2] = hashNode(level[i], level[i + 1]);
            if (level.length % 2 == 1) nxt[nxtLen - 1] = level[level.length - 1];
            level = nxt;
            Set<Integer> nextNeeded = new TreeSet<>();
            for (int t : needed) nextNeeded.add(t / 2);
            needed = nextNeeded;
        }
        return out;
    }

    static long fnv1a(byte[] data) {
        long h = 2166136261L;
        for (byte b : data) {
            h = ((h ^ (b & 0xFF)) * 16777619L) & MASK;
        }
        return h;
    }

    static long hashLeaf(String value) {
        byte[] data = new byte[value.length() + 1];
        for (int i = 0; i < value.length(); i++) data[i + 1] = (byte) value.charAt(i);
        return fnv1a(data);
    }

    static long hashNode(long left, long right) {
        byte[] data = new byte[8];
        for (int i = 0; i < 4; i++) {
            data[i] = (byte) ((left >> (8 * (3 - i))) & 0xFF);
            data[4 + i] = (byte) ((right >> (8 * (3 - i))) & 0xFF);
        }
        return fnv1a(data);
    }

    static boolean verify(long root, String value, List<long[]> path) {
        long h = hashLeaf(value);
        for (long[] step : path) {
            h = step[1] == 0 ? hashNode(step[0], h) : hashNode(h, step[0]);
        }
        return h == root;
    }

    static boolean verifyMulti(long root, int n, Map<Integer, String> targets, List<List<long[]>> proof) {
        Map<Integer, Long> known = new HashMap<>();
        for (Map.Entry<Integer, String> e : targets.entrySet()) known.put(e.getKey(), hashLeaf(e.getValue()));
        int levelLen = n;
        for (List<long[]> siblings : proof) {
            Map<Integer, Long> sibMap = new HashMap<>();
            for (long[] s : siblings) sibMap.put((int) s[0], s[1]);
            Map<Integer, Long> parents = new HashMap<>();
            for (int i = 0; i + 1 < levelLen; i += 2) {
                boolean haveLeft = known.containsKey(i);
                boolean haveRight = known.containsKey(i + 1);
                if (!haveLeft && !haveRight) continue;
                if (haveLeft && haveRight) {
                    parents.put(i / 2, hashNode(known.get(i), known.get(i + 1)));
                } else {
                    int pos = haveLeft ? i + 1 : i;
                    if (!sibMap.containsKey(pos)) return false;
                    long left = haveLeft ? known.get(i) : sibMap.get(i);
                    long right = haveRight ? known.get(i + 1) : sibMap.get(i + 1);
                    parents.put(i / 2, hashNode(left, right));
                }
            }
            if (levelLen % 2 == 1 && known.containsKey(levelLen - 1)) {
                parents.put((levelLen - 1) / 2, known.get(levelLen - 1));
            }
            known = parents;
            levelLen = (levelLen + 1) / 2;
        }
        Long got = known.get(0);
        return got != null && got == root;
    }

    static long updatedRoot(String newValue, List<long[]> path) {
        long h = hashLeaf(newValue);
        for (long[] step : path) {
            h = step[1] == 0 ? hashNode(step[0], h) : hashNode(h, step[0]);
        }
        return h;
    }
}

public class Solution {
    public static void main(String[] args) {
        List<String> leaves8 = new ArrayList<>();
        for (int i = 0; i < 8; i++) leaves8.add("tx" + i);
        AuthenticatedLog log = new AuthenticatedLog(leaves8);
        check(log.root() == 955777637L, "n=8 root");

        List<long[]> path2 = log.proof(2);
        check(path2.size() == 3 && path2.get(0)[0] == 1947293384L && path2.get(1)[0] == 240787827L
                && path2.get(2)[0] == 3271786783L, "record 2 audit path hashes");
        check(path2.get(0)[1] == 1 && path2.get(1)[1] == 0 && path2.get(2)[1] == 1, "record 2 path sides");

        List<List<long[]>> mp01 = log.multiProof(new int[]{0, 1});
        check(mp01.size() == 3 && mp01.get(0).isEmpty() && mp01.get(1).size() == 1 && mp01.get(2).size() == 1, "mp01 shape");
        check(mp01.get(1).get(0)[1] == 1737906683L && mp01.get(2).get(0)[1] == 3271786783L, "mp01 sibling hashes");
        Map<Integer, String> t01 = new HashMap<>();
        t01.put(0, "tx0");
        t01.put(1, "tx1");
        check(AuthenticatedLog.verifyMulti(log.root(), 8, t01, mp01), "mp01 verifies");
        check(2 < log.proof(0).size() + log.proof(1).size(), "batching saves hashes");

        List<List<long[]>> mp03 = log.multiProof(new int[]{0, 3});
        check(mp03.get(0).size() == 2 && mp03.get(1).isEmpty() && mp03.get(2).size() == 1, "mp03 shape");
        Map<Integer, String> t03 = new HashMap<>();
        t03.put(0, "tx0");
        t03.put(3, "tx3");
        check(AuthenticatedLog.verifyMulti(log.root(), 8, t03, mp03), "mp03 verifies");

        List<List<long[]>> mpAll = log.multiProof(new int[]{0, 1, 2, 3, 4, 5, 6, 7});
        check(mpAll.get(0).isEmpty() && mpAll.get(1).isEmpty() && mpAll.get(2).isEmpty(), "full batch needs no siblings");
        Map<Integer, String> allTargets = new HashMap<>();
        for (int i = 0; i < 8; i++) allTargets.put(i, "tx" + i);
        check(AuthenticatedLog.verifyMulti(log.root(), 8, allTargets, mpAll), "full batch verifies");

        List<List<long[]>> mp246 = log.multiProof(new int[]{2, 4, 6});
        check(mp246.get(0).size() == 3 && mp246.get(1).size() == 1 && mp246.get(2).isEmpty(), "mp246 shape");
        Map<Integer, String> t246 = new HashMap<>();
        t246.put(2, "tx2");
        t246.put(4, "tx4");
        t246.put(6, "tx6");
        check(AuthenticatedLog.verifyMulti(log.root(), 8, t246, mp246), "mp246 verifies");

        List<List<long[]>> mp5 = log.multiProof(new int[]{5});
        List<Long> mp5Hashes = new ArrayList<>();
        for (List<long[]> lvl : mp5) for (long[] s : lvl) mp5Hashes.add(s[1]);
        List<Long> path5Hashes = new ArrayList<>();
        for (long[] step : log.proof(5)) path5Hashes.add(step[0]);
        Collections.sort(mp5Hashes);
        Collections.sort(path5Hashes);
        check(mp5Hashes.equals(path5Hashes), "single-target batch equals audit path");
        Map<Integer, String> t5 = new HashMap<>();
        t5.put(5, "tx5");
        check(AuthenticatedLog.verifyMulti(log.root(), 8, t5, mp5), "mp5 verifies");

        List<List<long[]>> badSib = new ArrayList<>();
        for (List<long[]> lvl : mp01) {
            List<long[]> copy = new ArrayList<>();
            for (long[] s : lvl) copy.add(new long[]{s[0], s[1]});
            badSib.add(copy);
        }
        long[] tampered = badSib.get(1).get(0);
        badSib.get(1).set(0, new long[]{tampered[0], tampered[1] ^ 1L});
        check(!AuthenticatedLog.verifyMulti(log.root(), 8, t01, badSib), "tampered sibling fails");
        Map<Integer, String> t01bad = new HashMap<>();
        t01bad.put(0, "tx0x");
        t01bad.put(1, "tx1");
        check(!AuthenticatedLog.verifyMulti(log.root(), 8, t01bad, mp01), "tampered target fails");

        List<String> leaves7 = new ArrayList<>();
        for (int i = 0; i < 7; i++) leaves7.add("tx" + i);
        AuthenticatedLog log7 = new AuthenticatedLog(leaves7);
        List<List<long[]>> mp56 = log7.multiProof(new int[]{5, 6});
        check(mp56.get(0).size() == 1 && mp56.get(1).isEmpty() && mp56.get(2).size() == 1, "mp56 shape");
        Map<Integer, String> t56 = new HashMap<>();
        t56.put(5, "tx5");
        t56.put(6, "tx6");
        check(AuthenticatedLog.verifyMulti(log7.root(), 7, t56, mp56), "mp56 verifies");
        List<List<long[]>> mp5b = log7.multiProof(new int[]{5});
        check(mp5b.get(0).size() == 1 && mp5b.get(1).size() == 1 && mp5b.get(2).size() == 1, "mp5b shape");
        Map<Integer, String> t5b = new HashMap<>();
        t5b.put(5, "tx5");
        check(AuthenticatedLog.verifyMulti(log7.root(), 7, t5b, mp5b), "mp5b verifies");

        List<String> updated = new ArrayList<>(leaves8);
        updated.set(2, "tx2*");
        AuthenticatedLog logNew = new AuthenticatedLog(updated);
        check(logNew.root() == 3100337682L, "updated root");
        check(AuthenticatedLog.updatedRoot("tx2*", path2) == 3100337682L, "update path yields new root");
        check(!AuthenticatedLog.verify(logNew.root(), "tx2", path2), "stale proof fails after update");
        check(AuthenticatedLog.verify(logNew.root(), "tx2*", logNew.proof(2)), "new proof verifies");
        check(!AuthenticatedLog.verifyMulti(logNew.root(), 8, t01, mp01), "stale batch fails");
        List<List<long[]>> mp25New = logNew.multiProof(new int[]{2, 5});
        Map<Integer, String> t25New = new HashMap<>();
        t25New.put(2, "tx2*");
        t25New.put(5, "tx5");
        check(AuthenticatedLog.verifyMulti(logNew.root(), 8, t25New, mp25New), "updated batch verifies");
        check(!AuthenticatedLog.verify(logNew.root(), "tx5", log.proof(5)), "old proof of unaffected record fails");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
