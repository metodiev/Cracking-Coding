/** Task 471 - Implement vector clock for distributed event ordering. (Tier: Advanced, CTCI Ch 9 — System Design and Scalability) */
import java.util.*;

class Event {
    final int node;
    final char kind; // 'L' local, 'S' send, 'R' receive
    final int msg;
    final int[] clock;

    Event(int node, char kind, int msg, int[] clock) {
        this.node = node;
        this.kind = kind;
        this.msg = msg;
        this.clock = clock;
    }
}

class SentMsg {
    final int id;
    final int[] clock;

    SentMsg(int id, int[] clock) {
        this.id = id;
        this.clock = clock;
    }
}

public class Solution {
    private static final int NODES = 3;

    static int lcg(int seed) {
        return (seed * 75 + 74) % 65537;
    }

    static String classify(int[] a, int[] b) {
        boolean le = true, ge = true;
        for (int i = 0; i < a.length; i++) {
            if (a[i] > b[i]) le = false;
            if (a[i] < b[i]) ge = false;
        }
        if (le && ge) return "equal";
        if (le) return "before";
        if (ge) return "after";
        return "concurrent";
    }

    private static void emit(List<Event> out, int[][] c, int node, char kind, int msg) {
        out.add(new Event(node, kind, msg, c[node].clone()));
    }

    static List<Event> featuredEvents() {
        int[][] c = new int[NODES][NODES];
        List<Event> events = new ArrayList<>();
        c[0][0]++;
        emit(events, c, 0, 'L', -1);          // 0: A has 1 local event
        c[0][0]++;
        emit(events, c, 0, 'L', -1);          // 1: A has 2
        c[0][0]++;
        emit(events, c, 0, 'S', 0);           // 2: A sends {A:3} to B
        c[1][1]++;
        emit(events, c, 1, 'L', -1);          // 3: B local tick, unaware of A
        c[1][0] = Math.max(c[1][0], 3);
        c[1][1]++;
        emit(events, c, 1, 'R', 0);           // 4: {A:3, B:2}
        c[1][1]++;
        emit(events, c, 1, 'S', 1);           // 5: B sends {A:3, B:3} to C
        c[2][0] = Math.max(c[2][0], 3);
        c[2][1] = Math.max(c[2][1], 3);
        c[2][2]++;
        emit(events, c, 2, 'R', 1);           // 6: C merges -> {3,3,1}
        return events;
    }

    static List<Event> genScript(int count, int seed) {
        List<Event> events = new ArrayList<>();
        List<List<SentMsg>> pending = new ArrayList<>();
        for (int i = 0; i < NODES; i++) pending.add(new ArrayList<>());
        int[][] clocks = new int[NODES][NODES];
        int nextMsg = 0;
        for (int k = 0; k < count; k++) {
            seed = lcg(seed);
            int node = seed % NODES;
            seed = lcg(seed);
            int kind = seed % 3;
            if (kind == 2 && pending.get(node).isEmpty()) kind = 0;
            if (kind == 1) {
                seed = lcg(seed);
                int target = (node + 1 + seed % (NODES - 1)) % NODES;
                clocks[node][node]++;
                pending.get(target).add(new SentMsg(nextMsg, clocks[node].clone()));
                events.add(new Event(node, 'S', nextMsg, clocks[node].clone()));
                nextMsg++;
            } else if (kind == 2) {
                SentMsg m = pending.get(node).remove(0);
                for (int i = 0; i < NODES; i++) clocks[node][i] = Math.max(clocks[node][i], m.clock[i]);
                clocks[node][node]++;
                events.add(new Event(node, 'R', m.id, clocks[node].clone()));
            } else {
                clocks[node][node]++;
                events.add(new Event(node, 'L', -1, clocks[node].clone()));
            }
        }
        return events;
    }

    static boolean[][] causalReachability(List<Event> events) {
        int n = events.size();
        boolean[][] reach = new boolean[n][n];
        int[] last = new int[NODES];
        Arrays.fill(last, -1);
        int[] sentAt = new int[n];
        Arrays.fill(sentAt, -1);
        for (int i = 0; i < n; i++) {
            Event e = events.get(i);
            if (last[e.node] != -1) reach[last[e.node]][i] = true;
            last[e.node] = i;
            if (e.kind == 'S') sentAt[e.msg] = i;
            else if (e.kind == 'R') reach[sentAt[e.msg]][i] = true;
        }
        for (int k = 0; k < n; k++)
            for (int i = 0; i < n; i++)
                if (reach[i][k])
                    for (int j = 0; j < n; j++)
                        if (reach[k][j]) reach[i][j] = true;
        return reach;
    }

    public static void main(String[] args) {
        List<Event> fe = featuredEvents();
        int[][] cl = new int[fe.size()][];
        for (int i = 0; i < fe.size(); i++) cl[i] = fe.get(i).clock;
        check(Arrays.equals(cl[0], new int[]{1, 0, 0}), "clock 0");
        check(Arrays.equals(cl[1], new int[]{2, 0, 0}), "clock 1");
        check(Arrays.equals(cl[2], new int[]{3, 0, 0}), "clock 2");
        check(Arrays.equals(cl[3], new int[]{0, 1, 0}), "clock 3");
        check(Arrays.equals(cl[4], new int[]{3, 2, 0}), "clock 4");
        check(Arrays.equals(cl[5], new int[]{3, 3, 0}), "clock 5");
        check(Arrays.equals(cl[6], new int[]{3, 3, 1}), "clock 6");
        check(classify(cl[0], cl[1]).equals("before"), "0->1");
        check(classify(cl[1], cl[2]).equals("before"), "1->2");
        check(classify(cl[0], cl[2]).equals("before"), "0->2");
        check(classify(cl[3], cl[0]).equals("concurrent"), "3 vs 0");
        check(classify(cl[3], cl[1]).equals("concurrent"), "3 vs 1");
        check(classify(cl[3], cl[2]).equals("concurrent"), "3 vs 2");
        check(classify(cl[3], cl[4]).equals("before"), "3->4");
        check(classify(cl[2], cl[4]).equals("before"), "2->4");
        check(classify(cl[4], cl[5]).equals("before"), "4->5");
        check(classify(cl[5], cl[6]).equals("before"), "5->6");
        check(classify(cl[1], cl[6]).equals("before"), "1->6");
        check(classify(cl[6], cl[4]).equals("after"), "6 after 4");
        check(classify(cl[6], cl[0]).equals("after"), "6 after 0");
        for (int i = 0; i < fe.size(); i++)
            for (int j = i + 1; j < fe.size(); j++)
                check(!classify(cl[i], cl[j]).equals("equal"), "no equal clocks");

        List<Event> events = genScript(26, 1234);
        int l = 0, s = 0, r = 0;
        for (Event e : events) {
            if (e.kind == 'L') l++;
            else if (e.kind == 'S') s++;
            else r++;
        }
        check(l == 20 && s == 4 && r == 2, "script shape");
        boolean[][] reach = causalReachability(events);
        int order = 0, concurrent = 0;
        for (int i = 0; i < events.size(); i++) {
            for (int j = i + 1; j < events.size(); j++) {
                String got = classify(events.get(i).clock, events.get(j).clock);
                String expected = reach[i][j] ? "before" : reach[j][i] ? "after" : "concurrent";
                check(got.equals(expected), "pair " + i + "," + j + ": " + got + " vs " + expected);
                if (expected.equals("concurrent")) concurrent++;
                else order++;
            }
        }
        check(order == 140 && concurrent == 185, "pair counts");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
