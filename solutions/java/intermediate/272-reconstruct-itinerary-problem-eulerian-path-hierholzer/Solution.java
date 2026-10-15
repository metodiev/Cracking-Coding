/** Task 272 - Reconstruct itinerary (Eulerian path, Hierholzer). (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

public class Solution {
    /** Returns the airport sequence, or null when the tickets cannot all be used. */
    static List<String> findItinerary(String[][] tickets, String start) {
        Map<String, PriorityQueue<String>> adjacency = new TreeMap<>();
        for (String[] ticket : tickets) {
            adjacency.computeIfAbsent(ticket[0], k -> new PriorityQueue<>()).add(ticket[1]);
        }
        List<String> route = new ArrayList<>();
        Deque<String> stack = new ArrayDeque<>();
        stack.push(start);
        while (!stack.isEmpty()) {
            String airport = stack.peek();
            PriorityQueue<String> outgoing = adjacency.get(airport);
            if (outgoing != null && !outgoing.isEmpty()) {
                stack.push(outgoing.poll());
            } else {
                route.add(stack.pop());
            }
        }
        Collections.reverse(route);
        return route.size() == tickets.length + 1 ? route : null;
    }

    /** Independent check: consecutive pairs of `route` must consume the ticket multiset. */
    static boolean usesAllTickets(List<String> route, String[][] tickets) {
        if (route == null) return false;
        Map<String, Integer> used = new TreeMap<>();
        for (int i = 0; i + 1 < route.size(); i++) {
            String pair = route.get(i) + "->" + route.get(i + 1);
            used.merge(pair, 1, Integer::sum);
        }
        Map<String, Integer> wanted = new TreeMap<>();
        for (String[] ticket : tickets) {
            wanted.merge(ticket[0] + "->" + ticket[1], 1, Integer::sum);
        }
        return used.equals(wanted);
    }

    static String join(List<String> values) {
        return values == null ? "null" : String.join(",", values);
    }

    public static void main(String[] args) {
        String[][] first = {{"MUC","LHR"},{"JFK","MUC"},{"SFO","SJC"},{"LHR","SFO"}};
        List<String> route = findItinerary(first, "JFK");
        check(join(route).equals("JFK,MUC,LHR,SFO,SJC"), "classic itinerary");
        check(usesAllTickets(route, first), "all tickets used");

        String[][] second = {{"JFK","SFO"},{"JFK","ATL"},{"SFO","ATL"},{"ATL","JFK"},{"ATL","SFO"}};
        route = findItinerary(second, "JFK");
        check(join(route).equals("JFK,ATL,JFK,SFO,ATL,SFO"), "reconstruction itinerary");
        check(usesAllTickets(route, second), "all tickets used (2)");

        check(join(findItinerary(new String[][]{{"A","B"}}, "A")).equals("A,B"), "single ticket");
        check(join(findItinerary(new String[][]{}, "JFK")).equals("JFK"), "no tickets");
        check(findItinerary(new String[][]{{"A","B"},{"C","D"}}, "A") == null, "disconnected tickets");
        check(findItinerary(new String[][]{{"A","B"}}, "C") == null, "nothing leaves the start");

        String[][] dupes = {{"JFK","B"},{"JFK","A"},{"A","JFK"}};
        check(join(findItinerary(dupes, "JFK")).equals("JFK,A,JFK,B"), "smallest destination first");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
