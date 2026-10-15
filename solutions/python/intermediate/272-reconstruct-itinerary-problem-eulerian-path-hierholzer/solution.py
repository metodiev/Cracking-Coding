"""Task 272 - Reconstruct itinerary (Eulerian path, Hierholzer). (Tier: Intermediate, CTCI Ch 4)

Tickets are edges; the puzzle asks for the Eulerian path that is lexicographically smallest.
Hierholzer with min-heaps per airport: always fly the smallest unused destination first and
prepend the airport once it is exhausted (the post-order walk, then reversed).
"""

import heapq
from collections import Counter


def find_itinerary(tickets, start="JFK"):
    """Returns the airport sequence, or None when the tickets cannot all be used."""
    adjacency = {}
    for origin, destination in tickets:
        adjacency.setdefault(origin, []).append(destination)
    for heap in adjacency.values():
        heapq.heapify(heap)

    route = []
    stack = [start]
    while stack:
        airport = stack[-1]
        if adjacency.get(airport):
            stack.append(heapq.heappop(adjacency[airport]))
        else:
            route.append(stack.pop())
    route.reverse()
    return route if len(route) == len(tickets) + 1 else None


def uses_all_tickets(route, tickets):
    """Independent check: consecutive pairs of `route` must consume the ticket multiset."""
    if route is None:
        return False
    return Counter(zip(route, route[1:])) == Counter(tickets)


def main():
    first = [("MUC", "LHR"), ("JFK", "MUC"), ("SFO", "SJC"), ("LHR", "SFO")]
    route = find_itinerary(first)
    assert route == ["JFK", "MUC", "LHR", "SFO", "SJC"], route
    assert uses_all_tickets(route, first)

    second = [("JFK", "SFO"), ("JFK", "ATL"), ("SFO", "ATL"), ("ATL", "JFK"), ("ATL", "SFO")]
    route = find_itinerary(second)
    assert route == ["JFK", "ATL", "JFK", "SFO", "ATL", "SFO"], route
    assert uses_all_tickets(route, second)

    assert find_itinerary([("A", "B")], start="A") == ["A", "B"]
    assert find_itinerary([], start="JFK") == ["JFK"]

    # the graph is disconnected: the walk cannot cover every ticket
    assert find_itinerary([("A", "B"), ("C", "D")], start="A") is None
    # no ticket leaves the start airport
    assert find_itinerary([("A", "B")], start="C") is None

    # duplicates and a tie: always take the smallest destination available
    dupes = [("JFK", "B"), ("JFK", "A"), ("A", "JFK")]
    assert find_itinerary(dupes) == ["JFK", "A", "JFK", "B"]


if __name__ == "__main__":
    main()
    print("PASS")
