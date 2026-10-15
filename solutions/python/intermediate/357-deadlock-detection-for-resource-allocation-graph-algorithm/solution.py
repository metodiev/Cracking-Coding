"""Task 357 - Deadlock detection for resource allocation graph (algorithm). (Tier: Intermediate, CTCI Ch 15)

Deadlock detection by cycle finding: the resource allocation graph (processes hold and
request resources, a resource has at most one holder) collapses into the wait-for graph -
each request (p, r) with r held by q becomes the edge p -> q; requests for free resources
contribute no edge. A deadlock exists exactly when the wait-for graph has a cycle, so
find_cycle() colour-DFSes (white/gray/black) and returns the first cycle it closes,
rotated so its smallest vertex comes first (a process requesting a resource it already
holds yields the self-loop [p]). The self-test checks hand-built deadlocked, acyclic and
self-loop graphs, then cross-checks cycle existence against an independent Kosaraju SCC
check on 300 LCG-generated resource allocation graphs.
"""
SCRIPT_TRIALS = 300


def canonical_cycle(cycle):
    pivot = cycle.index(min(cycle))
    return cycle[pivot:] + cycle[:pivot]


def build_wait_for(processes, resources, holds, requests):
    """Collapse the RAG: process adjacency lists of the processes each one waits for."""
    holder = [-1] * resources
    for process, resource in holds:
        if holder[resource] != -1 and holder[resource] != process:
            raise ValueError("resource {} held by more than one process".format(resource))
        holder[resource] = process
    adjacency = [[] for _ in range(processes)]
    for process, resource in requests:
        current_holder = holder[resource]
        if current_holder != -1:                     # waiting for a free resource is not a wait
            adjacency[process].append(current_holder)
    return adjacency


def find_cycle(processes, adjacency):
    """Returns a canonical cycle or None. 0 = white, 1 = gray (on the DFS path), 2 = black."""
    color = [0] * processes
    path = []

    def dfs(node):
        color[node] = 1
        path.append(node)
        for next_node in adjacency[node]:
            if color[next_node] == 1:                # back edge: the path slice is a cycle
                return path[path.index(next_node):]
            if color[next_node] == 0:
                found = dfs(next_node)
                if found is not None:
                    return found
        path.pop()
        color[node] = 2
        return None

    for start in range(processes):
        if color[start] == 0:
            found = dfs(start)
            if found is not None:
                return canonical_cycle(found)
    return None


def has_cycle_via_scc(processes, adjacency):
    """Independent check: a cycle exists iff some SCC has size > 1 or a self-loop exists."""
    reverse = [[] for _ in range(processes)]
    for u in range(processes):
        for v in adjacency[u]:
            reverse[v].append(u)

    visited = [False] * processes
    order = []

    def dfs_finish(u):
        visited[u] = True
        for v in adjacency[u]:
            if not visited[v]:
                dfs_finish(v)
        order.append(u)

    for u in range(processes):
        if not visited[u]:
            dfs_finish(u)

    component = [-1] * processes
    sizes = []
    for u in reversed(order):
        if component[u] != -1:
            continue
        component[u] = len(sizes)
        size = 0
        stack = [u]
        while stack:
            x = stack.pop()
            size += 1
            for y in reverse[x]:
                if component[y] == -1:
                    component[y] = component[u]
                    stack.append(y)
        sizes.append(size)

    if any(size > 1 for size in sizes):
        return True
    return any(u in adjacency[u] for u in range(processes))


def assert_valid_cycle(adjacency, cycle):
    assert cycle == canonical_cycle(cycle), "cycle returned canonicalised"
    assert len(set(cycle)) == len(cycle), "cycle vertices are distinct"
    for index in range(len(cycle)):
        assert cycle[(index + 1) % len(cycle)] in adjacency[cycle[index]], "cycle edges exist"


def random_rag(processes, resources, seed):
    """LCG RAG: every resource gets one holder or stays free; processes issue 0..2 requests."""
    holder = [-1] * resources
    holds = []
    for resource in range(resources):
        seed = (seed * 75 + 74) % 65537
        if seed % 3 != 0:
            process = seed % processes
            holder[resource] = process
            holds.append((process, resource))
    requests = []
    for process in range(processes):
        seed = (seed * 75 + 74) % 65537
        for _ in range(seed % 3):
            seed = (seed * 75 + 74) % 65537
            requests.append((process, seed % resources))
    return holds, requests, seed


def main():
    # Hand-built deadlock: two processes wait for each other; canonical cycle [0, 1].
    adjacency = build_wait_for(2, 2, [(0, 0), (1, 1)], [(0, 1), (1, 0)])
    assert adjacency == [[1], [0]], "wait-for edges"
    assert find_cycle(2, adjacency) == [0, 1], "classic two-process deadlock"

    # Acyclic: process 0 waits for process 1, which waits for nothing.
    adjacency = build_wait_for(2, 2, [(0, 0), (1, 1)], [(0, 1)])
    assert adjacency == [[1], []]
    assert find_cycle(2, adjacency) is None, "no cycle without a wait cycle"

    # Self-loop: a process requests a resource it already holds.
    adjacency = build_wait_for(1, 1, [(0, 0)], [(0, 0)])
    assert adjacency == [[0]]
    assert find_cycle(1, adjacency) == [0], "self-loop is detected"

    # Three-process cycle; the returned cycle is the hand-checked canonical rotation.
    adjacency = build_wait_for(3, 3, [(1, 0), (2, 1), (0, 2)], [(0, 0), (1, 1), (2, 2)])
    assert adjacency == [[1], [2], [0]]
    assert find_cycle(3, adjacency) == [0, 1, 2], "three-process deadlock"

    # Requests for free resources create no edges.
    adjacency = build_wait_for(2, 2, [(0, 0)], [(1, 1)])
    assert adjacency == [[], []]
    assert find_cycle(2, adjacency) is None, "waiting for a free resource is not a deadlock"

    # All processes idle, and an extra process outside the cycle.
    adjacency = build_wait_for(3, 3, [(0, 0), (1, 1), (2, 2)], [])
    assert find_cycle(3, adjacency) is None, "idle processes never deadlock"
    adjacency = build_wait_for(3, 3, [(0, 0), (1, 1), (2, 2)], [(0, 1), (1, 0)])
    assert find_cycle(3, adjacency) == [0, 1], "cycle independent of an idle process"

    # A resource may have only one holder.
    try:
        build_wait_for(2, 1, [(0, 0), (1, 0)], [])
        raise AssertionError("a double-held resource must be rejected")
    except ValueError:
        pass

    # Cross-check cycle existence against the independent SCC check on LCG RAGs.
    seed = 13579
    cycles_seen = 0
    for trial in range(SCRIPT_TRIALS):
        seed = (seed * 75 + 74) % 65537
        processes = 2 + seed % 6
        seed = (seed * 75 + 74) % 65537
        resources = 1 + seed % 6
        holds, requests, seed = random_rag(processes, resources, seed)
        adjacency = build_wait_for(processes, resources, holds, requests)
        cycle = find_cycle(processes, adjacency)
        scc_says_cycle = has_cycle_via_scc(processes, adjacency)
        assert (cycle is not None) == scc_says_cycle, "SCC cross-check at trial {}".format(trial)
        if cycle is not None:
            cycles_seen += 1
            assert_valid_cycle(adjacency, cycle)
    assert cycles_seen >= 20, "LCG graphs should contain deadlocks (saw {})".format(cycles_seen)


if __name__ == "__main__":
    main()
    print("PASS")
