"""Task 946 - Implement high-performance external-memory graph traversal with partitioning.  (Tier: Master, CTCI Ch 9 - System Design and Scalability)

A block-level external-memory model.  The graph lives in fixed-size edge blocks sorted by source vertex
(the partitioned layout), preceded by a blocked directory mapping a vertex to its first entry; one LRU
cache covers both arrays and every physical read is charged either as a *random* read (cost 10) or a
*sequential* read (cost 1), which is exactly what the layout buys.  `naive_bfs` pulls each adjacency
with random reads in FIFO order, while `external_bfs` sorts the frontier, processes it in batches
bounded by a memory budget and so touches the directory and the edge file in increasing block order.

Simplified versus a real engine (Ligra/GraphChi): blocks in a list instead of a file, a semi-external
model where the visited set fits in memory, and a two-constant cost model instead of latencies.
Verified: distances match an in-memory BFS oracle; the directory and edge reads of every pass are each
non-decreasing while the baseline's order is not; the modelled cost is far lower; hit/miss and
read-kind accounting is exact; the batch stays inside the memory budget and the entries examined equal
the reached degree sum; larger caches only reduce reads; empty, isolated, unknown-start and
multi-source boundaries; determinism.
"""
import collections

BLOCK, DIR_BLOCK = 8, 16                # edges per block, directory entries per block
RANDOM_COST, SEQUENTIAL_COST = 10, 1    # cost units of a random vs a sequential block read


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def cost_of(disk):
    return RANDOM_COST * disk.random_reads + SEQUENTIAL_COST * disk.sequential_reads


def pseudo(value):
    """Deterministic 64-bit mix: builds a reproducible graph without any RNG object."""
    value = (value * 0x9E3779B97F4A7C15 + 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value ^= value >> 29
    return ((value * 0x94D049BB133111EB) & ((1 << 64) - 1)) >> 11


def build_graph(vertices, degree, shortcut):
    """A path (so the traversal has several levels) plus deterministic long-range shortcuts."""
    edges = {(vertex, vertex + 1) for vertex in range(vertices - 1)}
    for source in range(vertices):
        for step in range(degree):
            edges.add((source, (source + 1 + pseudo(source * 97 + step * 13)) % vertices))
    for index in range(shortcut):
        edges.add((pseudo(1000 + index) % vertices, pseudo(2000 + index) % vertices))
    return sorted((source, target) for source, target in edges if source != target)


class Disk:
    """Block store with an LRU cache and exact read accounting."""

    def __init__(self, blocks, capacity):
        self.blocks, self.capacity = blocks, capacity
        self.cache = collections.OrderedDict()
        self.requests = self.hits = self.misses = 0
        self.random_reads = self.sequential_reads = 0
        self.trace = []

    def read(self, index, sequential=False):
        self.requests += 1
        if index in self.cache:
            self.hits += 1
            self.cache.move_to_end(index)
            return self.blocks[index]
        self.misses += 1
        self.cache[index] = True
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        if sequential:
            self.sequential_reads += 1
        else:
            self.random_reads += 1
        self.trace.append((index, "sequential" if sequential else "random"))
        return self.blocks[index]

class ExternalGraph:
    """Edge blocks sorted by source plus a blocked directory, sharing one cache."""

    def __init__(self, edges, block_size=BLOCK, dir_size=DIR_BLOCK, cache_capacity=4):
        self.edge_blocks = [tuple(edges[i:i + block_size])
                            for i in range(0, len(edges), block_size)]
        first = []
        for block_index, block in enumerate(self.edge_blocks):
            for offset, (source, _) in enumerate(block):
                if not first or first[-1][0] != source:
                    first.append((source, block_index, offset))
        self.dir_blocks = [tuple(first[i:i + dir_size]) for i in range(0, len(first), dir_size)]
        self.location = {entry[0]: (dir_index, position)
                         for dir_index, block in enumerate(self.dir_blocks)
                         for position, entry in enumerate(block)}
        self.vertices, self.edge_offset = [entry[0] for entry in first], len(self.dir_blocks)
        self.disk = Disk(self.dir_blocks + self.edge_blocks, cache_capacity)

    def adjacency(self, vertex, sequential):
        """Read a vertex's adjacency; its edge blocks are visited in increasing order."""
        if vertex not in self.location:
            return ()
        dir_index, position = self.location[vertex]
        entry = self.disk.read(dir_index, sequential)[position]
        block_index, offset, targets = self.edge_offset + entry[1], entry[2], []
        while block_index < self.edge_offset + len(self.edge_blocks):
            block = self.disk.read(block_index, sequential)
            while offset < len(block):
                source, target = block[offset]
                if source > vertex:
                    return tuple(targets)
                if source == vertex:
                    targets.append(target)
                offset += 1
            block_index, offset = block_index + 1, 0
        return tuple(targets)


def in_memory_bfs(adjacency, start):
    """Oracle: plain BFS with the adjacency already in memory."""
    distance, queue = {start: 0}, collections.deque([start])
    while queue:
        vertex = queue.popleft()
        for target in adjacency.get(vertex, ()):
            if target not in distance:
                distance[target] = distance[vertex] + 1
                queue.append(target)
    return distance


def external_bfs(graph, start, memory=64, sequential=True):
    """Semi-external BFS: sorted frontier batches with an explicit memory budget and read log."""
    distance = {start: 0}
    frontier, examined, peak, passes = [start], 0, 0, []
    while frontier:
        following = []
        for begin in range(0, len(frontier), memory):
            batch = sorted(frontier[begin:begin + memory])
            peak = max(peak, len(batch))
            mark = len(graph.disk.trace)
            for vertex in batch:
                for target in graph.adjacency(vertex, sequential):
                    examined += 1
                    if target not in distance:
                        distance[target] = distance[vertex] + 1
                        following.append(target)
            passes.append([index for index, _ in graph.disk.trace[mark:]])
        frontier = following
    return distance, examined, peak, passes


def naive_bfs(graph, start):
    """Random-access baseline: FIFO frontier, every physical read charged as random."""
    distance, queue, examined = {start: 0}, collections.deque([start]), 0
    while queue:
        vertex = queue.popleft()
        for target in graph.adjacency(vertex, sequential=False):
            examined += 1
            if target not in distance:
                distance[target] = distance[vertex] + 1
                queue.append(target)
    return distance, examined


def main():
    edges = build_graph(300, 6, 40)
    reference = {}
    for source, target in edges:
        reference.setdefault(source, []).append(target)
    oracle = in_memory_bfs(reference, 0)
    fast_graph = ExternalGraph(edges)                     # partitioned: sequential in-order reads
    slow_graph = ExternalGraph(edges, cache_capacity=0)   # baseline: random reads, no cache
    partitioned, examined, peak, passes = external_bfs(fast_graph, 0)
    naive, naive_examined = naive_bfs(slow_graph, 0)

    # 1) correctness against the in-memory BFS oracle: reached set and distances.
    check(partitioned == oracle == naive and len(partitioned) == 300,
          f"partitioned traversal must match the oracle on {len(partitioned)} reached vertices")

    # 2) locality: inside every pass the directory reads and the edge reads are each non-decreasing
    #    (the file is scanned in order), while the baseline's read order is not sorted.
    offset = fast_graph.edge_offset
    ordered = all([i for i in reads if i < offset] == sorted(i for i in reads if i < offset)
                  and [i for i in reads if i >= offset] == sorted(i for i in reads
                                                                  if i >= offset)
                  for reads in passes)
    baseline_reads = [index for index, kind in slow_graph.disk.trace if kind == "random"]
    check(len(passes) >= 2 and ordered and len(baseline_reads) == slow_graph.disk.random_reads
          and baseline_reads != sorted(baseline_reads),
          f"passes must scan the file in order: {[reads[:3] for reads in passes[:4]]} "
          f"vs baseline {baseline_reads[:4]}")

    # 3) the partitioned cost model beats the random-access baseline, with fewer physical reads.
    fast_cost, slow_cost = cost_of(fast_graph.disk), cost_of(slow_graph.disk)
    check(fast_cost < slow_cost and slow_graph.disk.random_reads > 0
          and fast_graph.disk.random_reads == 0 and fast_graph.disk.misses < slow_graph.disk.misses
          and naive_examined == examined,
          f"partitioned cost {fast_cost} must beat {slow_cost} "
          f"({fast_graph.disk.misses} vs {slow_graph.disk.misses} physical reads)")

    # 4) read accounting is exact: hits + misses == requests, misses == physical reads == trace.
    disk = fast_graph.disk
    check(disk.hits + disk.misses == disk.requests
          and disk.random_reads + disk.sequential_reads == disk.misses == len(disk.trace)
          and disk.hits > 0,
          f"accounting: {disk.hits}+{disk.misses}={disk.requests} requests, {disk.misses} misses")

    # 5) memory budget and work bound: the batch stays inside the budget and each reached vertex's
    #    adjacency is examined exactly once.
    degree_sum = sum(len(reference.get(vertex, ())) for vertex in partitioned)
    check(peak <= 64 and examined == degree_sum == naive_examined and len(partitioned) == 300,
          f"work bound: examined {examined} vs degree sum {degree_sum}, peak batch {peak}")

    # 6) a bigger cache never costs more physical reads than a smaller one.
    reads = []
    for capacity in (0, 2, 8, 32):
        probe = ExternalGraph(edges, cache_capacity=capacity)
        external_bfs(probe, 0)
        reads.append(probe.disk.misses)
    check(reads == sorted(reads, reverse=True) and reads[0] > reads[-1],
          f"cache capacity must not increase physical reads: {reads}")
    # 7) boundaries: empty graph, unknown start, isolated pair.
    empty = ExternalGraph([])
    check(external_bfs(empty, 0)[0] == {0: 0} and empty.disk.requests == 0
          and naive_bfs(ExternalGraph([]), 0) == ({0: 0}, 0),
          "an empty graph must reach only the start vertex and perform no I/O")
    isolated = ExternalGraph([(7, 9)])
    check(external_bfs(isolated, 7)[0] == {7: 0, 9: 1}
          and external_bfs(isolated, 5)[0] == {5: 0} and isolated.vertices == [7],
          "an isolated pair and an unknown start vertex must both be handled")

    # 8) every start vertex matches the oracle and the union of all reaches is the vertex set.
    sweep, union, matched = ExternalGraph(edges), set(), 0
    for vertex in sweep.vertices:
        reached = external_bfs(sweep, vertex)[0]
        matched += reached == in_memory_bfs(reference, vertex)
        union |= set(reached)
    check(matched == len(sweep.vertices) and union == set(partitioned),
          f"all {matched}/{len(sweep.vertices)} start vertices must match the oracle")

    # 9) determinism.
    again, again_examined, again_peak, again_passes = external_bfs(ExternalGraph(edges), 0)
    check(again == partitioned and again_examined == examined and again_peak == peak
          and again_passes == passes, "two traversals of the same graph must be identical")
    print(f"vertices={len(sweep.vertices)} edges={len(edges)} blocks={len(fast_graph.edge_blocks)} "
          f"passes={len(passes)} peak_batch={peak}; examined={examined} "
          f"partitioned_cost={fast_cost} random_access_cost={slow_cost} "
          f"(reads {fast_graph.disk.misses} vs {slow_graph.disk.misses})")
    print("PASS")


if __name__ == "__main__":
    main()
