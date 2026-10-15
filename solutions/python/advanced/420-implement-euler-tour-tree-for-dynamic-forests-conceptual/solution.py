"""Task 420 - Implement Euler tour tree for dynamic forests (conceptual). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

An Euler tour tree stores one Euler circuit of each tree as a sequence of DIRECTED EDGE
events (u -> v): travelling the circuit enters u and leaves towards v, so the sequence
chains (each event's target is the next event's source, cyclically) and every directed
edge of the tree appears exactly once. A real ETT keeps these sequences in balanced
search trees and links/cuts by splitting/merging them in O(log n); this conceptual
implementation uses plain lists, so each operation costs O(size) — the operations are
identical, only the balance is simulated away (documented deviation).

  * connect(u, v): rotate u's tour to start leaving u and v's tour to start leaving v,
    then splice:  tour(u) + [u->v] + tour(v) + [v->u]  — a valid circuit of the merged tree.
  * disconnect(u, v) (edge exists): rotate to start with [u->v], find the matching
    [v->u]; the events strictly between them form the v-side circuit, the rest (after
    [v->u]) forms the u-side circuit.
  * reroot(v): rotate v's circuit so it starts with an event leaving v.
  * connected / component_vertices: answered from the owner map (a list simulation of
    the tree lookup a real ETT answers via the roots of its balanced trees).
"""


class EulerTourForest:
    def __init__(self, n):
        self.n = n
        self.tours = {v: [] for v in range(1, n + 1)}
        self.owner = {v: v for v in range(1, n + 1)}
        self.next_id = n + 1

    # ---- helpers -----------------------------------------------------------------
    def tour_of(self, x):
        return self.tours[self.owner[x]]

    @staticmethod
    def _rotate(tour, x):
        """Rotate so the sequence starts with an event leaving x (no-op if isolated)."""
        for idx, (src, _) in enumerate(tour):
            if src == x:
                return tour[idx:] + tour[:idx]
        return list(tour)

    def _reassign(self, tour_id, members):
        for vert in members:
            self.owner[vert] = tour_id

    def _endpoints(self, tour):
        verts = set()
        for src, dst in tour:
            verts.add(src)
            verts.add(dst)
        return verts

    # ---- operations --------------------------------------------------------------
    def connect(self, u, v):
        """Link u and v (must be in different components); returns True when linked."""
        if self.owner[u] == self.owner[v]:
            return False
        tu, tv = self.tour_of(u), self.tour_of(v)
        merged = self._rotate(tu, u) + [(u, v)] + self._rotate(tv, v) + [(v, u)]
        old = {self.owner[u], self.owner[v]}
        new_id = self.next_id
        self.next_id += 1
        for tid in old:
            del self.tours[tid]
        self.tours[new_id] = merged
        members = self._endpoints(merged)
        members.add(u)
        members.add(v)
        self._reassign(new_id, members)
        return True

    def disconnect(self, u, v):
        """Cut the edge {u, v} when it exists; returns True when cut."""
        if self.owner[u] != self.owner[v]:
            return False
        tour = self.tour_of(u)
        forward = None
        for idx, event in enumerate(tour):
            if event == (u, v):
                forward = idx
                break
        reverse = None
        for idx, event in enumerate(tour):
            if event == (v, u):
                reverse = idx
                break
        if forward is None or reverse is None:
            return False
        tour = tour[forward:] + tour[:forward]          # starts with u -> v
        q = tour.index((v, u))
        side_v = tour[1:q]                              # closed walk at v
        side_u = tour[q + 1:]                           # closed walk at u
        old_id = self.owner[u]
        del self.tours[old_id]
        id_u = self.next_id
        self.next_id += 1
        id_v = self.next_id
        self.next_id += 1
        self.tours[id_u] = side_u
        self.tours[id_v] = side_v
        self._reassign(id_u, self._endpoints(side_u) | {u})
        self._reassign(id_v, self._endpoints(side_v) | {v})
        return True

    def reroot(self, x):
        tid = self.owner[x]
        self.tours[tid] = self._rotate(self.tours[tid], x)

    def connected(self, u, v):
        return self.owner[u] == self.owner[v]

    def component_vertices(self, x):
        members = self._endpoints(self.tours[self.owner[x]])
        members.add(x)
        return sorted(members)

    def component_size(self, x):
        return len(self.component_vertices(x))


class OracleForest:
    """Adjacency oracle with BFS components."""

    def __init__(self, n):
        self.adj = {v: set() for v in range(1, n + 1)}

    def link(self, u, v):
        self.adj[u].add(v)
        self.adj[v].add(u)

    def cut(self, u, v):
        self.adj[u].remove(v)
        self.adj[v].remove(u)

    def has_edge(self, u, v):
        return v in self.adj[u]

    def component(self, x):
        seen = {x}
        queue = [x]
        while queue:
            cur = queue.pop()
            for nxt in self.adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return seen

    def connected(self, u, v):
        return v in self.component(u)


class Lcg:
    def __init__(self, state):
        self.state = state

    def next(self):
        self.state = (self.state * 75 + 74) % 65537
        return self.state


def validate(forest, oracle, label):
    """Every component's tour must be a valid Euler circuit of the oracle tree."""
    seen_vertices = set()
    for start in range(1, forest.n + 1):
        if start in seen_vertices:
            continue
        members = oracle.component(start)
        seen_vertices |= members
        tour = forest.tour_of(start)
        if len(members) == 1:
            assert tour == [], (label, start)
            continue
        assert len(tour) == 2 * (len(members) - 1), (label, start, len(tour))
        for i in range(len(tour)):
            assert tour[i][1] == tour[(i + 1) % len(tour)][0], (label, start, i)
        assert len(set(tour)) == len(tour), (label, start)
        covered = set()
        for (a, b) in tour:
            assert oracle.has_edge(a, b), (label, a, b)
            covered.add(a)
            covered.add(b)
        assert covered == members, (label, start)
        assert sorted(forest.component_vertices(start)) == sorted(members), (label, start)
        assert forest.component_size(start) == len(members), (label, start)


def main():
    # hand trace on three vertices
    forest = EulerTourForest(3)
    assert forest.connected(1, 1) and not forest.connected(1, 2)
    assert forest.component_vertices(1) == [1]
    assert forest.connect(1, 2)
    assert forest.tour_of(1) == [(1, 2), (2, 1)]
    assert forest.connect(2, 3)
    # tour of the walk 1 -> 2 -> 3 -> 2 -> 1, rotated to start leaving 2
    joined = forest.tour_of(2)
    assert joined == [(2, 1), (1, 2), (2, 3), (3, 2)], joined
    assert forest.connected(1, 3) and forest.component_size(1) == 3
    assert forest.component_vertices(1) == [1, 2, 3]

    # reroot rotates the circuit so that it starts leaving the new root
    forest.reroot(3)
    assert forest.tour_of(3) == [(3, 2), (2, 1), (1, 2), (2, 3)], forest.tour_of(3)

    # cut edge 2-3: the v-side is the singleton {3}, the rest keeps {1, 2}
    assert forest.disconnect(2, 3)
    assert not forest.connected(1, 3)
    assert forest.connected(1, 2)
    assert forest.tour_of(3) == []
    side = forest.tour_of(1)
    assert sorted(side) == [(1, 2), (2, 1)], side
    assert forest.component_vertices(3) == [3]

    # cutting again without the edge is a no-op
    assert not forest.disconnect(2, 3)
    assert forest.connected(1, 2)

    # a non-adjacent pair cannot be disconnected even when connected
    forest.connect(2, 3)
    assert forest.connected(1, 3)
    assert not forest.disconnect(1, 3)       # 1-2-3 path, no 1-3 edge
    assert forest.connected(1, 3)
    assert forest.disconnect(1, 2)
    assert not forest.connected(1, 2) and forest.connected(2, 3)

    # scripted operations against the union-find + tree oracle
    n = 25
    gen = Lcg(4200)
    forest = EulerTourForest(n)
    oracle = OracleForest(n)
    for step in range(320):
        kind = gen.next() % 4
        u = 1 + gen.next() % n
        v = 1 + gen.next() % n
        if kind == 0:
            if u != v and not oracle.connected(u, v):
                assert forest.connect(u, v), step
                oracle.link(u, v)
                assert forest.connected(u, v), step
        elif kind == 1:
            if u != v and oracle.has_edge(u, v):
                assert forest.disconnect(u, v), step
                oracle.cut(u, v)
                assert not forest.connected(u, v), step
        elif kind == 2:
            assert forest.connected(u, v) == oracle.connected(u, v), step
            assert forest.component_vertices(u) == sorted(oracle.component(u)), step
            assert forest.component_size(u) == len(oracle.component(u)), step
        else:
            forest.reroot(u)
            assert forest.connected(u, v) == oracle.connected(u, v), step
        # structural validity after every single operation
        validate(forest, oracle, step)
        # incremental spot checks on fresh random pairs
        for _ in range(3):
            a = 1 + gen.next() % n
            b = 1 + gen.next() % n
            assert forest.connected(a, b) == oracle.connected(a, b), step

    # final full audit: every pair, every component
    for u in range(1, n + 1):
        for v in range(u + 1, n + 1):
            assert forest.connected(u, v) == oracle.connected(u, v), (u, v)
    validate(forest, oracle, "final")
    # final shape: the deterministic script leaves one giant component plus a few
    # cut-off singletons; assert the giant exists (exact matching is oracle-checked)
    largest = max(len(oracle.component(v)) for v in range(1, n + 1))
    assert largest >= n - 5, largest


if __name__ == "__main__":
    main()
    print("PASS")
