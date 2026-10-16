"""Task 772 - Implement dynamic planar graph algorithms (face updates) outline.  (Tier: Expert, Ch 4 - Trees and Graphs)

The task is face maintenance for a *dynamically* growing planar embedding: an edge can only be
inserted when both endpoints already lie on the boundary of one common face, because then the edge
can be drawn inside that face and it splits that face in two.  The structure kept here is the
rotation system of the embedding: every vertex stores its neighbours in cyclic (clockwise) order.
From the rotation system the faces are recovered by the standard dart walk
    next(u -> v) = (v -> predecessor of u in the rotation of v),
so a face is a cycle of darts (tail, head) and every dart belongs to exactly one face.

Face update on inserting the chord u-v in a face F located by that walk: if the dart arriving at u
along F is (r -> u) and the one arriving at v is (z -> v), then v must be spliced into the rotation
of u immediately *before* r and u immediately before z in the rotation of v.  A direct substitution
into the walk rule then reproduces exactly the two boundary cycles
    F1 = (u -> v) + [walk after z ... r],   F2 = (v -> u) + [walk after r ... z],
which are the two faces of the split; a new vertex attached to v inside F simply becomes a spike in
F's walk (a pendant edge does not change the face count).  Euler's formula V - E + F = 2C (C = the
number of components of the tracked graph, isolated vertices are not tracked) is verified after
every insertion.

What is simplified (outline): faces are located by an O(E) walk instead of a logarithmic point
location structure, endpoints must be identified by the caller (no geometric input), and an edge
between two vertices that share no face is rejected rather than re-embedded, so the embedding never
changes during a step.  What is *not* simplified: the rotation-system update and the face split are
real, and they are verified against the dart-walk tracer after every single edge, plus Euler's
formula, the maximal-planar edge/face counts (E = 3V - 6, F = 2V - 4), triangularity of every face
of a triangulation, and rejection of a non-planar insertion (K5 must stop at 9 edges).
"""
import itertools
import random


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def darts(rot):
    """All darts of the rotation system in a deterministic order."""
    return [(u, v) for u in sorted(rot) for v in rot[u]]


def trace_faces(rot):
    """Recover the face cycles (lists of darts) of the embedding defined by the rotation system."""
    order = darts(rot)
    next_dart = {}
    for u, v in order:
        ring = rot[v]
        next_dart[(u, v)] = (v, ring[ring.index(u) - 1])     # predecessor in the rotation at v
    faces = []
    visited = set()
    for start in order:
        if start in visited:
            continue
        walk = []
        current = start
        while current not in visited:
            visited.add(current)
            walk.append(current)
            current = next_dart[current]
        faces.append(walk)
    return faces


def canonical(walk):
    """Rotate a cyclic dart walk to its lexicographically smallest rotation (comparable form)."""
    rotations = [tuple(walk[k:] + walk[:k]) for k in range(len(walk))]
    return min(rotations)


def canonical_set(faces):
    return sorted(canonical(walk) for walk in faces)


class PlanarEmbedding:
    """Dynamic planar embedding: rotation system + face updates + Euler/face invariants."""

    def __init__(self):
        self.rot = {}                 # vertex -> cyclic neighbour list
        self.edges = set()            # frozenset({u, v})
        self.components = 0
        self.splits = 0               # chord insertions that created a new face
        self.spikes = 0               # pendant insertions that kept the face count
        self.rejected = 0
        self.last_expected_faces = None

    # ---------------------------------------------------------------- queries
    def vertices(self):
        return sorted(self.rot)

    def edge_count(self):
        return len(self.edges)

    def faces(self):
        return trace_faces(self.rot)

    def face_count(self):
        if not self.edges:
            return 1 if self.rot else 0
        return len(self.faces())

    def euler(self):
        return len(self.vertices()) - len(self.edges) + self.face_count()

    # ---------------------------------------------------------------- updates
    def add_edge(self, u, v):
        """Insert the undirected edge u-v when it lies inside a face; returns True/False."""
        if u == v:
            self.rejected += 1
            return False
        key = frozenset((u, v))
        if key in self.edges:
            self.rejected += 1
            return False
        fresh_u, fresh_v = u not in self.rot, v not in self.rot
        if fresh_u and fresh_v:
            self.rot[u] = [v]
            self.rot[v] = [u]
            self.edges.add(key)
            self.components += 1
            self.last_expected_faces = None      # a new component simply adds one outer face
            return True
        if fresh_u or fresh_v:
            return self._attach_pendant(v if fresh_u else u, u if fresh_u else v)
        walks = self.faces()
        for face_index, walk in enumerate(walks):
            index_u = next((k for k, dart in enumerate(walk) if dart[1] == u), None)
            index_v = next((k for k, dart in enumerate(walk) if dart[1] == v), None)
            if index_u is None or index_v is None:
                continue
            self._insert_chord(walks, face_index, index_u, index_v)
            return True
        self.rejected += 1                            # no common face: leave the embedding alone
        return False

    def _attach_pendant(self, anchor, fresh):
        """Attach a brand-new vertex to `anchor`, as a spike inside the first face incident to it."""
        walks = self.faces()
        for face_index, walk in enumerate(walks):
            for index, dart in enumerate(walk):
                if dart[1] == anchor:
                    z = dart[0]
                    ring = self.rot[anchor]
                    ring.insert(ring.index(z), fresh)
                    self.rot[fresh] = [anchor]
                    self.edges.add(frozenset((anchor, fresh)))
                    expected = list(walks)
                    expected[face_index] = (walk[:index + 1] + [(anchor, fresh), (fresh, anchor)]
                                            + walk[index + 1:])
                    self.last_expected_faces = canonical_set(expected)
                    self.spikes += 1
                    return True
        self.rejected += 1
        return False

    def _insert_chord(self, walks, face_index, index_u, index_v):
        """Splice the chord (head(walk[index_u]), head(walk[index_v])) into the rotation system."""
        walk = walks[face_index]
        dart_u, dart_v = walk[index_u], walk[index_v]
        u, v = dart_u[1], dart_v[1]
        r, z = dart_u[0], dart_v[0]
        ring_u, ring_v = self.rot[u], self.rot[v]
        ring_u.insert(ring_u.index(r), v)             # v immediately before r at u
        ring_v.insert(ring_v.index(z), u)             # u immediately before z at v
        self.edges.add(frozenset((u, v)))

        size = len(walk)
        arc_v_to_u = [walk[(index_v + k) % size] for k in range(0, (index_u - index_v) % size + 1)]
        arc_u_to_v = [walk[(index_u + k) % size] for k in range(0, (index_v - index_u) % size + 1)]
        expected = list(walks)
        expected[face_index] = [(u, v)] + arc_v_to_u[1:]
        expected.append([(v, u)] + arc_u_to_v[1:])
        self.last_expected_faces = canonical_set(expected)
        self.splits += 1
        return True

    def triangulation_step(self):
        """Add one chord across a face with at least four sides; returns the new edge or None."""
        walks = self.faces()
        for face_index, walk in enumerate(walks):
            size = len(walk)
            if size < 4:
                continue
            for i in range(size):
                for j in range(i + 2, size):
                    u, v = walk[i][1], walk[j][1]
                    if u == v or frozenset((u, v)) in self.edges:
                        continue
                    self._insert_chord(walks, face_index, i, j)
                    return (u, v)
        return None


def main():
    # 1. boundary: an empty embedding has no darts; one edge gives a single two-dart face
    empty = PlanarEmbedding()
    check(empty.vertices() == [] and empty.edges == set(), "a fresh embedding must be empty")
    check(empty.faces() == [] and empty.face_count() == 0, "an empty embedding has no face")
    check(empty.add_edge(3, 3) is False, "self-loops must be rejected")
    check(empty.add_edge(1, 2) is True, "the very first edge must be accepted")
    check(empty.euler() == 2 and empty.face_count() == 1, "a single edge has one face (Euler = 2)")
    check(len(empty.faces()[0]) == 2, "the bridge face must consist of exactly two darts")
    check(empty.add_edge(1, 2) is False, "duplicate edges must be rejected")
    check(empty.rejected == 2, "both invalid insertions must be counted as rejections")

    # 2. triangle: the one chord that closes the 3-cycle splits the face, Euler stays 2
    tri = PlanarEmbedding()
    check(tri.add_edge(0, 1) and tri.add_edge(1, 2) is True, "a path must be insertable")
    check(tri.face_count() == 1, "a path keeps a single face")
    check(tri.add_edge(2, 0) is True, "closing the triangle inside the face must work")
    check(tri.face_count() == 2, "the triangle must have an inner and an outer face")
    check(all(len(walk) == 3 for walk in tri.faces()), "both triangle faces have three darts")
    check(tri.euler() == 2, "Euler's formula must hold for the triangle")

    # 3. the face update is exactly the documented split of the previous face walk
    chain = PlanarEmbedding()
    for step in range(6):
        before = canonical_set(chain.faces()) if chain.edges else []
        added = chain.add_edge(step, step + 1)
        check(added, "building a chain must succeed")
        if chain.last_expected_faces is not None:
            check(canonical_set(chain.faces()) == chain.last_expected_faces,
                  f"the traced faces diverged from the expected split at step {step}")
        else:
            check(len(before) == 0, "the first edge starts from an empty face set")
        check(chain.euler() == 2, "a tree keeps Euler = 2")
    check(chain.face_count() == 1, "a chain is a bridge-only graph with one face")
    check(chain.spikes == 5, "each pendant attachment must be recorded as a spike")
    check(chain.splits == 0, "a tree never splits a face")

    # 4. square: the fourth edge of the cycle closes the face and splits it
    square = PlanarEmbedding()
    for u, v in ((0, 1), (1, 2), (2, 3)):
        check(square.add_edge(u, v), "square edges must be insertable")
    check(square.face_count() == 1, "the open path still has one face")
    check(square.add_edge(3, 0), "the closing edge must be insertable")
    check(square.face_count() == 2 and square.splits == 1, "closing the square must split one face")
    check(square.euler() == 2, "the square must satisfy Euler")
    check(sorted(len(walk) for walk in square.faces()) == [4, 4], "both faces are quadrilaterals")
    check(canonical_set(square.faces()) == square.last_expected_faces,
          "the closing split must match the expected walks")

    # 5. K4: both diagonals fit, because the second one is drawn in the outer face
    check(square.add_edge(0, 2) is True, "the first diagonal must split a quadrilateral face")
    check(square.face_count() == 3, "adding a diagonal turns the square into two triangles")
    check(square.rejected == 0, "no insertion may have failed yet")
    check(sorted(len(walk) for walk in square.faces()) == [3, 3, 4],
          "the outer face stays a quadrilateral")
    check(square.add_edge(1, 3) is True, "the second diagonal goes into the outer face (K4)")
    check(square.face_count() == 4 and square.euler() == 2, "K4 has four faces and Euler = 2")
    check(all(len(walk) == 3 for walk in square.faces()), "K4 is fully triangulated")
    check(square.edge_count() == 3 * 4 - 6, "K4 saturates the planar edge bound 3V - 6")
    check(square.add_edge(0, 1) is False and square.rejected == 1, "duplicates must be rejected")
    check(square.euler() == 2, "Euler must survive a rejected insertion")

    # 6. triangulation of a hexagon: every face becomes a triangle, counts hit 3V - 6 / 2V - 4
    tri_graph = PlanarEmbedding()
    for step in range(6):
        check(tri_graph.add_edge(step, (step + 1) % 6), "hexagon edges must all be insertable")
    check(tri_graph.face_count() == 2 and tri_graph.edge_count() == 6, "the hexagon is a 6-cycle")
    triangle_edges = 0
    while True:
        added = tri_graph.triangulation_step()
        if added is None:
            break
        triangle_edges += 1
        check(canonical_set(tri_graph.faces()) == tri_graph.last_expected_faces,
              "each triangulation chord must produce the expected face split")
        check(tri_graph.euler() == 2, "Euler must hold after every triangulation chord")
    check(triangle_edges == 6, f"a hexagon needs six chords to triangulate, got {triangle_edges}")
    check(all(len(walk) == 3 for walk in tri_graph.faces()),
          "every face of the triangulation must be a triangle")
    vertices = len(tri_graph.vertices())
    check(tri_graph.edge_count() == 3 * vertices - 6,
          "a triangulation must have exactly 3V - 6 edges")
    check(tri_graph.face_count() == 2 * vertices - 4, "a triangulation must have exactly 2V - 4 faces")

    # 7. K5 must stop: five vertices admit at most 3V - 6 = 9 edges before insertions fail
    clique = PlanarEmbedding()
    accepted = 0
    for u, v in itertools.combinations(range(5), 2):
        if clique.add_edge(u, v):
            accepted += 1
            check(clique.euler() == 2, "every accepted K5 edge must keep the embedding planar")
            if clique.last_expected_faces is not None:   # the very first edge opens a component
                check(canonical_set(clique.faces()) == clique.last_expected_faces,
                      "each accepted K5 edge must produce the expected split")
    check(accepted == 9, f"K5 must stop at 9 edges, got {accepted}")
    check(clique.edge_count() == 9, "the tracked edge set must match the accepted count")
    for u, v in itertools.combinations(range(5), 2):
        if frozenset((u, v)) in clique.edges:
            continue
        check(clique.add_edge(u, v) is False, "the tenth K5 edge must be rejected")
    check(all(len(walk) == 3 for walk in clique.faces()),
          "a maximal planar graph on 5 vertices must be fully triangulated")
    check(clique.face_count() == 2 * 5 - 4, "maximal planarity needs F = 2V - 4")

    # 8. random insertion orders: Euler + expected split after every step, no exceptions
    rng = random.Random(20240607)
    for trial in range(6):
        embed = PlanarEmbedding()
        for _ in range(60):
            u, v = rng.randrange(9), rng.randrange(9)
            if embed.add_edge(u, v):
                if embed.last_expected_faces is not None:
                    check(canonical_set(embed.faces()) == embed.last_expected_faces,
                          "random insertion must produce the expected face update")
                check(embed.euler() == 2 * embed.components,
                      f"Euler must equal 2 * components (trial {trial})")
        for walk in embed.faces():
            size = len(walk)
            for k, dart in enumerate(walk):
                check(dart[1] == walk[(k + 1) % size][0],
                      "consecutive darts of a face must chain tail-to-head")
            check(dart[1] != dart[0], "no face may contain a self-loop")

    # 9. an edge between two components is rejected (it belongs to no common face)
    split_graph = PlanarEmbedding()
    split_graph.add_edge(0, 1)
    split_graph.add_edge(2, 3)
    check(split_graph.components == 2, "two disjoint edges must be two components")
    check(split_graph.euler() == 2 * split_graph.components, "Euler must count both components")
    check(split_graph.add_edge(0, 2) is False, "a bridge across components has no common face")
    check(split_graph.edge_count() == 2, "the rejected edge must not be recorded")

    # 10. determinism
    repeat = PlanarEmbedding()
    repeat.add_edge(0, 1)
    repeat.add_edge(1, 2)
    repeat.add_edge(2, 0)
    mirror = PlanarEmbedding()
    mirror.add_edge(0, 1)
    mirror.add_edge(1, 2)
    mirror.add_edge(2, 0)
    check(repeat.rot == mirror.rot, "identical insertions must give identical rotation systems")
    check(canonical_set(repeat.faces()) == canonical_set(mirror.faces()), "faces must be identical")

    print(f"772 v={len(tri_graph.vertices())} e={tri_graph.edge_count()} f={tri_graph.face_count()} "
          f"splits={tri_graph.splits} spikes={tri_graph.spikes}")
    print(f"772 k5_edges={clique.edge_count()} rejected={clique.rejected} "
          f"square_faces={[len(w) for w in square.faces()]}")
    print("PASS")


if __name__ == "__main__":
    main()
