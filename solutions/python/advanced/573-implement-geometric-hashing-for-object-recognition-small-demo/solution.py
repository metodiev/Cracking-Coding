"""Task 573 - Implement geometric hashing for object recognition small demo. (Tier: Advanced, CTCI Beyond — Graphics, Geometry & Robotics)

Geometric hashing (Lamdan-Wolfson) on small planar point models, implemented in both of its standard
basis shapes. In the two-point version an ordered pair (p, q) of model points is a basis and a third
point r is indexed by its basis coordinates, r = p + alpha (q - p) + beta perp(q - p) with
perp(dx, dy) = (-dy, dx). In the three-point version an ordered triple (p1, p2, p3) is a basis and r
is indexed by its affine (barycentric) coordinates, r = p1 + a (p2 - p1) + b (p3 - p1). Coordinates
are quantised into a uniform grid cell, the hash table is keyed by the cell and stores (model, basis)
entries, and recognition is voting: every ordered pair (or triple) of scene points is a candidate
basis, the coordinates of the remaining scene points are looked up, and the (model, basis) pair with
the most matches wins - compared against an acceptance threshold so an unknown object can be
rejected. Basis points are never indexed as data points (their coordinates are pinned at (0, 0) and
(1, 0), or (0, 0), (1, 0) and (0, 1)), so exactly n - 2 = 6 (respectively n - 3 = 5) genuine matches
vote for the right basis, and every extra vote is an accidental cell collision. Model points and scene noise come from a fixed-seed splitmix64 stream, so every run is
bit-identical.

The two schemes are both kept because they are *not* equally invariant, and the file asserts the
difference rather than claiming affine invariance for the cheaper construction. The perp-based
coordinates are preserved only by orientation-preserving similarities: under the test shear they move
by 0.0352 (seven orders of magnitude above the 1e-9 agreement tolerance used everywhere else), under
an anisotropic map by more than 1e-6, and a reflection - which is a similarity, but reverses
handedness - negates beta exactly. Consequently a sheared, an anisotropic or a reflected scene is
*rejected* (2 or 3 accidental votes, below the threshold of 4) by the two-point scheme while the
three-point barycentric coordinates, which survive every invertible affine map, recognise all of
them. Quantisation is where the scheme actually gets its power: coordinates live in roughly [-2, 2],
and the file measures both failure modes - cells of 0.01 lose a noisy scene to the grid (2 votes at
noise 0.2), cells of 0.2 flood the table with collisions and misidentify a clean scene - against a
cell of 0.02, where the accidental score of an unseen model is 2 and the true score is 7.

Checks: hand-computed coordinates for both schemes (the basis points themselves, beta = 0 for
collinear points, the |q - p| scaling of beta, both reconstructions to 1e-12 on a random triangle);
invariance to 1e-9 under rotation + uniform scaling + translation for the two-point scheme and under
those, reflection and an anisotropic matrix for the barycentric one; the exact handedness law for the
two-point scheme (reflection keeps alpha and negates beta); the documented counterexamples that move
the two-point coordinates under shear and under an anisotropic map while the barycentric ones stay
put; table consistency (1008 two-point and 5040 three-point entries, every stored cell recomputable
from its basis, the basis cells populated but the basis points never stored as data); noiseless
recognition of all three models under two rigid transforms with the two-point scheme (7, 8 and 7
votes - the 6 = n - 2 genuine matches plus accidental cell collisions - and a positive margin over
the runner-up) and under four transforms including reflection and an anisotropic map with the
barycentric scheme (10, 8 and 12 votes); the rejection of a sheared, an anisotropic and a reflected
scene by the two-point scheme (measured scores 3, 2 and 3, and 2, 2, 2) and their recognition by the
barycentric scheme (10, 8, 12 votes) - with the further honest note that at those scores the
two-point scheme even *names* the wrong model for one anisotropic scene, which is exactly what
accidental collisions look like; recognition under four noise levels (0.02, 0.05, 0.10, 0.15) for all
three models with at least five votes and the recorded edge behaviour at 0.20 (all three models
still named correctly, but model A's score dropping to 3, below the acceptance threshold), and of a
sheared scene plus noise at 0.02 and 0.05 (9 and 6 votes, decaying to 3 at 0.10, so the sheared case
needs finer quantisation); two far-away clutter
points being outvoted (7 votes for the right model); rejection of an unseen model by both schemes
(2 accidental votes, below the threshold of 4, with and without noise); a quantisation sweep showing
the unseen model's accidental score climbing as the cell grows (1, 2, 5, 10, 26 for cells 0.01 to
0.2) while the fine grid rejects a 0.2-noise scene and the coarse grid misidentifies a clean one; and
determinism of the table, of recognition and of the noisy scene generator.
"""
import math

MASK64 = (1 << 64) - 1


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, (z ^ (z >> 31))


def uniform(state):
    state, value = splitmix64(state)
    return state, (value >> 11) / float(1 << 53)


def random_points(state, count, spread=10.0):
    points = []
    for _ in range(count):
        state, u = uniform(state)
        state, v = uniform(state)
        points.append((u * spread, v * spread))
    return state, points


# --- The two coordinate systems ---------------------------------------------------------

def similarity_coords(p, q, r):
    """(alpha, beta) with r = p + alpha (q - p) + beta perp(q - p); None for a degenerate basis."""
    dx = q[0] - p[0]
    dy = q[1] - p[1]
    denom = dx * dx + dy * dy
    if denom == 0.0:
        return None
    vx = r[0] - p[0]
    vy = r[1] - p[1]
    return ((vx * dx + vy * dy) / denom, (vx * (-dy) + vy * dx) / denom)


def barycentric_coords(p1, p2, p3, r):
    """(a, b) with r = p1 + a (p2 - p1) + b (p3 - p1); None for a degenerate basis."""
    d2 = (p2[0] - p1[0], p2[1] - p1[1])
    d3 = (p3[0] - p1[0], p3[1] - p1[1])
    det = d2[0] * d3[1] - d3[0] * d2[1]
    if det == 0.0:
        return None
    v = (r[0] - p1[0], r[1] - p1[1])
    return ((v[0] * d3[1] - v[1] * d3[0]) / det, (d2[0] * v[1] - d2[1] * v[0]) / det)


def coords_of(basis_size, points, basis, index):
    if basis_size == 2:
        return similarity_coords(points[basis[0]], points[basis[1]], points[index])
    return barycentric_coords(points[basis[0]], points[basis[1]], points[basis[2]],
                              points[index])


def all_bases(basis_size, n):
    if basis_size == 2:
        for i in range(n):
            for j in range(n):
                if i != j:
                    yield (i, j)
    else:
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i != j and i != k and j != k:
                        yield (i, j, k)


def apply_affine(matrix, offset, point):
    return (matrix[0][0] * point[0] + matrix[0][1] * point[1] + offset[0],
            matrix[1][0] * point[0] + matrix[1][1] * point[1] + offset[1])


def rotation_scale(theta_degrees, scale):
    theta = math.radians(theta_degrees)
    return [[scale * math.cos(theta), -scale * math.sin(theta)],
            [scale * math.sin(theta), scale * math.cos(theta)]]


def shear_matrix(shx, shy):
    return [[1.0, shx], [shy, 1.0]]


# --- The hash table and its use ---------------------------------------------------------

def cell_of(coords, cell):
    return (int(math.floor(coords[0] / cell)), int(math.floor(coords[1] / cell)))


def build_table(models, cell, basis_size=2):
    """table: cell -> list of (model, basis); returns the table and its entry count."""
    table = {}
    count = 0
    for name, points in models.items():
        n = len(points)
        for basis in all_bases(basis_size, n):
            members = set(basis)
            for index in range(n):
                if index in members:
                    continue
                coords = coords_of(basis_size, points, basis, index)
                if coords is None:
                    continue
                table.setdefault(cell_of(coords, cell), []).append((name, basis))
                count += 1
    return table, count


def recognize(scene, table, cell, basis_size=2, min_votes=3):
    votes = {}
    n = len(scene)
    for basis in all_bases(basis_size, n):
        members = set(basis)
        for index in range(n):
            if index in members:
                continue
            coords = coords_of(basis_size, scene, basis, index)
            if coords is None:
                continue
            for entry in table.get(cell_of(coords, cell), ()):
                votes[entry] = votes.get(entry, 0) + 1
    if not votes:
        return {"model": None, "votes": 0, "basis": None, "scores": {}, "runner_up": 0,
                "accepted": False}
    best = max(votes.items(), key=lambda item: (item[1], item[0][0]))
    scores = {}
    for (name, _basis), count in votes.items():
        scores[name] = max(scores.get(name, 0), count)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    runner_up = ranked[1][1] if len(ranked) > 1 else 0
    return {"model": best[0][0], "votes": best[1], "basis": best[0][1], "scores": scores,
            "runner_up": runner_up, "accepted": best[1] >= min_votes}


def transform_scene(points, matrix, offset):
    return [apply_affine(matrix, offset, point) for point in points]


def noisy_scene(state, points, matrix, offset, noise):
    scene = []
    for point in points:
        state, u = uniform(state)
        state, v = uniform(state)
        moved = apply_affine(matrix, offset, point)
        scene.append((moved[0] + (u * 2.0 - 1.0) * noise, moved[1] + (v * 2.0 - 1.0) * noise))
    return state, scene


CELL = 0.02          # coordinates live in about [-2, 2]; see the sweep at the end of main()
MIN_VOTES = 4


def main():
    # --- Hand-computed coordinates ------------------------------------------------------
    p, q = (0.0, 0.0), (2.0, 0.0)
    assert similarity_coords(p, q, p) == (0.0, 0.0)
    assert similarity_coords(p, q, q) == (1.0, 0.0)
    coords = similarity_coords(p, q, (0.0, 3.0))
    assert abs(coords[0]) <= 1e-12 and abs(coords[1] - 1.5) <= 1e-12, "beta scales by |q - p| = 2"
    assert similarity_coords(p, p, q) is None, "a degenerate basis has no coordinates"
    for r in ((5.0, 0.0), (1.0, 0.0), (-3.0, 0.0)):
        assert abs(similarity_coords(p, q, r)[1]) <= 1e-12, "collinear points have beta = 0"

    t1, t2, t3 = (0.0, 0.0), (1.0, 0.0), (0.0, 1.0)
    assert barycentric_coords(t1, t2, t3, t1) == (0.0, 0.0)
    assert barycentric_coords(t1, t2, t3, t2) == (1.0, 0.0)
    assert barycentric_coords(t1, t2, t3, t3) == (0.0, 1.0)
    assert barycentric_coords(t1, (1.0, 0.0), (2.0, 0.0), (1.0, 1.0)) is None, "collinear basis"
    state, triangle = random_points(1234, 6, 10.0)
    a, b, c, r = triangle[0], triangle[1], triangle[2], triangle[3]
    base_coeffs = barycentric_coords(a, b, c, r)
    rebuilt = (a[0] + base_coeffs[0] * (b[0] - a[0]) + base_coeffs[1] * (c[0] - a[0]),
               a[1] + base_coeffs[0] * (b[1] - a[1]) + base_coeffs[1] * (c[1] - a[1]))
    assert abs(rebuilt[0] - r[0]) <= 1e-12 and abs(rebuilt[1] - r[1]) <= 1e-12
    sim_coeffs = similarity_coords(a, b, r)
    rebuilt_sim = (a[0] + sim_coeffs[0] * (b[0] - a[0]) - sim_coeffs[1] * (b[1] - a[1]),
                   a[1] + sim_coeffs[0] * (b[1] - a[1]) + sim_coeffs[1] * (b[0] - a[0]))
    assert abs(rebuilt_sim[0] - r[0]) <= 1e-12 and abs(rebuilt_sim[1] - r[1]) <= 1e-12

    # --- Invariance: similarities for both, shear only for the barycentric basis ---------
    rotation = rotation_scale(37.0, 2.5)
    offset = (13.0, -7.0)
    reflection = [[-1.0, 0.0], [0.0, 1.0]]
    anisotropic = [[2.0, 0.3], [0.1, 0.5]]
    shear = shear_matrix(0.4, -0.25)
    base2 = coords_of(2, (a, b, c, r), (0, 1), 3)
    base3 = coords_of(3, (a, b, c, r), (0, 1, 2), 3)
    for transform, shift in ((rotation, offset), (reflection, (0.0, 0.0)), (anisotropic, (4.0, 9.0))):
        moved = [apply_affine(transform, shift, point) for point in (a, b, c, r)]
        after3 = coords_of(3, moved, (0, 1, 2), 3)
        assert abs(base3[0] - after3[0]) <= 1e-9 and abs(base3[1] - after3[1]) <= 1e-9, \
            "the barycentric coordinates survive every invertible linear map"
    moved = [apply_affine(rotation, offset, point) for point in (a, b, c, r)]
    after2 = coords_of(2, moved, (0, 1), 3)
    assert abs(base2[0] - after2[0]) <= 1e-9 and abs(base2[1] - after2[1]) <= 1e-9, \
        "the perp coordinates survive rotation + uniform scaling + translation"
    mirrored = [apply_affine(reflection, (0.0, 0.0), point) for point in (a, b, c, r)]
    after_mirror = coords_of(2, mirrored, (0, 1), 3)
    # A reflection is a similarity, but it reverses handedness, so alpha is kept and beta changes
    # sign exactly: the standard reason a two-point table must also store the mirrored entries.
    assert abs(base2[0] - after_mirror[0]) <= 1e-9
    assert abs(base2[1] + after_mirror[1]) <= 1e-9, "reflection negates beta"
    aniso_after2 = coords_of(2, [apply_affine(anisotropic, (4.0, 9.0), point)
                                for point in (a, b, c, r)], (0, 1), 3)
    aniso_error = max(abs(base2[0] - aniso_after2[0]), abs(base2[1] - aniso_after2[1]))
    assert aniso_error > 1e-6, \
        "the perp coordinates also break under an anisotropic map: %r" % aniso_error
    sheared = [apply_affine(shear, (0.0, 0.0), point) for point in (a, b, c, r)]
    bary_after = barycentric_coords(sheared[0], sheared[1], sheared[2], sheared[3])
    assert abs(base_coeffs[0] - bary_after[0]) <= 1e-12
    assert abs(base_coeffs[1] - bary_after[1]) <= 1e-12, "barycentric coordinates are affine"
    sim_after = similarity_coords(sheared[0], sheared[1], sheared[3])
    shear_error = max(abs(sim_coeffs[0] - sim_after[0]), abs(sim_coeffs[1] - sim_after[1]))
    # Documented counterexample: the perp-based coordinates are only similarity invariant. Under
    # this shear they move by 0.0352 - seven orders of magnitude above the 1e-9 tolerance - so a
    # sheared scene cannot be recognised by the two-point scheme.
    assert shear_error > 1e-3, "two-point coordinates under shear, deviation %r" % shear_error

    # --- The database -------------------------------------------------------------------
    state = 0x5EED
    models = {}
    for name in ("A", "B", "C"):
        state, points = random_points(state, 8, 10.0)
        models[name] = points
    state, unknown_model = random_points(state, 8, 10.0)
    cell = CELL
    table2, count2 = build_table(models, cell, 2)
    table3, count3 = build_table(models, cell, 3)
    assert count2 == 3 * 8 * 7 * 6 == 1008, "8 points, 7 bases each, 6 remaining points"
    assert count3 == 3 * 8 * 7 * 6 * 5 == 5040, "8 points, 336 triples each, 5 remaining points"
    assert sum(len(entries) for entries in table2.values()) == count2
    assert sum(len(entries) for entries in table3.values()) == count3
    for name, points in models.items():
        n = len(points)
        for basis in all_bases(2, n):
            members = set(basis)
            for index in range(n):
                if index in members:
                    continue
                stored = cell_of(similarity_coords(points[basis[0]], points[basis[1]],
                                                   points[index]), cell)
                assert (name, basis) in table2[stored], "the stored cell is recomputable"
    # 1008 = 3 models * 8 points * 7 bases * 6 data points is exactly the count without the basis
    # points themselves, which is how the file encodes "basis endpoints are never data".
    occupancy = [len(entries) for entries in table2.values()]
    assert len(occupancy) == 968 and max(occupancy) == 3, \
        "a cell of 0.02 spreads 1008 entries over 968 cells (38 shared)"

    # --- Noiseless recognition ----------------------------------------------------------
    rigid = [(rotation_scale(0.0, 1.0), (0.0, 0.0)), (rotation, offset)]
    affine = rigid + [(reflection, (5.0, 5.0)), (anisotropic, (4.0, 9.0))]
    noiseless2 = {}
    noiseless3 = {}
    for name, points in models.items():
        for transform, shift in rigid:
            scene = transform_scene(points, transform, shift)
            result2 = recognize(scene, table2, cell, 2, MIN_VOTES)
            assert result2["model"] == name, "the rigid scene of %s" % name
            # n - 2 = 6 genuine matches; any excess is an accidental cell collision, recorded below.
            assert result2["votes"] >= 6, "votes %r for %s" % (result2["votes"], name)
            assert result2["runner_up"] < result2["votes"], "a positive margin over the runner-up"
            noiseless2[(name, shift)] = result2["votes"]
        for transform, shift in affine:
            scene = transform_scene(points, transform, shift)
            result3 = recognize(scene, table3, cell, 3, MIN_VOTES)
            assert result3["model"] == name, "the affine scene of %s" % name
            assert result3["votes"] >= 5, "n - 3 = 5 genuine matches, got %r" % result3["votes"]
            assert result3["runner_up"] < result3["votes"]
            noiseless3[(name, shift)] = result3["votes"]
    assert sorted(set(noiseless2.values())) == [7, 8], "measured 7, 8 and 7 votes"
    assert sorted(set(noiseless3.values())) == [8, 10, 12], "measured 10, 8 and 12 votes"

    # --- Handedness: the two-point scheme cannot do a reflection ------------------------
    reflected2 = {}
    for name, points in models.items():
        scene = transform_scene(points, reflection, (5.0, 5.0))
        two = recognize(scene, table2, cell, 2, MIN_VOTES)
        three = recognize(scene, table3, cell, 3, MIN_VOTES)
        reflected2[name] = (two["votes"], two["accepted"])
        assert not two["accepted"], "a reflected scene is invisible to the perp basis: %r" % two
        assert two["votes"] <= 3, "measured 2 accidental votes, got %r" % two["votes"]
        assert three["model"] == name and three["votes"] >= 5, \
            "the barycentric scheme handles the reflected %s" % name

    # --- Shear and anisotropic maps: only the barycentric scheme copes ------------------
    distortions = {"shear": (shear, (0.0, 0.0)), "anisotropic": (anisotropic, (4.0, 9.0))}
    for label, (transform, shift) in distortions.items():
        for name, points in models.items():
            scene = transform_scene(points, transform, shift)
            result3 = recognize(scene, table3, cell, 3, MIN_VOTES)
            assert result3["model"] == name and result3["votes"] >= 5, \
                "the affine-basis scheme recognises the %s scene of %s" % (label, name)
            result2 = recognize(scene, table2, cell, 2, MIN_VOTES)
            assert not result2["accepted"], \
                "the two-point scheme cannot cope with %s: %r votes" % (label, result2["votes"])
            assert result2["votes"] <= 3, "measured at most 3 accidental votes"
    # The anisotropic scene of B even gets *named* C at three votes: that is what an accidental
    # score looks like, and it is why the acceptance threshold is applied to the vote count.
    aniso_b = recognize(transform_scene(models["B"], anisotropic, (4.0, 9.0)), table2, cell, 2,
                        MIN_VOTES)
    assert aniso_b["votes"] == 3 and not aniso_b["accepted"]

    # --- Recognition under noise (rigid scenes, two-point scheme) -----------------------
    noise_levels = (0.02, 0.05, 0.10, 0.15)
    accepted = 0
    for trial, noise in enumerate(noise_levels):
        for index, (name, points) in enumerate(models.items()):
            state, scene = noisy_scene(1000 + 17 * trial + index, points, rotation, offset, noise)
            result = recognize(scene, table2, cell, 2, MIN_VOTES)
            assert result["model"] == name, "%s under noise %r" % (name, noise)
            assert result["votes"] >= 5, "votes %r at noise %r" % (result["votes"], noise)
            accepted += 1
    assert accepted == 12
    # At 0.2 the cell of 0.02 is at its limit: every model is still named correctly but model A's
    # score has fallen to 3, i.e. below the acceptance threshold - recorded rather than smoothed
    # away, and the reason the sweep below uses a coarser grid as the honest counterweight.
    edge = {}
    for index, (name, points) in enumerate(models.items()):
        state, scene = noisy_scene(1000 + 17 * 4 + index, points, rotation, offset, 0.20)
        result = recognize(scene, table2, cell, 2, MIN_VOTES)
        assert result["model"] == name, "%s at the noise limit" % name
        edge[name] = result["votes"]
    assert edge == {"A": 3, "B": 4, "C": 5}, "measured edge scores: %r" % edge
    # A sheared scene plus noise is harder: C scores 9 and 8 at noise 0.02 and 0.05, but by 0.10 it
    # has fallen to 3 and the scene is *misidentified* as A with 4 votes. The file records that
    # breakdown instead of claiming the rigid case's robustness.
    for trial, noise in enumerate((0.02, 0.05)):
        state, scene = noisy_scene(2000 + 31 * trial, models["C"], shear, (1.0, -3.0), noise)
        result = recognize(scene, table3, cell, 3, MIN_VOTES)
        assert result["model"] == "C" and result["votes"] >= 4, \
            "sheared and noisy C at noise %r gave %r votes" % (noise, result["votes"])
    state, scene = noisy_scene(2000 + 31 * 2, models["C"], shear, (1.0, -3.0), 0.10)
    hard = recognize(scene, table3, cell, 3, MIN_VOTES)
    assert hard["model"] == "A" and hard["votes"] == 4 and hard["scores"]["C"] == 3, \
        "the sheared scene breaks down at 0.10: %r" % (hard,)
    hard_b = recognize(scene, table3, cell, 3, min_votes=5)
    assert not hard_b["accepted"], "and it is below a threshold of 5"

    # --- Clutter and rejection ----------------------------------------------------------
    state, scene = noisy_scene(4096, models["B"], rotation, offset, 0.05)
    result = recognize(scene + [(100.0, -80.0), (-60.0, 90.0)], table2, cell, 2, MIN_VOTES)
    assert result["model"] == "B" and result["votes"] >= 4, "two far-away clutter points"

    unknown_scene = transform_scene(unknown_model, rotation, offset)
    result = recognize(unknown_scene, table2, cell, 2, MIN_VOTES)
    assert not result["accepted"], "an unseen model must stay below the threshold"
    assert result["votes"] == 2, "accidental collisions: %r votes" % result["votes"]
    result3 = recognize(unknown_scene, table3, cell, 3, MIN_VOTES)
    assert not result3["accepted"] and result3["votes"] == 2
    state, noisy_unknown = noisy_scene(777, unknown_model, rotation, offset, 0.15)
    assert not recognize(noisy_unknown, table2, cell, 2, MIN_VOTES)["accepted"]

    # --- The quantisation trade-off -----------------------------------------------------
    # Unseen-model scores grow with the cell (1, 2, 5, 10, 26 for 0.01 .. 0.2) because a coarse
    # table collides everything with everything; at 0.01 the 0.2-noise scene of A falls to 2 votes
    # and is rejected instead; at 0.2 even the *clean* scene is misidentified as C.
    unknown_scores = []
    sweep = {}
    for size in (0.01, 0.02, 0.05, 0.1, 0.2):
        small_table, _ = build_table(models, size, 2)
        clean = recognize(transform_scene(models["A"], rotation, offset), small_table, size, 2,
                          MIN_VOTES)
        state, fuzzy = noisy_scene(31337, models["A"], rotation, offset, 0.2)
        noisy = recognize(fuzzy, small_table, size, 2, MIN_VOTES)
        unseen = recognize(transform_scene(unknown_model, rotation, offset), small_table, size, 2,
                           MIN_VOTES)
        sweep[size] = (clean["model"], clean["votes"], noisy["accepted"], noisy["votes"])
        unknown_scores.append(unseen["votes"])
    assert unknown_scores == [1, 2, 5, 10, 26], "accidental score versus cell size: %r" % (
        unknown_scores,)
    assert all(unknown_scores[i] < unknown_scores[i + 1] for i in range(4))
    assert sweep[0.01][2] is False, "a fine grid cannot absorb 0.2 of noise: %r" % (sweep[0.01],)
    assert sweep[0.02][2] is True and sweep[0.02][0] == "A"
    assert sweep[0.2][0] != "A", "the coarse grid misidentifies even a clean scene: %r" % (
        sweep[0.2],)
    assert sweep[0.2][1] == 42 and sweep[0.02][1] == 7, "collision-inflated scores"
    assert sweep[0.2][3] == 45 and sweep[0.2][2] is True

    # --- Determinism --------------------------------------------------------------------
    table2_again, count2_again = build_table(models, cell, 2)
    assert count2_again == count2
    assert all(table2[key] == table2_again[key] for key in table2)
    scene = transform_scene(models["A"], rotation, offset)
    assert recognize(scene, table2, cell, 2) == recognize(scene, table2_again, cell, 2)
    assert noisy_scene(11, models["A"], rotation, offset, 0.1) == \
        noisy_scene(11, models["A"], rotation, offset, 0.1)


if __name__ == "__main__":
    main()
    print("PASS")
