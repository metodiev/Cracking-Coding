"""Task 460 - Implement small epsilon-approximate nearest neighbor in high-dim (LSH idea). (Tier: Advanced, CTCI Beyond)

Two answers to nearest neighbour in high dimension:
* brute force: scan all points (exact, serves as the oracle / certified answer).
* LSH-style bucketing: T hash tables, each hashing an integer projection a.x + b into floor
  buckets of width r; the query collects the union of its T buckets as candidates and returns
  the closest one. When every bucket is empty (the query is far outside the data range) the demo
  falls back to the brute-force scan, which is exact by construction. Everything is integer
  arithmetic (floor division), so all four language ports agree bit for bit.

Assertions: the returned point obeys the (1+eps) epsilon-guarantee with eps = 0.5, checked as
4 * d_approx^2 <= 9 * d_true^2 in exact integers (division-free), for every LCG query and for a
hand example {(0..),(10,0..),(0,10,0..)} with query (9,0..) whose true NN is at distance 1; the
answer is never closer than the true nearest neighbour; the bucket path is exercised (queries
answered from buckets) and the fallback path is exercised (the far outlier query); collisions
exist (some bucket holds >= 2 points).
"""

DIM = 10
TABLES = 6
BUCKET_WIDTH = 700
NUM_POINTS = 40


def lcg(seed):
    return (seed * 75 + 74) % 65537


def floor_div(num, den):
    return num // den


def squared_distance(p, q):
    total = 0
    for i in range(len(p)):
        diff = p[i] - q[i]
        total += diff * diff
    return total


def make_points(count, seed):
    points = []
    for _ in range(count):
        coords = []
        for _ in range(DIM):
            seed = lcg(seed)
            coords.append(seed % 61 - 30)
        points.append(coords)
    return points, seed


def make_tables(seed):
    tables = []
    for _ in range(TABLES):
        projection = []
        for _ in range(DIM):
            seed = lcg(seed)
            projection.append(1 + seed % 20)
        seed = lcg(seed)
        offset = seed % BUCKET_WIDTH
        buckets = {}
        tables.append((projection, offset, buckets))
    return tables, seed


def hash_of(table, point):
    projection, offset, _ = table
    total = offset
    for i in range(DIM):
        total += projection[i] * point[i]
    return floor_div(total, BUCKET_WIDTH)


def build_index(points, tables):
    for table in tables:
        for idx, point in enumerate(points):
            table[2].setdefault(hash_of(table, point), []).append(idx)


def brute_force_nn(points, query):
    """Exact nearest neighbour (the epsilon-oracle / certified scan)."""
    best_idx = 0
    best_dist = squared_distance(points[0], query)
    for idx in range(1, len(points)):
        dist = squared_distance(points[idx], query)
        if dist < best_dist:
            best_idx = idx
            best_dist = dist
    return best_idx, best_dist


def lsh_approx_nn(points, tables, query):
    """Returns (index, distance, used_bucket). Falls back to exact when all buckets empty."""
    candidates = set()
    for table in tables:
        candidates.update(table[2].get(hash_of(table, query), []))
    if not candidates:
        idx, dist = brute_force_nn(points, query)
        return idx, dist, False
    best_idx = None
    best_dist = None
    for idx in sorted(candidates):
        dist = squared_distance(points[idx], query)
        if best_dist is None or dist < best_dist:
            best_idx = idx
            best_dist = dist
    return best_idx, best_dist, True


def bucket_sizes(tables):
    sizes = []
    for table in tables:
        for bucket in table[2].values():
            sizes.append(len(bucket))
    return sizes


def check_query(points, tables, query, counters, label):
    true_idx, true_dist = brute_force_nn(points, query)
    approx_idx, approx_dist, used_bucket = lsh_approx_nn(points, tables, query)
    assert approx_dist >= true_dist, label
    assert 4 * approx_dist <= 9 * true_dist, (label, approx_dist, true_dist)
    assert squared_distance(points[approx_idx], query) == approx_dist
    if used_bucket:
        counters["bucket"] += 1
    else:
        counters["fallback"] += 1
    return true_idx, true_dist, approx_idx, approx_dist, used_bucket


def main():
    points, seed = make_points(NUM_POINTS, 4601)
    tables, seed = make_tables(seed)
    build_index(points, tables)

    sizes = bucket_sizes(tables)
    assert len(sizes) > 0
    assert max(sizes) >= 2

    counters = {"bucket": 0, "fallback": 0}
    for q in range(12):
        query = []
        for _ in range(DIM):
            seed = lcg(seed)
            query.append(seed % 61 - 30)
        check_query(points, tables, query, counters, "LCG query %d" % q)

    outlier = [10000] * DIM
    true_idx, true_dist, approx_idx, approx_dist, used_bucket = check_query(
        points, tables, outlier, counters, "outlier")
    assert not used_bucket
    assert approx_idx == true_idx and approx_dist == true_dist

    hand_points = [[0] * DIM, [10] + [0] * (DIM - 1), [0, 10] + [0] * (DIM - 2)]
    hand_tables, _ = make_tables(99999)
    build_index(hand_points, hand_tables)
    hand_query = [9] + [0] * (DIM - 1)
    hand_idx, hand_dist = brute_force_nn(hand_points, hand_query)
    assert hand_idx == 1 and hand_dist == 1
    approx_idx, approx_dist, used_bucket = lsh_approx_nn(hand_points, hand_tables, hand_query)
    assert approx_dist >= 1
    assert 4 * approx_dist <= 9 * 1
    assert squared_distance(hand_points[approx_idx], hand_query) == approx_dist

    assert counters["bucket"] >= 1
    assert counters["fallback"] >= 1


if __name__ == "__main__":
    main()
    print("PASS")
