"""Task 972 - Implement personalized learning paths using performance clustering.  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)

Learners are described by a mastery vector over six skills plus two pacing features (solve speed,
consistency).  The pipeline scales the features to [0, 1], runs deterministic k-means on them (seeded
k-means++ seeding, Lloyd iterations, ties broken towards the lowest cluster index) and then derives one
personalized path per cluster: repeatedly the *ready* skill (every prerequisite already on the path) with
the lowest cluster mastery, so each cohort starts from what it is weakest at.

Simplified versus a production recommender: k-means on eight features, no dimensionality reduction, no
collaborative filtering, one path per cluster rather than per learner.  Verified: feature scaling
including the all-zero-column boundary, the assignment against a brute-force nearest-centroid oracle,
non-increasing Lloyd inertia with a re-derived final value, cluster-size conservation and purity, the
duplicate-point and k>n boundaries, and each path against a brute-force smallest-topological-order oracle.
"""
from collections import Counter
from itertools import permutations

SKILLS = ("arrays", "hashing", "sorting", "binary-search", "graphs", "dp")
PREREQS = (("arrays", "hashing"), ("arrays", "sorting"), ("sorting", "binary-search"),
           ("hashing", "graphs"), ("binary-search", "dp"), ("graphs", "dp"))
PROFILES = (("foundation", 20, 12), ("core", 55, 12), ("advanced", 85, 10))
CLUSTERS, LEARNER_COUNT, SEED = 3, 24, 99
MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Rng:
    """splitmix64 - fixed seed, so seeding and therefore the clusters are reproducible."""

    def __init__(self, seed):
        self.state = seed & MASK64

    def below(self, bound):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
        return (value ^ (value >> 31)) % bound


class Learner:
    __slots__ = ("name", "label", "features")

    def __init__(self, name, label, features):
        self.name, self.label, self.features = name, label, features


def clamp(value, low=0, high=100):
    return low if value < low else (high if value > high else value)


def make_learners(count=LEARNER_COUNT, seed=SEED):
    """Three latent profiles with noise, so the clusters have real structure to find."""
    rng = Rng(seed)
    learners = []
    for index in range(count):
        label, base, spread = PROFILES[index % len(PROFILES)]
        mastery = tuple(clamp(base + rng.below(2 * spread) - spread) for _ in SKILLS)
        speed = clamp(10 + (100 - base) // 2 + rng.below(11) - 5)
        consistency = clamp(base + rng.below(21) - 10)
        learners.append(Learner(f"L{index:02d}", label, mastery + (speed, consistency)))
    return learners


def scale(rows):
    """Per-column max scaling into [0, 1]; an all-zero column collapses to 0 instead of dividing by 0."""
    dimension = len(rows[0]) if rows else 0
    maxima = [max(row[column] for row in rows) for column in range(dimension)]
    return [tuple(value / maxima[column] if maxima[column] else 0.0 for column, value in enumerate(row))
            for row in rows]


def squared_distance(point, other):
    return sum((left - right) ** 2 for left, right in zip(point, other))


def assign(points, centroids):
    """Nearest centroid per point, ties going to the lowest cluster index."""
    return [min(range(len(centroids)), key=lambda c: (squared_distance(point, centroids[c]), c))
            for point in points]


def inertia(points, centroids, assignment):
    return sum(squared_distance(point, centroids[cluster]) for point, cluster in zip(points, assignment))


def update(points, centroids, assignment):
    moved = []
    for cluster, previous in enumerate(centroids):
        members = [point for point, owner in zip(points, assignment) if owner == cluster]
        if members:
            moved.append(tuple(sum(point[d] for point in members) / len(members) for d in range(len(previous))))
        else:
            moved.append(previous)  # an empty cluster keeps its centre; the emptiness is reported
    return moved


def kmeans_plus_plus(points, k, rng):
    """Deterministic seeding: start at point 0, then draw proportional to the squared distance."""
    seeds = [0]
    while len(seeds) < k:
        weights = [min(squared_distance(points[index], points[seed]) for seed in seeds)
                   for index in range(len(points))]
        total = sum(weights)
        if total <= 0:  # all remaining points coincide: take the next unused one
            seeds.append(next(index for index in range(len(points)) if index not in seeds))
            continue
        pick = rng.below(total)
        for index, weight in enumerate(weights):
            if pick < weight:
                seeds.append(index)
                break
            pick -= weight
    return [points[index] for index in seeds]


def lloyd(points, centroids, iterations=50):
    """Lloyd iterations; the recorded inertia can only go down (assignment and update both improve it)."""
    history = [inertia(points, centroids, assign(points, centroids))]
    current = list(centroids)
    while len(history) <= iterations:
        moved = update(points, current, assign(points, current))
        value = inertia(points, moved, assign(points, moved))
        history.append(value)
        if moved == current:
            break
        current = moved
    return current, assign(points, current), history


def kmeans(points, k, seed=SEED):
    if k < 1 or k > len(points):
        raise ValueError(f"k={k} cannot partition {len(points)} learners")
    rng = Rng(seed)
    centroids, assignment, history = lloyd(points, kmeans_plus_plus(points, k, rng))
    empties = [cluster for cluster in range(k) if cluster not in assignment]
    return {"centroids": centroids, "assignment": assignment, "history": history, "empties": empties}


def greedy_path(mastery, prereqs=PREREQS):
    """Weakest-ready-first topological order - the personalized sequence for one cohort."""
    done, path = set(), []
    while len(path) < len(mastery):
        ready = [skill for skill in SKILLS
                 if skill not in done and all(pre in done for pre, _post in prereqs if _post == skill)]
        chosen = min(ready, key=lambda skill: (mastery[skill], skill))
        path.append(chosen)
        done.add(chosen)
    return tuple(path)


def lexicographic_oracle(mastery, prereqs=PREREQS):
    """Independent oracle: brute-force every valid topological order, keep the smallest mastery tuple."""
    valid = [order for order in permutations(SKILLS)
             if all(order.index(pre) < order.index(post) for pre, post in prereqs)]
    return min(tuple(mastery[skill] for skill in order) for order in valid)


def purity(learners, assignment, k):
    """Share of learners whose cluster is dominated by their latent profile."""
    total = 0
    for cluster in range(k):
        labels = [learner.label for learner, owner in zip(learners, assignment) if owner == cluster]
        total += max(Counter(labels).values()) if labels else 0
    return total / len(learners) if learners else 0.0


def cluster_paths(result):
    return [greedy_path(dict(zip(SKILLS, centroid[:len(SKILLS)]))) for centroid in result["centroids"]]


def main():
    learners = make_learners()
    points = scale([learner.features for learner in learners])
    result = kmeans(points, CLUSTERS, SEED)
    # 1) scaling: every feature lands in [0, 1], each column peaks at exactly 1, an all-zero column is 0.
    flat = scale([(5, 3), (5, 9), (5, 0)])
    zero = scale([(0, 2), (0, 4)])
    check(all(0.0 <= value <= 1.0 for point in points for value in point)
          and all(max(point[column] for point in points) == 1.0 for column in range(len(points[0])))
          and [point[0] for point in flat] == [1.0, 1.0, 1.0]
          and [point[1] for point in flat] == [1 / 3, 1.0, 0.0]
          and [point[0] for point in zero] == [0.0, 0.0] and scale([]) == [],
          f"feature scaling is wrong: {flat}")
    # 2) assignment oracle: every point's own centroid is the brute-force nearest one.
    brute = [min(range(CLUSTERS), key=lambda c: (squared_distance(point, result["centroids"][c]), c))
             for point in points]
    check(result["assignment"] == brute
          and all(len(point) == len(next(iter(result["centroids"]))) for point in points),
          f"assignment disagrees with the nearest-centroid oracle: {result['assignment']}")
    # 3) inertia: the Lloyd history never grows and its last value is the value we can re-derive.
    history = result["history"]
    check(all(history[i + 1] <= history[i] + 1e-9 for i in range(len(history) - 1)) and history[0] > history[-1] > 0.0
          and abs(inertia(points, result["centroids"], result["assignment"]) - history[-1]) < 1e-9,
          f"the inertia history must be non-increasing: {[round(value, 4) for value in history]}")
    # 4) cluster quality: sizes conserve learners, no cluster is empty, the latent profiles are recovered.
    sizes = Counter(result["assignment"])
    check(sum(sizes.values()) == len(learners) and sorted(sizes) == list(range(CLUSTERS))
          and result["empties"] == [] and purity(learners, result["assignment"], CLUSTERS) >= 0.75,
          f"cluster quality is off: {dict(sizes)} purity={purity(learners, result['assignment'], CLUSTERS)}")
    # 5) boundary: coincident points leave an empty cluster, k = n gives singletons, k > n is rejected.
    duplicates = [(0.0, 0.0)] * 3 + [(1.0, 1.0)] * 4
    duplicate_run = kmeans(duplicates, 3, SEED)
    singleton_run = kmeans([(0.0,), (1.0,), (3.0,), (7.0,)], 4, SEED)
    try:
        kmeans(points, len(points) + 1, SEED)
        raise AssertionError("k above the learner count must be rejected")
    except ValueError:
        pass
    check(len(duplicate_run["empties"]) == 1 and duplicate_run["assignment"] == [0, 0, 0, 1, 1, 1, 1]
          and singleton_run["empties"] == [] and abs(singleton_run["history"][-1]) < 1e-12
          and kmeans([(0.0,), (2.0,)], 1, SEED)["assignment"] == [0, 0],
          f"the empty-cluster and singleton boundaries are wrong: {duplicate_run['empties']}")
    # 6) paths: a permutation per cluster, prerequisites respected, and equal to the brute-force optimum.
    paths = cluster_paths(result)
    for centroid, path in zip(result["centroids"], paths):
        mastery = dict(zip(SKILLS, centroid[:len(SKILLS)]))
        check(sorted(path) == sorted(SKILLS)
              and all(path.index(pre) < path.index(post) for pre, post in PREREQS)
              and tuple(mastery[skill] for skill in path) == lexicographic_oracle(mastery)
              and path == greedy_path(mastery),
              f"the path {path} is not the weakest-ready-first order for {mastery}")
    # 7) personalization: two cohorts weak in different prerequisites really do get different paths.
    weak_hashing = dict(zip(SKILLS, (10, 5, 40, 50, 60, 70)))
    weak_sorting = dict(weak_hashing, hashing=40, sorting=5)
    first, second = greedy_path(weak_hashing), greedy_path(weak_sorting)
    check(first[:2] == ("arrays", "hashing") and second[:2] == ("arrays", "sorting") and first != second
          and first[-1] == "dp" == second[-1],
          f"per-cluster paths are not personalized: {first} vs {second}")
    # 8) determinism: the whole pipeline - learners, seeding, clusters, inertia and paths - reproduces.
    repeat = kmeans(scale([learner.features for learner in make_learners()]), CLUSTERS, SEED)
    check([learner.features for learner in make_learners()] == [learner.features for learner in learners]
          and repeat["centroids"] == result["centroids"] and repeat["assignment"] == result["assignment"]
          and repeat["history"] == history and cluster_paths(repeat) == paths
          and len({centroid for centroid in result["centroids"]}) == CLUSTERS
          and purity(make_learners(), repeat["assignment"], CLUSTERS)
          == purity(learners, result["assignment"], CLUSTERS),
          "the pipeline must be reproducible")
    print(f"learners={len(learners)} features={len(points[0])} clusters={CLUSTERS} sizes={dict(sorted(sizes.items()))} "
          f"purity={purity(learners, result['assignment'], CLUSTERS):.2f} inertia={history[-1]:.3f}")
    print(f"paths={[' > '.join(path) for path in paths]}")
    print("PASS")


if __name__ == "__main__":
    main()
