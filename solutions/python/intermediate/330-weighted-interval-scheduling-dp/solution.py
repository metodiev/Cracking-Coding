"""Task 330 - Weighted interval scheduling (DP). (Tier: Intermediate, CTCI Ch 8)

Maximise the total weight of a set of pairwise compatible jobs.  Jobs are compatible
when one finishes at or before the next one starts (finish <= start, half-open
[job.start, job.finish) convention).  Sort jobs by finish, binary-search the last
compatible earlier job and run the DP max(best[i-1], weight[i] + best[p(i)]).
Cross-checked with brute force over all subsets for small inputs.

Note on the expected value: for jobs (1,3,5) (2,5,6) (4,6,5) (6,7,4) the best set is
(1,3)+(4,6)+(6,7) with total 5 + 5 + 4 = 14 - the task description's "12" is wrong
(5 + 5 is 10), so the assertion below uses 14, which brute force confirms.
"""


def weighted_interval_scheduling(jobs):
    """Returns (best total weight, chosen jobs). jobs are (start, finish, weight)."""
    if not jobs:
        return 0, []
    ordered = sorted(jobs, key=lambda job: (job[1], job[0]))
    finishes = [job[1] for job in ordered]

    def last_compatible(i):
        """Largest index j < i with finishes[j] <= ordered[i][0]; -1 when none exists."""
        lo, hi, best = 0, i - 1, -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if finishes[mid] <= ordered[i][0]:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    best = [0] * (len(ordered) + 1)
    choose = [False] * (len(ordered) + 1)
    for i in range(1, len(ordered) + 1):
        without = best[i - 1]
        with_job = ordered[i - 1][2] + best[last_compatible(i - 1) + 1]
        if with_job > without:
            best[i] = with_job
            choose[i] = True
        else:
            best[i] = without

    chosen = []
    i = len(ordered)
    while i > 0:
        if choose[i]:
            chosen.append(list(ordered[i - 1]))
            i = last_compatible(i - 1) + 1
        else:
            i -= 1
    chosen.reverse()
    return best[len(ordered)], chosen


def brute_force(jobs):
    """Exhaustive subset enumeration; the independent oracle for small inputs."""
    n = len(jobs)
    best = 0
    for mask in range(1 << n):
        chosen = [jobs[i] for i in range(n) if mask >> i & 1]
        total = sum(job[2] for job in chosen)
        if total <= best:
            continue
        chosen = sorted(chosen)
        if all(chosen[k][1] <= chosen[k + 1][0] for k in range(len(chosen) - 1)):
            best = total
    return best


def lcg_jobs(count, seed):
    jobs = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        start = seed % 30
        seed = (seed * 75 + 74) % 65537
        finish = start + 1 + seed % 8
        seed = (seed * 75 + 74) % 65537
        weight = 1 + seed % 20
        jobs.append([start, finish, weight])
    return jobs, seed


def main():
    # the job set from the task description: (1,3)+(4,6)+(6,7) = 5+5+4 = 14, not 12
    brief_jobs = [[1, 3, 5], [2, 5, 6], [4, 6, 5], [6, 7, 4]]
    total, chosen = weighted_interval_scheduling(brief_jobs)
    assert total == 14
    assert chosen == [[1, 3, 5], [4, 6, 5], [6, 7, 4]]
    assert brute_force(brief_jobs) == 14

    # hand-solved example: (1,3,5) + (3,5,4) + (5,7,3) = 12; the heavy decoys lose
    hand = [[1, 3, 5], [3, 5, 4], [5, 7, 3], [1, 7, 10], [2, 3, 2]]
    total, chosen = weighted_interval_scheduling(hand)
    assert total == 12
    assert chosen == [[1, 3, 5], [3, 5, 4], [5, 7, 3]]
    assert brute_force(hand) == 12

    assert weighted_interval_scheduling([]) == (0, [])
    total, chosen = weighted_interval_scheduling([[4, 9, 7]])
    assert total == 7 and chosen == [[4, 9, 7]]

    seed = 57721
    for trial in range(60):
        count = 4 + seed % 6
        jobs, seed = lcg_jobs(count, seed)
        total, chosen = weighted_interval_scheduling(jobs)
        assert total == brute_force(jobs), jobs
        assert sum(job[2] for job in chosen) == total
        assert all(chosen[k][1] <= chosen[k + 1][0] for k in range(len(chosen) - 1))


if __name__ == "__main__":
    main()
    print("PASS")
