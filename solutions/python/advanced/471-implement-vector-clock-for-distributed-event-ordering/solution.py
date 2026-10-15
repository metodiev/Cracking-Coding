"""Task 471 - Implement vector clock for distributed event ordering. (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)

Vector clocks over a scripted 3-node trace: every node keeps one counter per node; a send
piggybacks the sender's post-tick clock; a receive merges (pointwise max) and then ticks the
local component. Two events are ordered iff one vector is pointwise <= the other (strictly
smaller somewhere); otherwise they are concurrent. Every pairwise classification is checked
against a causal-order oracle: transitive closure (Floyd-Warshall) of program order plus
send -> matching-receive edges.
"""

NODES = 3


def lcg(seed):
    return (seed * 75 + 74) % 65537


def classify(a, b):
    le = all(x <= y for x, y in zip(a, b))
    ge = all(x >= y for x, y in zip(a, b))
    if le and ge:
        return "equal"
    if le:
        return "before"
    if ge:
        return "after"
    return "concurrent"


def featured_events():
    c = [[0] * NODES for _ in range(NODES)]
    events = []

    def emit(node, kind, msg=-1):
        events.append((node, kind, msg, c[node][:]))

    c[0][0] += 1
    emit(0, "L")                     # 0: A has 1 local event
    c[0][0] += 1
    emit(0, "L")                     # 1: A has 2
    c[0][0] += 1
    emit(0, "S", 0)                  # 2: A sends {A:3} to B
    c[1][1] += 1
    emit(1, "L")                     # 3: B local tick, still unaware of A's message
    for i in range(NODES):
        c[1][i] = max(c[1][i], (3, 0, 0)[i])
    c[1][1] += 1
    emit(1, "R", 0)                  # 4: B merges A's clock -> {A:3, B:2}
    c[1][1] += 1
    emit(1, "S", 1)                  # 5: B sends {A:3, B:3} to C
    for i in range(NODES):
        c[2][i] = max(c[2][i], (3, 3, 0)[i])
    c[2][2] += 1
    emit(2, "R", 1)                  # 6: C merges -> {A:3, B:3, C:1}
    return events


def gen_script(count, seed):
    events = []
    pending = [[] for _ in range(NODES)]
    clocks = [[0] * NODES for _ in range(NODES)]
    next_msg = 0
    for _ in range(count):
        seed = lcg(seed)
        node = seed % NODES
        seed = lcg(seed)
        kind = seed % 3
        if kind == 2 and not pending[node]:
            kind = 0
        if kind == 1:
            seed = lcg(seed)
            target = (node + 1 + seed % (NODES - 1)) % NODES
            clocks[node][node] += 1
            pending[target].append((next_msg, clocks[node][:]))
            events.append((node, "S", next_msg, clocks[node][:]))
            next_msg += 1
        elif kind == 2:
            msg, sent = pending[node].pop(0)
            for i in range(NODES):
                clocks[node][i] = max(clocks[node][i], sent[i])
            clocks[node][node] += 1
            events.append((node, "R", msg, clocks[node][:]))
        else:
            clocks[node][node] += 1
            events.append((node, "L", -1, clocks[node][:]))
    return events


def causal_reachability(events):
    n = len(events)
    reach = [[False] * n for _ in range(n)]
    last = [-1] * NODES
    sent_at = [-1] * n
    for i, (node, kind, msg, _clock) in enumerate(events):
        if last[node] != -1:
            reach[last[node]][i] = True
        last[node] = i
        if kind == "S":
            sent_at[msg] = i
        elif kind == "R":
            reach[sent_at[msg]][i] = True
    for k in range(n):
        for i in range(n):
            if reach[i][k]:
                for j in range(n):
                    if reach[k][j]:
                        reach[i][j] = True
    return reach


def main():
    fe = featured_events()
    cl = [e[3] for e in fe]
    assert cl[0] == [1, 0, 0]
    assert cl[1] == [2, 0, 0]
    assert cl[2] == [3, 0, 0]
    assert cl[3] == [0, 1, 0]
    assert cl[4] == [3, 2, 0]
    assert cl[5] == [3, 3, 0]
    assert cl[6] == [3, 3, 1]
    assert classify(cl[0], cl[1]) == "before"
    assert classify(cl[1], cl[2]) == "before"
    assert classify(cl[0], cl[2]) == "before"
    assert classify(cl[3], cl[0]) == "concurrent"
    assert classify(cl[3], cl[1]) == "concurrent"
    assert classify(cl[3], cl[2]) == "concurrent"
    assert classify(cl[3], cl[4]) == "before"
    assert classify(cl[2], cl[4]) == "before"
    assert classify(cl[4], cl[5]) == "before"
    assert classify(cl[5], cl[6]) == "before"
    assert classify(cl[1], cl[6]) == "before"
    assert classify(cl[6], cl[4]) == "after"
    assert classify(cl[6], cl[0]) == "after"
    for i in range(len(fe)):
        for j in range(i + 1, len(fe)):
            assert classify(cl[i], cl[j]) != "equal"

    events = gen_script(26, 1234)
    kinds = {"L": 0, "S": 0, "R": 0}
    for e in events:
        kinds[e[1]] += 1
    assert kinds == {"L": 20, "S": 4, "R": 2}
    reach = causal_reachability(events)
    order = concurrent = 0
    for i in range(len(events)):
        for j in range(i + 1, len(events)):
            got = classify(events[i][3], events[j][3])
            if reach[i][j]:
                expected = "before"
            elif reach[j][i]:
                expected = "after"
            else:
                expected = "concurrent"
            assert got == expected, (i, j, got, expected)
            if expected == "concurrent":
                concurrent += 1
            else:
                order += 1
    assert order == 140
    assert concurrent == 185


if __name__ == "__main__":
    main()
    print("PASS")
