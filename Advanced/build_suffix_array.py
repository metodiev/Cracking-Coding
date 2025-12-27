def build_suffinx_array(s):
    n = len(s)
    sa = list(range(n)) # suffix indicies
    rank = [ord(c) for c in s] # intitial ranks
    tmp = [0] * n 

    k = 1
    while k < n:
        sa.sort(key=lambda i: (rank[i], rank[i+k] if i + k < n else -1))

        tmp[sa[0]] = 0
        for i in range(1, n):
            prev, curr = sa[i-1], sa[i]
            tmp[curr] = tmp[prev] + (
                (rank[prev], rank[prev + k] if prev + k < n else -1) <
                (rank[curr], rank[curr + k] if curr + k < n else -1)
            )

        rank, tmp = tmp , rank
        k <<= 1
    return sa


s = "banana"
print(build_suffinx_array(s))

        

