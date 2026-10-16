# Chapter 9 · System Design and Scalability

> Book reference: **Ch 9 — System Design and Scalability** (“Design: Step-by-Step”, “Algorithms that Scale”).

System design interviews are evaluated as a conversation. Drive it with a structure, then dive deep.

```mermaid
flowchart TD
    A["1. Scope & requirements<br/>functional + non-functional"] --> B["2. Back-of-envelope math<br/>QPS, storage, bandwidth"]
    B --> C["3. High-level boxes<br/>client → LB → service → data"]
    C --> D["4. Deep dive on 1–2 components"]
    D --> E["5. Bottlenecks & trade-offs<br/>CAP, hot keys, failure modes"]
    E --> F["6. Wrap-up: monitor, evolve"]
```

---

## 1. Scale out, not up

```mermaid
flowchart LR
    subgraph Vertical["Vertical scaling (bigger box)"]
      V1["1 server<br/>16 CPU"] --> V2["1 server<br/>64 CPU"]
    end
    subgraph Horizontal["Horizontal scaling (more boxes)"]
      LB["Load balancer"] --> A1["app 1"]
      LB --> A2["app 2"]
      LB --> A3["app 3"]
    end
```

Horizontal wins for availability and cost — but forces you to solve **statelessness, coordination,
and data partitioning**.

### Load balancing

```mermaid
flowchart TD
    C["Clients"] --> LB{"Load balancer"}
    LB -->|"round robin / least-conn / hash"| S1["server 1"]
    LB --> S2["server 2"]
    LB --> S3["server 3"]
    S1 -.->|"health checks"| LB
```

| Algorithm | Good for | Watch out |
|-----------|----------|-----------|
| round robin | uniform requests | ignores server load |
| least connections | long-lived connections | needs connection tracking |
| consistent hash (by key) | cache locality | hot keys; add virtual nodes |

---

## 2. Caching — every layer

```mermaid
flowchart LR
    U["User"] --> CDN["CDN / edge cache<br/>(static assets)"]
    CDN --> APP["App servers<br/>(in-process cache)"]
    APP --> R["Redis / Memcached<br/>(shared cache)"]
    R --> DB[("Database<br/>query cache, buffer pool")]
```

Cache checklist for interviews:

1. **What to cache** (read-heavy, expensive, tolerable staleness)
2. **Eviction** — LRU / LFU / TTL (tasks [160](../../tasks/beginner/160-implement-lru-cache-using-linked-list-hashmap.md), [485](../../tasks/advanced/485-implement-cache-eviction-policies-lru-lfu-arc-simulate.md))
3. **Invalidation** — TTL vs write-through vs explicit bust (hardest problem in CS!)
4. **Stampedes** — request coalescing, jittered TTLs

---

## 3. Databases at scale

### Replication (leader–follower)

```mermaid
sequenceDiagram
    participant W as Leader
    participant F1 as Follower 1
    participant F2 as Follower 2
    W->>W: write (commit)
    W->>F1: replicate log
    W->>F2: replicate log
    Note over F1,F2: reads scale out, async replicas may lag
```

### Sharding (partitioning)

```mermaid
flowchart TD
    K["user_id = 42"] --> H{"hash(42) mod 4"}
    H --> S0["shard 0"]
    H --> S1["shard 1 ← 42 lands here"]
    H --> S2["shard 2"]
    H --> S3["shard 3"]
```

| Strategy | Pro | Con |
|----------|-----|-----|
| range (id 0–1M → shard 0) | range scans | hot ranges (new ids) |
| hash mod N | even spread | reshuffles everything when N changes |
| consistent hashing ring | only ~1/N keys move on change | needs virtual nodes |

```mermaid
flowchart LR
    subgraph Ring["Consistent hash ring"]
      direction LR
      N1(["node A"]) --> K1["key 7f"] --> N2(["node B"]) --> K2["key 9c"] --> N3(["node C"]) --> N1
    end
```

### CAP & consistency

```mermaid
flowchart TD
    C{"Network partition?"} -->|"happens"| CHOICE{"Pick per operation"}
    CHOICE -->|"Consistency: fail/wait"| CP["CP — like a bank ledger"]
    CHOICE -->|"Availability: serve possibly stale"| AP["AP — like a social feed"]
```

Quorums: with replicas `n`, write quorum `W`, read quorum `R` — if `W + R > n` reads see the latest write.

---

## 4. Async processing & MapReduce

```mermaid
flowchart LR
    API["API"] -->|"enqueue job"| Q[("Queue")]
    Q --> W1["worker 1"]
    Q --> W2["worker 2"]
    W1 --> DB[("store")]
    W2 --> DB
```

Decouples spikes from capacity; enables retries/backoff (task [508](../../tasks/advanced/508-implement-background-job-scheduler-with-retry-backoff-semantics.md)).

```mermaid
flowchart LR
    subgraph Map["Map (parallel)"]
      I1["split 1"] --> M1["map → (k,v)"]
      I2["split 2"] --> M2["map → (k,v)"]
    end
    M1 --> SH["Shuffle<br/>group by key"]
    M2 --> SH
    SH --> R1["reduce key k1"]
    SH --> R2["reduce key k2"]
```

Word count: map emits `(word, 1)`; shuffle groups; reduce sums. Task [469](../../tasks/advanced/469-implement-distributed-aggregations-map-reduce-toy.md) builds the toy version.

---

## 5. A 10-minute worked design — URL shortener

```mermaid
flowchart LR
    U["client"] --> LB["LB"] --> API["shortener service"]
    API -->|"write"| ID["ID generator<br/>(counter or hash)"]
    API --> KV[("key → URL store")]
    U -->|"read /abc123"| CDN["CDN cache"] --> API
    API --> KV
```

| Step | Answer sketch |
|------|---------------|
| Requirements | ~100M writes/day, 10× reads, low latency, links never expire |
| ID | base62 counter chunks or `hash(longURL)` + collision retry (task [344](../../tasks/intermediate/344-design-and-implement-a-url-shortener-basic-hash-collision-handli.md)) |
| Storage | KV store, partition by short key; replicate for read scaling |
| Cache | CDN for hot redirects; LRU in-memory |
| Bottleneck | the ID generator → use per-node ranges (Snowflake-style) |

Other mapped system tasks to practice: [470 consistent hashing](../../tasks/advanced/470-implement-consistent-hashing-ring-virtual-nodes.md),
[471 vector clocks](../../tasks/advanced/471-implement-vector-clock-for-distributed-event-ordering.md),
[472 Raft](../../tasks/advanced/472-implement-basic-raft-consensus-pseudo-simulation-leader-election.md),
[473 Paxos](../../tasks/advanced/473-implement-paxos-basic-single-decree-outline.md),
[616 Chandy-Lamport](../../tasks/expert/616-implement-distributed-snapshot-algorithm-chandy-lamport.md),
[801 production Raft](../../tasks/master/801-implement-production-grade-raft-library-replication-reconfigurat.md),
[803 mini SQL engine](../../tasks/master/803-build-a-small-sql-engine-with-parser-planner-and-executor-suppor.md).

> Full list: [ctci-mapping.md → Ch 9](../ctci-mapping.md).
