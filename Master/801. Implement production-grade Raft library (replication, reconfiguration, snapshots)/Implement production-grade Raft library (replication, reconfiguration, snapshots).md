# Raft Consensus Algorithm Library (Production-Grade)

This project implements a **production-grade Raft library** in JavaScript/Node.js, covering **leader election, log replication, cluster reconfiguration, and snapshots**.


## Task Description

**Problem:**  
Implement a distributed consensus library based on the **Raft algorithm** that supports:

1. **Leader Election** – nodes elect a leader to coordinate log replication.
2. **Log Replication** – leader replicates client commands to follower nodes.
3. **Cluster Reconfiguration** – dynamically add or remove nodes without compromising safety.
4. **Snapshots** – compress log history to reduce memory/disk usage for long-running systems.

**Goals:**

- Safety: no two leaders at the same time, logs consistent.
- Liveness: clients eventually see committed entries.
- Fault-tolerance: tolerate failures up to ⌊(N-1)/2⌋ nodes.


##  Core Components

### 1. Node

- Maintains **state**: `Follower`, `Candidate`, `Leader`.
- Tracks **current term**, **votedFor**, **log entries**.
- Handles RPCs: `RequestVote` and `AppendEntries`.

### 2. Leader Election

- Timeout-based elections.
- Candidate requests votes from other nodes.
- Leader established if **majority votes** received.

### 3. Log Replication

- Leader appends client commands to its log.
- Sends `AppendEntries` RPCs to followers.
- Followers append and acknowledge logs.
- Leader commits entries after **majority replication**.

### 4. Reconfiguration

- Supports **joint consensus** to safely add/remove nodes.
- Ensures logs remain consistent during topology changes.

### 5. Snapshots

- Periodically snapshot state to reduce log size.
- Follower can install snapshot if missing log entries.


##  Architecture Overview

```javascript
Raft Node
├─ State: Follower / Candidate / Leader
├─ Log: [term, command]
├─ Current Term
├─ Voted For
├─ RPCs:
│ ├─ RequestVote
│ └─ AppendEntries
└─ Snapshot Management
```

## Mermaid Diagram

```mermaid
flowchart TD
    A[Raft Cluster] --> B[Node 1]
    A --> C[Node 2]
    A --> D[Node 3]

    B --> B1[State: Follower/Candidate/Leader]
    B --> B2[Current Term]
    B --> B3[Voted For]
    B --> B4[Log Entries]
    B --> B5[Snapshot]

    C --> C1[State: Follower/Candidate/Leader]
    C --> C2[Current Term]
    C --> C3[Voted For]
    C --> C4[Log Entries]
    C --> C5[Snapshot]

    D --> D1[State: Follower/Candidate/Leader]
    D --> D2[Current Term]
    D --> D3[Voted For]
    D --> D4[Log Entries]
    D --> D5[Snapshot]

    %% Leader Election
    E[Leader Election] --> F[Start Election Timer]
    F --> G[Timeout? Become Candidate]
    G --> H[Request Votes from Peers]
    H --> I{Majority Votes?}
    I -->|Yes| J[Become Leader]
    I -->|No| F

    %% Log Replication
    J --> K[AppendEntries RPC]
    K --> B4
    K --> C4
    K --> D4

    %% Snapshot Management
    L[Snapshot] --> B5
    L --> C5
    L --> D5

    %% Cluster Reconfiguration
    M[Cluster Reconfiguration] --> N[Add/Remove Node]
    N --> A

```