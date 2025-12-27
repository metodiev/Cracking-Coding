# Production-Grade Raft Library

##  Overview

This is a **Python implementation of a production-grade Raft consensus algorithm** library. Raft is a distributed consensus protocol used to manage replicated logs across nodes reliably, even in the presence of failures.

The library supports:

* Leader election
* Log replication
* Dynamic cluster reconfiguration
* Snapshots and state persistence

This project is suitable for **learning, testing, and prototyping distributed systems**.



##  What is Raft?

Raft ensures that multiple nodes maintain a **consistent replicated log**. Core goals:

* Elect a single leader at any time
* Replicate logs to followers
* Ensure safety: committed entries cannot be lost or overwritten
* Support dynamic membership changes
* Support snapshotting to compact the log

### Core Concepts

* **Leader**: Handles client requests, replicates log entries
* **Follower**: Receives log entries, votes for leaders
* **Candidate**: A follower that starts an election
* **LogEntry**: Records command and term
* **Snapshot**: Compact state storage to truncate logs


##  Features

1. **Leader Election**

   * Timeout-based elections
   * Majority vote ensures a single leader

2. **Log Replication**

   * Leader appends entries
   * Followers replicate and acknowledge
   * Commit happens after majority confirmation

3. **Cluster Reconfiguration**

   * Safe addition/removal of nodes using joint consensus

4. **Snapshots**

   * Persist application state
   * Truncate old logs
   * Enable fast recovery

5. **RPC Communication**

   * `AppendEntries` and `RequestVote`
   * In-memory for tests, extendable to network transport



##  File Structure

```
raft/
│
├─ __init__.py
├─ node.py        # RaftNode class and state machine
├─ log.py         # LogEntry class and log management
├─ network.py     # RPC transport layer, message handling
├─ config.py      # Cluster configuration
├─ snapshot.py    # Snapshot persistence and recovery
└─ raft.py        # High-level Raft controller / API
```



##  Example Usage

```python
from raft.raft import RaftNode

# Initialize a Raft node
node = RaftNode(node_id=1, cluster=[1,2,3])

# Start the node
node.start()

# Append a command
node.append_command("SET key value")

# Create snapshot
node.create_snapshot()
```

---

## ⏱ Complexity

| Feature         | Complexity                 |
| --------------- | -------------------------- |
| Leader election | O(n) messages              |
| Log replication | O(n) per log entry         |
| Snapshots       | O(log N) disk writes       |
| Reconfiguration | O(n) per membership change |



## Testing Strategy

1. **Unit Tests**

   * Leader election correctness
   * Log replication and commit
   * Snapshot creation and recovery
   * Cluster reconfiguration transitions

2. **Integration Tests**

   * Simulate network partitions
   * Node crashes and recovery
   * Majority failure handling



##  Extensions

* Add persistent storage backend (LevelDB, RocksDB)
* Implement network transport via gRPC / TCP
* Metrics & monitoring for production
* Port to Go or C++ for high-performance systems



##  Summary

This library provides a **robust Python implementation of Raft** for learning, prototyping, and testing distributed systems. While Python is not ideal for very high-throughput production workloads, the **algorithm and structure** remain fully representative of real-world Raft deployments.

It can be extended with snapshots, persistence, and real network communication for production-grade experiments.
