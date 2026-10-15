# Chapter 15 · Threads and Locks

> Book reference: **Ch 15 — Threads and Locks** (threads in Java, synchronization, deadlocks & prevention).

Concurrency questions check whether you understand **interleaving, memory visibility, and deadlock** —
the three classic failure modes.

```mermaid
mindmap
  root((Threads & Locks))
    Coordination
      mutex / monitor
      condition variables
      semaphores / barriers
    Problems
      race conditions
      deadlock / livelock
      priority inversion
    Scale patterns
      thread pools
      producer-consumer
      readers-writers
    Lock-free
      atomics / CAS
      memory ordering
```

---

## 1. Threads vs processes

```mermaid
flowchart LR
    subgraph Process["Process (isolated memory)"]
      T1["thread 1"] --- M["shared heap"]
      T2["thread 2"] --- M
      T1 --- S1["stack 1"]
      T2 --- S2["stack 2"]
    end
```

| | Processes | Threads |
|--|-----------|---------|
| memory | isolated | shared heap, private stacks |
| cost | heavy (fork) | light (~MB stack) |
| failure | contained | one crash kills all |
| communication | IPC/sockets | shared memory (+locks!) |

---

## 2. Race conditions — the interleaving timeline

`counter++` is **three** operations: load, add, store of:

```mermaid
sequenceDiagram
    participant T1 as Thread 1
    participant M as counter = 5
    participant T2 as Thread 2
    T1->>M: load 5
    T2->>M: load 5
    T1->>M: store 6
    T2->>M: store 6
    Note over M: two increments → 6, not 7. LOST UPDATE
```

```mermaid
flowchart LR
    L["lock(m)"] --> R["read-modify-write"] --> U["unlock(m)"]
```

```java
// Java — synchronized makes the read-modify-write atomic & visible
class Counter {
    private int value = 0;
    synchronized void increment() { value++; }
    synchronized int get() { return value; }
}
// or: AtomicInteger value = new AtomicInteger(); value.incrementAndGet();
```

```csharp
// C#
class Counter {
    private readonly object gate = new();
    private int value;
    public void Increment() { lock (gate) { value++; } }
}
// or: Interlocked.Increment(ref value);
```

```python
# Python — threads share the GIL, but `value += 1` is still not atomic across bytecodes
import threading

class Counter:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()

    def increment(self):
        with self._lock:
            self._value += 1
```

```javascript
// JavaScript — single-threaded event loop: no data races in user code…
let counter = 0;
counter++; // safe: one thread of execution per isolate
// …but async interleaving still causes logical races across await points.
```

---

## 3. Deadlock — four conditions & the wait-for cycle

```mermaid
flowchart LR
    T1["Thread 1<br/>holds L1, wants L2"] -->|wait| T2["Thread 2<br/>holds L2, wants L1"]
    T2 -->|wait| T1
```

All four must hold: mutual exclusion, hold-and-wait, no preemption, circular wait.

```mermaid
flowchart TD
    D["Deadlock risk"] --> P1["1. Global lock ordering<br/>(always lock A before B)"]
    D --> P2["2. Timeout / tryLock + backoff"]
    D --> P3["3. One big lock (coarse — simpler, slower)"]
    D --> P4["4. Lock-free structures / message passing"]
```

The dining-philosophers fix is exactly rule 1: order the forks, one philosopher picks right-first.

```mermaid
flowchart LR
    P0["P0: fork0 → fork1"] --> OK["all follow the same order → no cycle"]
    P1["P1: fork1 → fork2"] --> OK
    P4["P4: fork4 → fork0"] --> OK
```

Mapped: [358 dining philosophers](../../tasks/intermediate/358-dining-philosophers-solution-using-ordered-locks.md),
[357 deadlock detection](../../tasks/intermediate/357-deadlock-detection-for-resource-allocation-graph-algorithm.md),
[670 banker's algorithm](../../tasks/expert/670-implement-resource-allocation-graph-and-banker-s-algorithm-simul.md).

---

## 4. Producer–consumer with a bounded buffer

```mermaid
sequenceDiagram
    participant P as Producer
    participant B as Buffer (cap N)
    participant C as Consumer
    P->>B: wait while full → produce → signal notEmpty
    C->>B: wait while empty → consume → signal notFull
    Note over P,B: condition variables make waiting efficient (no busy-spin)
```

```java
// Java — the canonical three-lock (or use BlockingQueue and skip the drama)
class BoundedBuffer<T> {
    private final Queue<T> q = new ArrayDeque<>();
    private final int cap;
    BoundedBuffer(int cap) { this.cap = cap; }

    synchronized void put(T item) throws InterruptedException {
        while (q.size() == cap) wait();      // always while, never if
        q.add(item);
        notifyAll();
    }
    synchronized T take() throws InterruptedException {
        while (q.isEmpty()) wait();
        T item = q.remove();
        notifyAll();
        return item;
    }
}
```

Python equivalent: `queue.Queue` (thread-safe), or `threading.Condition`. C#: `BlockingCollection<T>`.
JavaScript: the event loop + promises play the role of the buffer.

Mapped: [195 producer-consumer](../../tasks/beginner/195-implement-basic-producer-consumer-using-threads-mutex-condition.md),
[348 multi-producer](../../tasks/intermediate/348-producer-consumer-using-semaphore-locks-with-multiple-producers.md).

---

## 5. The rest of the toolbox

```mermaid
classDiagram
    class Semaphore {
        -int permits
        +acquire()
        +release()
    }
    class Barrier {
        -int parties
        +await()
    }
    class ThreadPool {
        -BlockingQueue~Task~ queue
        -Worker[] workers
        +submit(task)
    }
    class ReadWriteLock {
        +readLock()
        +writeLock()
    }
```

| Primitive | Use for |
|-----------|---------|
| mutex / monitor | one-at-a-time critical section |
| semaphore (counting) | limit concurrent access (pool of N) |
| condition variable | wait-for-predicate without spinning |
| barrier | phase synchronization (all threads meet) |
| read-write lock | many readers OR one writer |
| thread pool | reuse threads, bound concurrency |
| CAS / atomics | lock-free updates, counters |
| volatile | visibility (not atomicity!) |

### Lock-free — CAS in one picture

```mermaid
flowchart TD
    A["read expected"] --> B{"CAS(addr, expected, new)"}
    B -->|"still expected"| C["swapped ✓"]
    B -->|"changed by someone"| D["retry loop"]
    D --> A
```

ABA problem, memory reordering (use acquire/release semantics), and safe reclamation
(epoch / hazard pointers) are the advanced traps — mapped at tasks
[354](../../tasks/intermediate/354-implement-wait-free-counter-sketch-atomic-operations.md),
[355](../../tasks/intermediate/355-implement-epoch-based-reclamation-idea-for-concurrent-data-struc.md),
[356](../../tasks/intermediate/356-implement-lock-free-stack-outline-cas.md),
[621 Michael-Scott queue](../../tasks/expert/621-implement-scalable-concurrent-queue-michael-scott-queue.md).

### Visibility — the JMM/CLR memory model

```mermaid
sequenceDiagram
    participant T1 as Writer thread
    participant M as main memory
    participant T2 as Reader thread
    T1->>M: value = 42 (in CPU cache!)
    Note over M: reader may never see it without a barrier
    T1->>M: synchronized/volatile/lock → flush + barrier
    T2->>M: read → sees 42
```

Without synchronization (or `volatile`/`Interlocked`/`threading.Lock`), writes may never become
visible to other threads. „Works on my machine" concurrency bugs are usually visibility bugs.

---

## 6. Language-specific gotchas

| Language | Reality |
|----------|---------|
| Java | `synchronized`, `wait/notify`, `java.util.concurrent`, JMM with happens-before |
| C# | `lock`, `Monitor`, `Interlocked`, `Task`/`async`, no `volatile` for compound ops |
| Python | GIL serializes bytecode but not multi-step invariants; use `threading`/`multiprocessing`/asyncio |
| JavaScript | single-threaded loop; concurrency = interleaved async tasks; workers for CPU |

## Mapped tasks to practice

- [195 producer-consumer](../../tasks/beginner/195-implement-basic-producer-consumer-using-threads-mutex-condition.md) · [347 read-write lock](../../tasks/intermediate/347-implement-read-write-lock-basic-no-starvation.md) · [349 barrier](../../tasks/intermediate/349-barrier-synchronization-primitive-simple-implementation.md) · [350 singleton](../../tasks/intermediate/350-implement-thread-safe-singleton-double-checked-locking.md)
- [351 thread pool](../../tasks/intermediate/351-implement-thread-pool-basic-with-worker-threads-and-task-queue.md) · [352 fork-join sum](../../tasks/intermediate/352-implement-fork-join-style-parallel-sum-threads.md) · [353 sharded hashmap](../../tasks/intermediate/353-implement-concurrent-hashmap-simplified-sharding.md) · [354 atomic counter](../../tasks/intermediate/354-implement-wait-free-counter-sketch-atomic-operations.md)
- [356 lock-free stack](../../tasks/intermediate/356-implement-lock-free-stack-outline-cas.md) · [358 dining philosophers](../../tasks/intermediate/358-dining-philosophers-solution-using-ordered-locks.md) · [359 readers-writers fairness](../../tasks/intermediate/359-readers-writers-fairness-implementation-and-tests.md) · [492 lock-free ring buffer](../../tasks/advanced/492-implement-circular-buffer-with-lock-free-design-atomic.md)
- [615 transactional memory](../../tasks/expert/615-implement-transactional-memory-simulator-basics.md) · [618 linearizability checker](../../tasks/expert/618-implement-linearizability-checker-for-concurrent-histories.md) · [623 hazard pointers](../../tasks/expert/623-implement-hazard-pointers-memory-reclamation-in-concurrent-struc.md) · [626 work-stealing deque](../../tasks/expert/626-implement-work-stealing-deque-chase-lev-for-task-scheduler.md)

> Full list: [ctci-mapping.md → Ch 15](../ctci-mapping.md).
