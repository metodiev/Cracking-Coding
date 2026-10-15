#!/usr/bin/env python3
"""Map the 998 README tasks onto the Cracking the Coding Interview (6th ed.) structure.

Reads README.md, classifies every task into a CTCI chapter (or a clearly-named
"Beyond the Book" topic), then writes:
  - docs/ctci-mapping.json   {task number: chapter label}
  - tutorials/ctci-mapping.md   human-readable mapping grouped by chapter

Classification is rule-based (ordered keyword rules) with explicit overrides for
tasks where keywords are ambiguous. Re-runnable and idempotent.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Chapter metadata (labels follow the CTCI 6th edition TOC, §X = Advanced Topics)
# ---------------------------------------------------------------------------
CH = {
    "ch01": "Ch 1 — Arrays and Strings",
    "ch02": "Ch 2 — Linked Lists",
    "ch03": "Ch 3 — Stacks and Queues",
    "ch04": "Ch 4 — Trees and Graphs",
    "ch05": "Ch 5 — Bit Manipulation",
    "ch06": "Ch 6 — Math and Logic Puzzles",
    "ch07": "Ch 7 — Object-Oriented Design",
    "ch08": "Ch 8 — Recursion and Dynamic Programming",
    "ch09": "Ch 9 — System Design and Scalability",
    "ch10": "Ch 10 — Sorting and Searching",
    "ch11": "Ch 11 — Testing",
    "ch12": "Ch 12 — C and C++",
    "ch13": "Ch 13 — Java",
    "ch14": "Ch 14 — Databases",
    "ch15": "Ch 15 — Threads and Locks",
    "ch16": "Ch 16 — Moderate (additional review problems)",
    "ch17": "Ch 17 — Hard (additional review problems)",
    "advx": "CTCI §X — Advanced Topics",
    "bads": "Beyond — Advanced Data Structures & Algorithms",
    "bml": "Beyond — Machine Learning & AI",
    "bgfx": "Beyond — Graphics, Geometry & Robotics",
    "bnum": "Beyond — Numerical & Scientific Computing",
    "bsec": "Beyond — Security & Privacy",
    "bnet": "Beyond — Networking, OS & Real-Time",
    "bcpl": "Beyond — Compilers, Languages & Solvers",
    "beng": "Beyond — Software Engineering & Tooling",
}

CHAPTER_ORDER = ["ch12", "ch16", "ch17"]  # shown only in the doc footer/notes

# Explicit overrides for tasks with ambiguous keywords.
OVERRIDES: dict[int, str] = {}


def _ov(*numbers: int, ch: str) -> None:
    for n in numbers:
        OVERRIDES[n] = ch


# Beginner ---------------------------------------------------------------
_ov(9, 10, ch="ch08")            # factorial (recursion intro)
_ov(11, 12, 13, ch="ch08")       # Fibonacci
_ov(14, ch="ch05")               # swap without temp (XOR trick)
_ov(29, ch="ch05")               # power of two (bit trick)
_ov(20, ch="ch10")               # merge two sorted arrays
_ov(30, 31, ch="ch06")           # reverse digits / palindrome number (no string)
_ov(41, 42, ch="ch03")           # balanced brackets
_ov(68, ch="ch10")               # heapsort
_ov(86, 87, ch="ch10")           # inversion counting
_ov(93, ch="ch08")               # regex-like matching (backtracking)
_ov(74, ch="ch10")               # stable partition (sorting)
_ov(155, ch="ch02")              # Josephus (circular linked list)
_ov(157, ch="ch07")              # finite state machine for number validation
_ov(163, ch="ch05")              # XOR encryption
_ov(168, ch="ch05")              # byte checksum
_ov(164, 179, ch="ch04")         # running median via heaps
_ov(165, ch="ch06")              # reservoir sampling (probability)
_ov(174, ch="ch06")              # naive matrix multiplication
_ov(175, ch="ch06")              # multiply large integers as strings
_ov(177, ch="ch07")              # sparse matrix storage
_ov(178, ch="ch04")              # Floyd-Warshall transitive closure
_ov(184, 185, 186, 187, ch="bads")  # KMP, Rabin-Karp, rolling hash
_ov(190, 191, 192, 193, ch="ch03")  # RPN / shunting-yard / tokenizer
_ov(194, ch="ch07")              # TTL cache (design)
_ov(195, ch="ch15")              # producer-consumer
_ov(196, ch="ch07")              # lazy file line reader (design)
_ov(197, ch="bnet")              # rsync-style file diff
_ov(198, 199, ch="ch07")         # JSON serialize/deserialize (data representation)

_ov(201, ch="ch01")              # two-sum family keyed by keyword below
_ov(202, 203, ch="ch01")         # 3-sum / 4-sum
_ov(220, ch="ch08")              # bitmask DP for TSP
_ov(222, 223, ch="ch08")         # stock DP family
_ov(225, ch="ch08")              # stock DP family
_ov(227, 228, 229, ch="ch08")    # bitonic / increasing / alternating subsequences
_ov(318, ch="ch01")              # minimum window subsequence
_ov(322, 328, ch="bgfx")         # skyline problem (D&C / heap+sweep)
_ov(331, ch="ch08")              # weighted job scheduling (DP)
_ov(339, ch="bads")              # Mo's algorithm (with unicode apostrophe)
_ov(342, ch="ch03")              # expression evaluator (stacks)
# Intermediate -----------------------------------------------------------
_ov(212, 213, 214, ch="ch08")    # Kadane / max product / subarray product (DP-ish)
_ov(221, ch="ch08")              # circular Kadane variant
_ov(230, 231, ch="ch01")         # longest palindrome substring / Manacher
_ov(235, ch="ch04")              # serialize N-ary tree
_ov(247, ch="advx")              # AVL tree (CTCI §X AVL Trees)
_ov(248, ch="advx")              # red-black tree (CTCI §X Red-Black Trees)
_ov(254, 255, 467, ch="bads")    # bloom filters
_ov(268, 269, ch="ch04")         # word ladder (graph BFS)
_ov(276, 277, 278, 279, ch="ch04")  # Hungarian / max flow / min-cut
_ov(284, ch="ch08")              # word search board (backtracking)
_ov(301, 302, ch="bnum")         # FFT / NTT
_ov(310, 311, 312, ch="bads")    # suffix array / automaton
_ov(313, 314, 315, 316, 317, ch="bads")  # Z-function, Aho-Corasick, hashing
_ov(332, ch="ch07")              # calendar booking
_ov(333, ch="ch07")              # hit counter
_ov(334, ch="ch07")              # rolling median with delete
_ov(336, ch="ch07")              # snapshot array
_ov(337, ch="ch07")              # majority vote generalized (design)
_ov(340, ch="bads")              # eertree
_ov(343, ch="ch04")              # graph isomorphism checks
_ov(344, ch="ch07")              # URL shortener
_ov(345, ch="ch07")              # randomized set
_ov(346, ch="ch07")              # optimistic concurrency sketch (design)
_ov(347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 358, 359, ch="ch15")
_ov(357, ch="ch15")              # deadlock detection
_ov(360, ch="ch14")              # versioned key-value store
_ov(362, ch="ch04")              # 2D segment tree
_ov(363, 364, ch="ch04")         # kd-tree / quad-tree
_ov(365, ch="bgfx")              # skyline divide & conquer (geometry)
_ov(366, 367, 368, 369, 370, ch="bgfx")  # convex hull, PIP, closest pair, sweep
_ov(371, ch="bgfx")              # segment intersection
_ov(372, 373, ch="bgfx")         # shoelace, rotating calipers
_ov(374, ch="ch05")              # bitset operations
_ov(375, 376, ch="bnum")         # Gaussian elimination / rational solve
_ov(377, ch="bnum")              # simplex LP
_ov(378, 379, 380, 381, ch="ch06")  # partitions, nCr, modular inverse, CRT
_ov(382, 383, 384, ch="ch06")    # Miller-Rabin / Pollard / BSGS
_ov(385, ch="bsec")              # RSA demo
_ov(386, ch="bsec")              # elliptic curve
_ov(387, ch="bads")              # minimal perfect hash
_ov(388, ch="ch01")              # run-length encoding
_ov(389, 390, 391, ch="bads")    # Huffman / Shannon-Fano / adaptive Huffman
_ov(392, 393, ch="bads")         # LZW / deflate ideas
_ov(394, ch="bgfx")              # skyline merging with multiset
_ov(395, 396, ch="bads")         # Li Chao / CHT
_ov(397, 398, ch="ch08")         # Knuth optimization / D&C DP optimization
_ov(399, 400, ch="ch04")         # centroid decomposition / HLD

# Advanced / Expert / Master --------------------------------------------
_ov(401, 402, 403, 404, 405, 406, 407, 408, 409, 410, ch="bads")  # suffix structures, BWT, FM
_ov(411, ch="bads")
_ov(412, ch="bads")              # LZ77 parse
_ov(413, ch="bads")              # rope
_ov(414, 415, ch="ch08")         # Hirschberg / affine gaps
_ov(416, ch="ch08")              # bitset DP
_ov(417, ch="ch08")              # monotone queue DP
_ov(418, ch="bads")              # dynamic convex hull
_ov(419, 420, ch="ch04")         # link-cut / Euler tour tree
_ov(421, 422, ch="ch14")         # persistent segment tree / persistent UF (versioned structures)
_ov(423, ch="ch08")              # CDQ divide & conquer
_ov(424, 425, ch="ch04")         # small-to-large / centroid decomposition
_ov(426, 427, ch="ch04")         # tree DP / rerooting
_ov(428, ch="ch04")              # binary lifting LCA
_ov(429, ch="bads")              # range kth via persistent segtree
_ov(430, ch="ch04")              # Mo's on trees
_ov(431, ch="bads")              # suffix automaton
_ov(432, 433, ch="bads")         # order-statistics tree
_ov(434, 435, 436, ch="bads")    # scapegoat / finger tree / rope
_ov(437, ch="ch14")              # B-tree node ops
_ov(438, ch="ch10")              # external-memory (multiway merge) sort
_ov(439, 440, ch="bads")         # compressed Fenwick / vEB
_ov(441, 442, ch="ch04")         # lazy segtree / segment tree beats
_ov(443, 444, ch="ch04")         # 2D Fenwick / 2D segtree
_ov(445, ch="bads")              # fractional cascading
_ov(446, 447, ch="ch04")         # offline dynamic connectivity / dynamic MST
_ov(448, ch="bads")              # parametric search
_ov(449, 450, ch="ch04")         # Karger / Stoer-Wagner min cut
_ov(451, ch="ch04")              # Gomory-Hu
_ov(452, ch="ch04")              # min mean cycle
_ov(453, 454, 455, ch="ch04")    # flow variations
_ov(456, 457, ch="ch04")         # blossom matching
_ov(458, ch="ch04")              # tree knapsack DP
_ov(459, ch="ch04")              # virtual tree
_ov(460, 461, 462, 463, ch="bads")  # LSH / kd-tree NN / VP-tree / simhash
_ov(464, ch="ch06")              # weighted reservoir sampling
_ov(465, 466, ch="bads")         # count-min / hyperloglog
_ov(468, ch="bads")              # streaming quantiles
_ov(469, ch="advx")              # map-reduce toy (CTCI §X MapReduce)
_ov(470, 471, ch="ch09")         # consistent hashing / vector clocks
_ov(472, 473, ch="ch09")         # Raft / Paxos outlines
_ov(474, 475, ch="bsec")         # Merkle tree / authenticated DS
_ov(476, ch="ch14")              # snapshot isolation simulator
_ov(477, 478, 479, 480, 481, 482, ch="ch14")  # B+tree / LSM / compaction / WAL / checkpoints
_ov(483, 484, ch="ch14")         # MVCC snapshot / KV store
_ov(485, 486, 487, 488, ch="ch14")  # cache eviction / ARC / prefetch / layout
_ov(489, ch="bads")              # suffix-prefix automaton
_ov(490, 491, ch="ch10")         # exponential search / bitonic search
_ov(492, ch="ch15")              # lock-free circular buffer
_ov(493, ch="bads")              # bloom + cmsketch composition
_ov(494, 495, ch="bml")          # bandits
_ov(496, 497, ch="bml")          # gradient descent / L-BFGS
_ov(498, 499, 500, 501, 502, ch="bml")  # kmeans / DBSCAN / spectral / PCA / SVD
_ov(503, 504, ch="bads")         # simhash LSH / product quantization
_ov(505, ch="ch09")              # bloom-backed cache for web responses
_ov(506, 507, 508, 509, 510, 511, ch="ch09")  # throttle/scheduler/map-reduce worker
_ov(512, 513, 514, 515, 516, 517, ch="bml")  # autodiff / NN / SGD / Adam
_ov(518, ch="bml")               # kNN with kd-tree
_ov(519, 520, ch="bml")          # metrics
_ov(521, 522, 523, 524, 525, ch="bml")  # sampling / MCMC / HMM
_ov(526, ch="bads")              # suffix automaton endpos
_ov(527, ch="bads")              # Booth rotation
_ov(528, ch="bads")              # Karp-Rabin automaton
_ov(529, 530, ch="bads")         # Lyndon
_ov(531, ch="ch10")              # weighted median selection
_ov(532, ch="ch09")              # ski rental (online algorithms) — systems-ish; keep ch09? better bnum? 
_ov(533, ch="ch10")              # median of medians
_ov(534, ch="ch04")              # Edmonds practical
_ov(535, ch="ch08")              # DP on DAG
_ov(536, ch="ch08")              # constrained shortest path DP
_ov(537, 538, ch="bgfx")         # TSP heuristics / Christofides
_ov(539, 540, ch="bnum")         # FWHT / Möbius (transform math)
_ov(541, ch="ch08")              # D&C monotone DP
_ov(542, ch="ch04")              # tree isomorphism hashing
_ov(543, 544, ch="ch04")         # graph sketches / triangle counting
_ov(545, 546, ch="ch04")         # pagerank
_ov(547, 548, 549, ch="bml")     # spectral partitioning / Louvain / label prop
_ov(550, ch="bads")              # MinHash
_ov(551, 552, 553, 554, ch="ch04")  # subgraph iso / motifs
_ov(555, 556, ch="bads")         # matrix exponentiation path counting
_ov(557, ch="bads")              # tropical semiring
_ov(558, ch="ch05")              # boolean matmul bitsets
_ov(559, ch="bads")              # linear recurrences matrix power
_ov(560, ch="bads")              # Berlekamp-Massey
_ov(561, ch="bads")              # sliding-window distinct HLL
_ov(562, 563, 564, 565, ch="bcpl")  # MIP / SAT / 2-SAT / CDCL
_ov(566, ch="ch04")              # Steiner tree
_ov(567, 568, 569, ch="bml")     # facility location / submodular / max coverage (optimization)
_ov(570, ch="bnum")              # simplex demo
_ov(571, ch="bsec")              # collision-resistant hash tests
_ov(572, ch="bnum")              # median-of-means
_ov(573, 574, ch="bgfx")         # geometric hashing / RANSAC
_ov(575, 576, ch="bgfx")         # epipolar / bundle adjustment
_ov(577, ch="bads")              # kd-tree bulk
_ov(578, 579, ch="bads")         # wavelet tree
_ov(580, ch="bads")              # FM-index
_ov(581, 582, ch="bads")         # CSA / RLE segtree
_ov(583, ch="bads")              # persistent binary trie
_ov(584, ch="ch08")              # bitset DP tricks
_ov(585, 586, ch="bnum")         # NTT small prime / chirp z
_ov(587, 588, ch="bnum")         # determinant / inverse
_ov(589, ch="bml")               # spectral clustering sparse
_ov(590, 591, 592, 593, ch="bml")  # NMF / K-SVD / OMP / lasso
_ov(594, 595, 596, ch="bnum")    # random projection / approximate matmul / CUR
_ov(597, ch="bml")               # streaming PCA
_ov(598, ch="bml")               # online SVM
_ov(599, ch="bads")              # LPH demo
_ov(600, ch="beng")              # checkpoint-restore for computations

# Expert ----------------------------------------------------------------
_ov(601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, ch="bads")
_ov(612, 613, ch="bads")         # persistent structures
_ov(614, ch="ch14")              # persistent B-tree for DB index
_ov(615, ch="ch15")              # transactional memory simulator
_ov(616, 617, ch="ch09")         # Chandy-Lamport / snapshot across shards
_ov(618, ch="ch15")              # linearizability checker
_ov(619, ch="ch15")              # happens-before model
_ov(620, ch="ch15")              # lock elision
_ov(621, 622, 623, 624, 625, 626, ch="ch15")  # concurrent structures
_ov(627, 628, 629, 630, 631, 632, 633, 634, 635, 636, ch="ch09")  # distributed systems
_ov(637, 638, 639, 640, ch="ch14")  # isolation levels / 2PL / OCC
_ov(641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, ch="ch14")
_ov(657, 658, ch="ch09")         # shuffle sim / straggler mitigation
_ov(659, 660, 661, 662, 663, ch="ch09")  # consistent hashing / quorum / chain repl / paxos / log compaction
_ov(664, 665, 666, ch="ch09")    # CRDT editor / OT / CRDT JSON
_ov(667, 668, 669, 670, ch="bnet")  # real-time scheduling / priority inheritance / EDF / banker's
_ov(671, 672, 673, 674, 675, 676, ch="bcpl")  # program analysis / SSA / reg alloc / branch prediction
_ov(677, 678, 679, ch="bnet")    # MESI / TSO / TM conflict manager
_ov(680, ch="bcpl")              # auto-parallelization dependence analysis
_ov(681, ch="bcpl")              # GPU kernel mapping
_ov(682, ch="ch09")              # map-reduce skewed keys
_ov(683, 684, 685, ch="bnet")    # IP routing / Patricia trie / TCAM
_ov(686, 687, 688, ch="bnet")    # BGP / OSPF / RIP
_ov(689, 690, 691, ch="bnet")    # congestion control
_ov(692, ch="ch09")              # reliable multicast
_ov(693, 694, 695, ch="ch09")    # erasure coding / anti-entropy
_ov(696, ch="bnum")              # Kahan
_ov(697, 698, ch="bnum")         # CG / multigrid
_ov(699, ch="bnet")              # parallel prefix-sum simulation
_ov(700, ch="ch09")              # work partitioning clusters
_ov(701, ch="bads")              # Ukkonen approximate matching
_ov(702, ch="bads")              # HNSW sketch
_ov(703, ch="ch04")              # dynamic connectivity link-cut
_ov(704, ch="ch04")              # offline dynamic connectivity
_ov(705, ch="ch04")              # Steiner forest
_ov(706, 707, 708, ch="ch04")    # flows
_ov(709, 710, 711, 712, 713, ch="bsec")  # differential privacy / HE / PSI / secure agg
_ov(714, ch="bml")               # federated averaging
_ov(715, 716, 717, 718, ch="bml")  # pruning / quantization / mixed precision / compression
_ov(719, 720, ch="ch09")         # parameter server / ring allreduce
_ov(721, 722, ch="bml")          # checkpoint compression / gradient checkpointing
_ov(723, 724, 725, ch="bml")     # batching / memory planner / fusion
_ov(726, 727, ch="bnum")         # sparse-dense matmul / Strassen
_ov(728, ch="bnum")              # cache-oblivious matmul
_ov(729, ch="bnum")              # pivot stability
_ov(730, ch="bnum")              # randomized low rank
_ov(731, 732, ch="bgfx")         # mesh refinement / PIC
_ov(733, 734, ch="bgfx")         # Barnes-Hut / FMM
_ov(735, 736, 737, ch="bgfx")    # dynamic hull / BVH / OBB
_ov(738, 739, 740, 741, 742, 743, 744, 745, ch="bgfx")  # motion planning / triangulation / Voronoi / predicates
_ov(746, ch="ch04")              # topological sort cycle debugging
_ov(747, 748, 749, ch="bcpl")    # SAT techniques / SMT / symbolic execution
_ov(750, 751, ch="ch11")         # fuzzing
_ov(752, 753, 754, ch="bsec")    # taint / CFI / shadow memory
_ov(755, 756, ch="bsec")         # timing attack / constant time
_ov(757, 758, ch="bsec")         # ORAM / MPC
_ov(759, ch="ch09")              # PBFT
_ov(760, ch="ch09")              # snapshot-based auditing
_ov(761, ch="bsec")              # TEE simulation
_ov(762, ch="bsec")              # searchable encryption
_ov(763, 764, ch="bml")          # inverted index / TF-IDF+BM25 (IR)
_ov(765, 766, ch="bads")         # IVF+PQ ANN / HNSW integration
_ov(767, ch="bml")               # learning-to-rank
_ov(768, ch="ch08")              # skyline queries Pareto (multi-criteria DP? make ch08? -> better ch10? keep ch08)
_ov(769, ch="ch08")              # knapsack PTAS
_ov(770, ch="bads")              # heavy hitters streaming
_ov(771, ch="bnum")              # coreset convex hull
_ov(772, ch="ch04")              # dynamic planar graphs
_ov(773, ch="bads")              # self-adjusting skip lists
_ov(774, 775, 776, ch="ch14")    # concurrent B-tree / speculative inserts / rebalancing
_ov(777, ch="ch14")              # LSM tuning
_ov(778, ch="ch14")              # cache warming
_ov(779, ch="bads")              # billion-scale ANN
_ov(780, ch="ch09")              # distributed reservoir sampling
_ov(781, ch="ch09")              # distributed join planning
_ov(782, 783, 784, 785, ch="bml")  # PGMs / VI / LDA Gibbs
_ov(786, 787, 788, ch="bnum")    # ODE integrators
_ov(789, 790, ch="bgfx")         # ICP / bundle adjustment
_ov(791, 792, ch="bgfx")         # EKF-SLAM / particle filter
_ov(793, ch="bgfx")              # RANSAC line/circle
_ov(794, ch="bgfx")              # voxelization / octree
_ov(795, ch="bads")              # LSH tuning
_ov(796, ch="bsec")              # privacy-preserving federated evaluation
_ov(797, 798, ch="bml")          # contextual bandits / TS hierarchical
_ov(799, ch="bnum")              # Lagrange numeric solver
_ov(800, ch="bnum")              # simulated annealing

# Master ----------------------------------------------------------------
_ov(801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, ch="ch09")
_ov(813, ch="ch09")              # encrypted collaborative editor
_ov(814, ch="bsec")              # searchable encrypted index
_ov(815, ch="bads")              # billion-scale vector search
_ov(816, 818, 819, ch="bml")     # distributed training / model parallel / checkpointing
_ov(817, ch="ch09")              # heterogeneous cluster scheduler
_ov(820, ch="bml")               # sparse embedding sharding
_ov(821, 822, 823, ch="bml")     # bidding / fraud / anomaly
_ov(824, 825, 826, ch="bml")     # differentiable renderer / physics / AMP
_ov(827, 828, ch="bml")          # BOHB / Bayesian optimization
_ov(829, ch="bsec")              # federated + DP
_ov(830, ch="bsec")              # verifiable ML inference
_ov(831, 832, 833, ch="ch09")    # blockchain client / sharding / federated consensus
_ov(834, 835, 836, ch="bsec")    # zk-SNARK / recursive SNARK / VDF
_ov(837, ch="bsec")              # DID
_ov(838, 839, 840, ch="bsec")    # MPC aggregation / OT PSI / FHE
_ov(841, 842, ch="ch09")         # enclave chain-of-trust / light node gossip
_ov(843, 844, ch="ch09")         # Pregel / streaming graph
_ov(845, ch="ch04")              # personalized pagerank bookmark coloring
_ov(846, ch="ch09")              # subgraph-centric distributed processing
_ov(847, 848, 849, ch="ch09")    # distributed sketches / sampling / joins
_ov(850, ch="beng")              # feature store platform
_ov(851, 852, ch="ch09")         # consistency benchmarks / global snapshot
_ov(853, ch="ch14")              # access control and auditing
_ov(854, 855, 856, 857, ch="ch14")  # DB testing / plan regression / replanning / ML cost model
_ov(858, ch="bml")               # transfer learning drift
_ov(859, ch="ch14")              # storage layout optimizer
_ov(860, 861, 862, ch="bml")     # PPL / MCMC / ABC
_ov(863, 864, ch="bml")          # graph embedding / distributed HNSW
_ov(865, ch="bml")               # model serving
_ov(866, 867, 868, 869, ch="bml")  # SHAP / counterfactual / fairness / Granger
_ov(870, 871, 872, 873, 874, ch="beng")  # CI reproducibility / registry / differential testing
_ov(875, 876, ch="bnum")         # FFT tuned / parallel n-body
_ov(877, 878, ch="bgfx")         # geometry library / physics engine
_ov(879, 880, 881, 882, ch="bgfx")  # raytracer / rendering / GI / denoise
_ov(883, 884, 885, ch="bcpl")    # autodiff GPU kernels / autotuner / recompilation
_ov(886, ch="bsec")              # encrypted search corpus
_ov(887, 888, ch="ch14")         # compression / IVM
_ov(889, 890, ch="bsec")         # MPC PSI / PIR
_ov(891, 892, 893, 894, ch="beng")  # benchmarks / chaos / trace / RCA
_ov(895, 896, ch="ch14")         # learned index / RL optimization
_ov(897, 898, ch="bml")          # compressed NN / NAS
_ov(899, ch="bsec")              # encrypted collaborative ML
_ov(900, ch="beng")              # research platform
_ov(901, ch="beng")              # provable bounds + tests
_ov(902, ch="bsec")              # self-verifying logs (Merkle)
_ov(903, ch="ch09")              # adaptive replication
_ov(904, ch="beng")              # SLO tiered storage migration
_ov(905, 906, ch="ch09")         # SLA scheduler / multi-fidelity orchestration
_ov(907, ch="bml")               # differentiable programming fitting
_ov(908, ch="bml")               # certified ML predictions
_ov(909, 910, ch="bml")          # continual learning / neural-symbolic
_ov(911, ch="bads")              # heavy hitters mergeable sketch
_ov(912, ch="bsec")              # privacy-preserving analytics
_ov(913, ch="beng")              # reproducibility seeds/audit
_ov(914, ch="bnet")              # zero-copy RPC
_ov(915, ch="bsec")              # remote attestation orchestration
_ov(916, ch="ch09")              # streaming joins
_ov(917, ch="ch09")              # self-healing storage cluster
_ov(918, 919, 920, 921, 922, ch="bgfx")  # QP solver / trajectory / SLAM / grasp / multi-robot
_ov(923, ch="ch09")              # privacy ledger
_ov(924, ch="ch09")              # WAN replicated state machine
_ov(925, 926, 927, 928, 929, ch="ch14")  # HTAP / caching / materialized views / feature store / automl ops
_ov(930, 931, ch="bml")          # adversarial / certified robustness
_ov(932, ch="bnum")              # differential testing CPU/GPU numerics
_ov(933, ch="bcpl")              # fused kernel runtime
_ov(934, ch="ch09")              # gradient compression under loss
_ov(935, ch="ch09")              # multi-tenant GPU scheduler
_ov(936, ch="beng")              # operator checkpointing
_ov(937, ch="beng")              # canary deployment pipeline
_ov(938, ch="beng")              # experiment platform SLO
_ov(939, ch="beng")              # trace replay
_ov(940, ch="ch11")              # load generator
_ov(941, 942, ch="beng")         # profiling / hot-path analyzer
_ov(943, ch="bsec")              # composable DP
_ov(944, ch="bnum")              # rational arithmetic
_ov(945, ch="bsec")              # certified RNG
_ov(946, ch="ch09")              # external-memory graph traversal
_ov(947, ch="ch09")              # graph partitioner
_ov(948, ch="ch04")              # sublinear property testing graphs
_ov(949, ch="ch04")              # approximate matching streaming
_ov(950, ch="bads")              # multi-index hashing rerank
_ov(951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, ch="beng")
# testing-flavoured platform tasks
_ov(952, ch="ch11")              # autograder runs tests
_ov(954, ch="beng")              # CI multi-JDK (kept beng)
_ov(957, ch="ch11")              # randomized test-case generator
_ov(958, ch="ch11")              # plagiarism detector
_ov(965, ch="bsec")              # secure sandbox for untrusted submissions
_ov(983, ch="bsec")              # security review checklist + scans

# ---------------------------------------------------------------------------
PATTERNS: list[tuple[str, str]] = [
    # (chapter key, regex, description order matters — first match wins)
]

RULES: list[tuple[str, str]] = [
    # Distributed systems / scalability (before ML/networking to catch Raft etc.)
    ("ch09", r"distributed|raft|paxos|consensus|replicat|shard|gossip|crdt|consistent hash|map-?reduce|"
             r"byzantine|quorum|erasure|federated|multi-master|cross-datacenter|autoscal|leader election|"
             r"cross-shard|2pc|two-phase commit|snapshot isolation|lineariz|vector clock|hybrid logical clock|"
             r"microservice|cluster|multi-tenant|ring-allreduce|parameter server|serving|canary|"
             r"anti-entropy|membership|rebalanc|global snapshot|streaming pipeline|stream processing|"
             r"pregel|external-memory graph|graph partitioner|privacy-preserving ledger|ledger|"
             r"data center|datacenter|replicated state machine|replication|shuffle|heavy-hitter detection with distributed|"
             r"distributed sampling|approximate joins|skewed keys|work partitioning|straggler|"
             r"streaming graph|windowed aggregation"),
    # Security & privacy
    ("bsec", r"encrypt|decrypt|\brsa\b|elliptic curve|zk-?snark|blockchain|ledger|differential privacy|"
             r"homomorphic|secure multi-?party|secret sharing|oblivious transfer|private set intersection|"
             r"secure aggregation|attestation|enclave|searchable encryption|timing attack|side-channel|"
             r"constant-time|taint|control-flow integrity|sanitizer|privacy|anonym|blind signature|"
             r"hash property tests|merkle|authenticated data structure|secure sandbox|security review|"
             r"spv proof|vdf|decentralized identity|private information retrieval|\bpir\b|fhe\b|"
             r"oblivious ram|oram|certified random number|robust median estimator|secure remote|trust chain"),
    # Databases
    ("ch14", r"database|\bsql\b|query planner|query optimizer|query execution|query rewriter|materialized view|"
             r"mvcc|write-ahead|\bwal\b|log-structured|lsm|sstable|b\+ ?tree|\bb-tree\b|key-value store|"
             r"cache eviction|lru, lfu|join algorithm|join ordering|join planning|oltp|olap|htap|"
             r"column-store|column store|bitmap index|index selection|database cracking|predicate pushdown|"
             r"vectorized execution|online aggregation|cardinality estimation|compaction|memtable|"
             r"storage layout|view maintenance|snapshot isolation|two-phase locking|optimistic concurrency|"
             r"access control and auditing|learned index|query-aware|adaptive indexing"),
    # Concurrency / threads (book Ch15)
    ("ch15", r"\bthread|lock\b|locked|locks|mutex|semaphor|atomic|livelock|deadlock|starvation|"
             r"producer-consumer|dining philosopher|lock-free|wait-free|work-stealing|hazard pointer|"
             r"\brcu\b|memory reclamation|synchroniz|race condition|spinlock|concurrent|"
             r"transactional memory|linearizability|happens-before|lock elision|read-write lock|"
             r"latch-free|double-checked|fork-join|barrier synchronization|wait-free counter|"
             r"epoch-based|non-blocking|optimistic concurrency control|checkpointed parallel"),
    # ML & AI
    ("bml", r"machine learning|neural|network training|network inference|gradient|sgd\b|adam\b|l-bfgs|"
             r"k-?means|dbscan|pca\b|svd\b|auto-?diff|backprop|classifier|logistic regression|"
             r"k-?nearest|knn\b|hidden markov|hmm\b|markov chain|metropolis|mcmc\b|variational|gibbs|"
             r"topic model|\blda\b|bayesian|bandit|thompson|upper confidence|\bucb\b|embedding|"
             r"clustering|spectral|community detection|pagerank|recommender|anomaly detection|"
             r"learning-to-rank|tf-?idf|bm25|inverted index|perplexity|coreset|k-?svd|sparse coding|"
             r"lasso|orthogonal matching|matrix factorization|shap|counterfactual|fairness|causal|"
             r"granger|confusion matrix|\broc\b|precision/recall|bootstrap|calibration|retention estimator|"
             r"predict|inference engine|model serving|model parallel|pruning|quantiz|mixed-precision|"
             r"checkpointing \(|feature attribution|sequence alignment|word embedding|negative sampling|"
             r"adversarial attack|certified robustness|concept drift|transfer learning|hyperparameter"),
    # Graphics / geometry / robotics
    ("bgfx", r"convex hull|geometry|point-in-polygon|polygon|voronoi|delaunay|triangulat|"
             r"ray ?trac|render|rasteriz|shader|mesh|\bslam\b|kalman|particle filter|motion planning|"
             r"\brrt\b|\bprm\b|robot|manipulation planner|trajectory|visibility graph|art gallery|"
             r"oriented bounding|bounding volume|barnes-hut|multipole|denois|\bbvh\b|kd-tree|quad-?tree|"
             r"octree|voxel|mesh refinement|collision detection|physics engine|point cloud|"
             r"iterative closest point|\bicp\b|bundle adjustment|epipolar|\bransac\b|geometric hashing|"
             r"skyline of buildings|skyline merging|closest pair|line sweep|rectangle area|"
             r"segment intersection|rotating calipers|shoelace|christofides|\btsp\b|noise addition|"
             r"poisson equation|extended kalman|gravity|n-body|qvto|qp solver|sturm|"
             r"differential renderer|global illumination|photorealistic|raytracer"),
    # Numerical & scientific
    ("bnum", r"numerical|linear algebra|matrix multiplication|strassen|eigen|lu decomposition|"
             r"gaussian elimination|floating|fp16|fp32|numerical|kahan|fft|convolut|\bntt\b|\bfwht\b|"
             r"number theoretic|runge|integrator|multigrid|conjugate gradient|determinant|matrix inverse|"
             r"simplex|linear program|polynomial|interval arithmetic|precision|rational arithmetic|"
             r"random projection|monte carlo|quantile|hyperloglog|reservoir|"
             r"probability|statistic|estimator|gaussian|pivot|integer overflow"),
    # Advanced data structures & algorithms (beyond the book)
    ("bads", r"suffix|\bbwt\b|burrows|fm-index|wavelet|rope|persistent segment|persistent union|"
             r"persistent binary trie|link-cut|euler tour tree|centroid|heavy-light|small-to-large|"
             r"segment tree|fenwick|binary indexed|sparse table|mo's|sqrt decomposition|"
             r"treap|splay|skip list|scapegoat|finger tree|van emde boas|\bvEB\b|order-statistics|"
             r"bloom filter|count-min|hyperloglog sketch|minhash|\blsh\b|locality sensitive|simhash|"
             r"product quantization|\bhnsw\b|\bivf\b|ann\b|nearest neighbor|vector index|vector search|"
             r"kmp|rabin-karp|aho-corasick|z-function|rolling hash|lyndon|booth|"
             r"huffman|shannon-fano|\blzw\b|lz77|compression encoder|eertree|palindrome tree|"
             r"matrix exponentiation|tropical semiring|berlekamp|convex hull trick|li chao|"
             r"parametric search|fractional cascading|heavy hitters|sketch(es)?\b|"
             r"order statistic|range kth|"
             r"galactic|approximate maximum matching|property testing"),
    # Compilers, languages, formal methods
    ("bcpl", r"compiler|parser generator|syntax tree|abstract syntax|\bssa\b|static analysis|liveness|"
             r"reaching definition|register allocation|instruction scheduling|branch prediction|"
             r"symbolic execution|grammar-based|interpreter|\bjit\b|bytecode|\bdsl\b|type inferen|"
             r"recompil|kernel|operator fusion|fused kernel|autotun|sat solver|dpll|2-sat|\bcdcl\b|"
             r"\bsmt\b|watched literals|clause learning|dependence analysis|auto-parallelization|"
             r"model checking|verif(y|ied|ication) |theorem|constraint solver|"
             r"mip\b|mixed integer|branch-and-bound"),
    # Networking, OS & real-time
    ("bnet", r"\btcp\b|reno|cubic|\bbbr\b|congestion|routing|bgp|ospf|\brip\b|ip rout|link-state|"
             r"distance-vector|multicast|socket|zero-copy|rpc\b|\bdns\b|http|grpc|"
             r"schedulability|rate-monotonic|earliest deadline|priority inheritance|banker|"
             r"resource allocation graph|cache coherence|\bmesi\b|memory model|litmus|\btso\b|"
             r"virtual memory|page table|file system|filesystem|os\b|operating system|"
             r"io_uring|epoll|interrupt|kernel scheduling|autoscaling|background job|cron|"
             r"real-time|scheduler|throttling|leaky-bucket|rate limiter middleware|"
             r"prefix-sum algorithm simulation|software engineering"),
    # Engineering tooling / platform (checked before testing)
    ("beng", r"readme|documentation|docs site|contributor guide|template|leaderboard|gamification|"
             r"mentoring|triage|repl|web ui|proctoring|accessibility|localization|curriculum|"
             r"dashboard|spaced-repetition|learning path|hint generator|explainer|paraphras|"
             r"difficulty calibration|retention|adaptive test|certification|feedback loop|"
             r"deployment|infra-as-code|monitoring|alerting|compliance|citation|licensing|"
             r"migration tools|cross-repo|offline mode|docker|benchmark suite|random-seed protocol|"
             r"ethics|moderation|quality review|ci pipeline|grading rubric|research platform|"
             r"checkpoint-restore|continuous integration|reproducible|provenance|"
             r"profiling|hot-path|canary analysis|experiment platform|trace-based|io\b|"
             r"stable sort wrapper|contributor|course|schedule|platform"),
    # Testing
    ("ch11", r"fuzz|autograder|grading|plagiarism|proctor|anticheat|test-case generator|"
             r"differential testing|regression detector|benchmark harness|load generator|"
             r"coverage-guided|fault injection|chaos|failure injection|incident|"
             r"test(ing|s)?\b|assert|verify.*correctness|autograde"),
    # Bit manipulation
    ("ch05", r"\bbits?\b|bit manipulation|set bits|bitmask|two's complement|bitwise|\bxor\b|"
             r"bitmap|bit index|boolean matrix|gray code|parity|checksum|base64|crc"),
    # Math and Logic Puzzles
    ("ch06", r"prime|sieve of|greatest common|\bgcd\b|\blcm\b|least common|"
             r"binary exponentiation|fast power|\bpower\(|integer power|pseudorandom|"
             r"linear congruential|random generator|multiply large integers|"
             r"probability|reservoir sampling|egg drop|magic (number|index)|"
             r"ncr|binomial|combinatoric|\bmodular\b|random number|"
             r"fizz|perfect number|abundant|goldbach|euclid|"
             r"hamming|roman numeral|excel column|happy number|ugly|collatz"),
    # Recursion & DP
    ("ch08", r"recursion|recursive|backtracking|subsets?\b|permutation|combination|"
             r"dynamic programming|\bdp\b|memo|knapsack|edit distance|\blcs\b|longest common|"
             r"longest increasing|climb(ing)? stairs|house robber|decode ways|word break|wildcard|"
             r"regular expression|regex|n-queens|sudoku|palindrome partition|partition.*(equal|subset)|"
             r"fibonacci|factorial|tower of hanoi|generate parentheses|restore ip|combination sum|"
             r"coin change|decode ways|root-to-leaf|path sum|catalan|jump game|minimax|"
             r"predict the winner|stone game|ways\b|ugly number|matrix.*obstacles|unique paths|"
             r"minimum path sum|interleaving|scrambl|game theory|divide & conquer dp|"
             r"skyline queries|pareto|prison cells|regex"),
    # Sorting & searching
    ("ch10", r"\bsort|binary search|quickselect|\bkth\b|k-th|top k|order statistic|"
             r"median|inversion|stable sort|insertion sort|selection sort|bubble|merge sort|"
             r"quicksort|counting sort|radix|bucket sort|search(ing)?\b|exponential search|"
             r"bitonic|search insert|first occurrence|last occurrence|element in stream|"
             r"weighted median|majority element"),
    # Stacks & queues
    ("ch03", r"stack|queue|deque|rpn|postfix|infix|shunting|bracket|parenthes|monotonic|"
             r"circular buffer|ring buffer"),
    # Linked lists
    ("ch02", r"linked list|linked-list|singly linked|doubly linked|list node|"
             r"remove nth node|middle node|merge two sorted linked|reverse singly"),
    # Trees & graphs
    ("ch04", r"\btree|trie|graph|binary heap|min-heap|max-heap|heap push|heap operations|"
             r"heapsort|priority queue|\bbfs\b|\bdfs\b|dijkstra|bellman|floyd|topolog|kruskal|"
             r"prim's|union-find|disjoint set|island|flood fill|\bgrid\b|adjacency|\blca\b|ancestor|"
             r"avl|red-black|euler|bipartite|matching|max flow|maxflow|min-cut|mincut|edmonds|dinic|"
             r"flow network|articulation|bridge edge|kosaraju|tarjan|connected component|cycle|"
             r"shortest path|\bmst\b|spanning tree|word ladder|isomorph|subgraph|motif|triangle|"
             r"pagerank|graph|huffman tree|n-ary|zigzag|vertical order|tree traversal|"
             r"level order|inorder|preorder|postorder|bst\b|binary search tree|serialize and deserialize|"
             r"root-to-leaf|tree depth|balanced tree|mirror|symmetric tree|path in|"
             r"grap|minimum height|itinerary|eulerian|steiner"),
    # Object-oriented design
    ("ch07", r"design|object-oriented|\bclass\b|lru|lfu|cache\b|iterator|singleton|"
             r"finite state machine|\bfsm\b|rate limit|url shortener|snapshot array|calendar|"
             r"hit counter|randomized set|parking|vending|elevator|deck of cards|call center|"
             r"online chat|jigsaw|library management|book|movie|model|versioned|caching|"
             r"sparse (matrix|table)|serialize|deserialize|json|streaming|stream\b|"
             r"observable|visitor|strategy|factory|builder|adapter|decorator pattern|"
             r"dependency injection|plugin|extension point|abstraction|interface|"
             r"parse|parser|tokeniz|grammar|render pipeline|scene|sprite"),
    # Arrays & strings
    ("ch01", r"string|char|array|substring|anagram|palindrome|vowel|consonant|matrix|rotate|"
             r"transpose|prefix sum|sliding window|two.sum|three.sum|interval|histogram|urlify|"
             r"atoi|itoa|ipv4|ip address|word count|trim|spaces|merge two|duplicate|"
             r"missing number|second largest|element|count|frequency|compress|encode|decode|"
             r"\burl\b|flatten|symmetric|matrix.*symmetric|diagonal|compress|sieve|"
             r"prefixes|common prefix|column|row|segment|list\b"),
]

TIER_FALLBACK = {"beginner": "ch16", "intermediate": "ch16", "advanced": "ch17",
                 "expert": "ch17", "master": "ch17"}


def parse_readme() -> list[dict]:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    tasks: list[dict] = []
    tier = None
    for line in text.splitlines():
        h = re.match(r"^#\s+(.+?)\s*$", line)
        if h and not re.match(r"^\d+\.\s", line):
            name = re.sub(r"[^a-z]", "", h.group(1).lower())
            tier = {"beginner": "beginner", "intermediate": "intermediate",
                    "advanced": "advanced", "expert": "expert",
                    "masterchallenge": "master", "master": "master"}.get(name, tier)
            continue
        m = re.match(r"^(\d+)\.\s+(.+?)\s*$", line)
        if m and tier is not None:
            tasks.append({"number": int(m.group(1)), "title": m.group(2), "tier": tier})
    return tasks


def classify(title: str, number: int, tier: str) -> str:
    if number in OVERRIDES:
        return OVERRIDES[number]
    t = title.lower()
    # Rule guards: specific phrases beat generic keyword matches
    if re.search(r"\btopolog", t):
        return "ch04"
    if re.search(r"\blru\b|\blfu\b", t):
        return "ch07"
    if re.search(r"binary search tree|\bbst\b|\btree\b|\btrie\b", t):
        return "ch04"
    for chapter, pattern in RULES:
        if re.search(pattern, t, re.IGNORECASE):
            return chapter
    return TIER_FALLBACK.get(tier, "ch17")


def main() -> None:
    tasks = parse_readme()
    mapping: dict[str, str] = {}
    by_chapter: dict[str, list[dict]] = {}
    for t in tasks:
        ch = classify(t["title"], t["number"], t["tier"])
        mapping[str(t["number"])] = CH[ch]
        by_chapter.setdefault(ch, []).append(t)

    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "ctci-mapping.json").write_text(
        json.dumps(mapping, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    # ---- human-readable mapping doc -------------------------------------
    tutorials = ROOT / "tutorials"
    tutorials.mkdir(exist_ok=True)
    tier_label = {"beginner": "Beginner", "intermediate": "Intermediate",
                  "advanced": "Advanced", "expert": "Expert", "master": "Master"}

    lines: list[str] = []
    lines.append("# Task → CTCI Chapter Mapping (all 998 tasks)\n")
    lines.append("Every task in the repo [README](../README.md) classified against the structure of\n"
                 "_**Cracking the Coding Interview, 6th Edition**_ (Gayle Laakmann McDowell).\n")
    lines.append("Generated by [`tools/ctci_map.py`](../tools/ctci_map.py) — re-run it after editing\n"
                 "the keywords/overrides to refresh this file.\n")
    lines.append("```mermaid\n"
                 "mindmap\n"
                 "  root((Cracking the Coding<br/>Interview 6th ed.))\n"
                 "    I. Interview Process\n"
                 "    VI. Big O\n"
                 "    VII. Technical Questions\n"
                 "    IX. Interview Questions\n"
                 "      Ch 1-4 Core Data Structures\n"
                 "      Ch 5-8 Algorithms & Design\n"
                 "      Ch 9-11 Systems & Testing\n"
                 "      Ch 12-15 Knowledge Based\n"
                 "      Ch 16-17 Additional Review\n"
                 "    X. Advanced Topics\n"
                 "    XI. Code Library\n"
                 "```\n")
    order = ["ch01", "ch02", "ch03", "ch04", "ch05", "ch06", "ch07", "ch08", "ch09",
             "ch10", "ch11", "ch12", "ch13", "ch14", "ch15", "advx", "ch16", "ch17",
             "bads", "bml", "bgfx", "bnum", "bsec", "bnet", "bcpl", "beng"]
    lines.append("```mermaid\n"
                 "flowchart LR\n"
                 "  subgraph F[\"I. The Interview Process\"]\n"
                 "    A1[Why interviews work this way] --> A2[How questions are selected]\n"
                 "  end\n"
                 "  subgraph B[\"VI. Big O\"]\n"
                 "    B1[Time & space complexity] --> B2[Amortized / log / recursive runtimes]\n"
                 "  end\n"
                 "  subgraph T[\"VII. Technical Questions\"]\n"
                 "    T1[Walking through a problem] --> T2[Optimize & test]\n"
                 "  end\n"
                 "  subgraph K[\"IX-X. Interview Questions\"]\n"
                 "    C1[Ch 1-4 Data Structures] --> C2[Ch 5-8 Algorithms & Design]\n"
                 "    C2 --> C3[Ch 9-11 Systems & Testing]\n"
                 "    C3 --> C4[Ch 12-15 Knowledge Based]\n"
                 "    C4 --> C5[Ch 16-17 Additional Review Problems]\n"
                 "    C5 --> C6[§X Advanced Topics]\n"
                 "  end\n"
                 "  F --> B --> T --> K\n"
                 "```\n")
    lines.append("**How to read this mapping** — tasks were classified by their dominant topic.\n"
                 "Chapters 12 (C and C++) and 13 (Java) are *knowledge-based* chapters with no numbered\n"
                 "questions; this repo covers their spirit through the per-language tutorials\n"
                 "([Java](languages/java.md), [Python](languages/python.md), [C#](languages/csharp.md),\n"
                 "[JavaScript](languages/javascript.md)). Tasks that go clearly beyond the book\n"
                 "(systems, ML, graphics, security, …) are marked **Beyond** so the book's chapters keep\n"
                 "their original meaning.\n")
    lines.append("> Re-generate with `python3 tools/ctci_map.py`.\n")
    lines.append("## Distribution\n")
    lines.append("| Chapter / topic | Tasks |")
    lines.append("|---|---|")
    for ch in order:
        n = len(by_chapter.get(ch, []))
        lines.append(f"| {CH[ch]} | {n} |")
    lines.append("")
    lines.append("```mermaid\n%%{init: {'theme': 'neutral'}}%%\n"
                 "pie showData\n"
                 "  title Task distribution across CTCI chapters")
    for ch in order:
        n = len(by_chapter.get(ch, []))
        if n:
            label = CH[ch].replace(":", "").replace("&", "and")
            lines.append(f'    "{label}" : {n}')
    lines.append("```\n")

    for ch in order:
        items = by_chapter.get(ch, [])
        if not items:
            continue
        lines.append(f"## {CH[ch]} — {len(items)} tasks\n")
        lines.append("| # | Task | Tier |")
        lines.append("|---|---|---|")
        for t in items:
            lines.append(f"| {t['number']} | {t['title']} | {tier_label[t['tier']]} |")
        lines.append("")

    (tutorials / "ctci-mapping.md").write_text("\n".join(lines), encoding="utf-8")

    # distribution report to stdout
    for ch in order:
        n = len(by_chapter.get(ch, []))
        print(f"{n:>4}  {CH[ch]}")
    print(f"total: {len(tasks)}")


if __name__ == "__main__":
    main()
