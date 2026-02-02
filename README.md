# incdbscan-rs

A high-performance Rust implementation of **IncrementalDBSCAN** with Python bindings.

IncrementalDBSCAN maintains DBSCAN clustering incrementally as data points are inserted or deleted one at a time, without re-running DBSCAN from scratch. After each update, the result is identical to running DBSCAN on the full updated dataset.

This is a complete rewrite of [incdbscan](https://github.com/DataOmbudsman/incdbscan) (Python) in Rust, using [PyO3](https://pyo3.rs/) for Python bindings. The algorithm and correctness are preserved; the implementation language and data structures change for dramatically better performance and stability.

Based on: Ester et al. 1998. *Incremental Clustering for Mining in a Data Warehousing Environment.* VLDB 1998.

## Installation

```bash
pip install incdbscan-rs
```

### From source (requires Rust toolchain)

```bash
# Install Rust if needed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build and install
pip install maturin
maturin develop --release
```

## Usage

```python
import numpy as np
from incdbscan_rs import IncrementalDBSCAN

# Create the model
db = IncrementalDBSCAN(eps=1.5, min_pts=5, p=2.0)

# Insert data points (numpy 2D array)
data = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.5, 0.5],
    [10.0, 10.0],
])
db.insert(data)

# Get cluster labels
# Returns: cluster IDs (>= 0), -1 for noise, NaN for unknown points
labels = db.get_cluster_labels(data)
# array([0., 0., 0., -1.])

# Insert more points incrementally
new_points = np.array([[10.5, 10.0], [10.0, 10.5], [11.0, 11.0], [10.5, 10.5]])
db.insert(new_points)

# Labels update incrementally - no need to recluster
labels = db.get_cluster_labels(np.array([[10.0, 10.0]]))
# Now part of a cluster instead of noise

# Delete points
deleted = db.delete(np.array([[0.0, 0.0]]))
# Returns [True] - point was found and removed
# Returns [False] if the point didn't exist
```

## API

### `IncrementalDBSCAN(eps=1.0, min_pts=5, p=2.0)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eps` | `float` | `1.0` | Radius for neighborhood queries. Two points are neighbors if their distance is <= `eps`. |
| `min_pts` | `int` | `5` | Minimum number of neighbors required for a point to be a core point. |
| `p` | `float` | `2.0` | Minkowski distance parameter. `p=2.0` is Euclidean, `p=1.0` is Manhattan, `p=inf` is Chebyshev. |

### Methods

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `insert(X)` | `ndarray (n, d)` | `None` | Insert points and update clustering. |
| `delete(X)` | `ndarray (n, d)` | `list[bool]` | Delete points. Returns whether each point was found. |
| `get_cluster_labels(X)` | `ndarray (n, d)` | `ndarray (n,)` | Get labels: `>= 0` = cluster, `-1` = noise, `NaN` = not found. |

**Important:** All input arrays must be `float64` (`np.float64`). If your data comes from pandas or another source as `float32` or `int`, convert it first:

```python
data = data.astype(np.float64)
```

## Performance

### Benchmarks vs Python incdbscan

Measured on the same machine with identical data (random 2D points, `eps=2.0`, `min_pts=5`). Each benchmark inserts all points, then deletes half.

#### Insertion speed

| Dataset size | Python | Rust | Speedup |
|---|---|---|---|
| 200 points | 0.296s | 0.001s | **210x** |
| 500 points | 0.494s | 0.003s | **147x** |
| 1000 points | 1.087s | 0.011s | **100x** |
| 500 pts, 10D | 0.484s | 0.001s | **425x** |

The Python version rebuilds a KD-tree (`sklearn.NearestNeighbors.fit()`) on every single insertion -- O(n log n) per insert. The Rust version uses a flat `Vec` with O(1) append and O(n) brute-force query, which wins massively because the tree rebuild is the bottleneck.

#### Deletion speed

| Dataset size | Python | Rust | Speedup |
|---|---|---|---|
| 200 pts, delete 100 | 0.003s | 0.0002s | **14x** |
| 500 pts, delete 250 | 0.098s | 0.014s | **7x** |
| 1000 pts, delete 500 | 1.478s | 0.240s | **6x** |

Deletion involves BFS-based split detection, which has similar algorithmic complexity in both versions. Gains come from Rust's tight loops, no Python object overhead, and no FFI callback crossing.

#### Overall (insert + delete)

| Dataset size | Python | Rust | Speedup |
|---|---|---|---|
| 200 points | 0.299s | 0.002s | **184x** |
| 500 points | 0.592s | 0.017s | **34x** |
| 1000 points | 2.566s | 0.251s | **10x** |

### Stress test: 10 batches of 500 points

Simulates a real workload: insert 500 points per batch, delete 100 per batch, 10 batches total (5000 inserts, 1000 deletes).

#### Python incdbscan

```
Batch  1: insert=1.85s   delete=0.21s    mem=810KB
Batch  5: insert=2.19s   delete=13.06s   mem=4,987KB
Batch 10: insert=3.13s   delete=58.22s   mem=14,009KB
```

Deletion time grows from 0.2s to 58s. Memory grows linearly at ~1.4 MB per batch. The Python version may crash with `RecursionError: maximum recursion depth exceeded` at larger scales due to circular object references and callback-based BFS (see [Stability](#stability)).

#### incdbscan-rs

```
Batch  1: insert=0.003s  delete=0.01s    mem=12KB
Batch  5: insert=0.027s  delete=0.60s    mem=12KB
Batch 10: insert=0.086s  delete=2.65s    mem=13KB
```

Deletion time grows from 0.01s to 2.6s (inherent to the algorithm), but remains 22x faster than Python throughout. Python-side memory stays flat at 12-13 KB because all data lives in Rust's heap.

### Correctness verification

The Rust version produces **identical results** to both the Python incdbscan and sklearn's DBSCAN:

- All benchmarks above show matching cluster counts and noise counts between Python and Rust
- Cross-validation against `sklearn.cluster.DBSCAN` confirms label assignments are isomorphic (same clustering, potentially different label numbering)
- Tested scenarios: cluster creation, absorption, merge, 2-way split, 3-way split, duplicate handling, noise detection, multi-dimensional data (2D through 100D)

## Stability

### Why the Python version crashes on long-running tasks

The Python `incdbscan` can hit `RecursionError: maximum recursion depth exceeded` after several batches of insertions/deletions. There are two root causes:

1. **Circular object references.** Each `Object` stores `self.neighbors = {self}`, and neighbors cross-reference each other. After thousands of insertions, Python's cyclic garbage collector must trace these chains, which can exceed the default recursion limit (1000) during GC sweeps.

2. **BFS via Python callbacks.** Split detection uses `rustworkx.bfs_search()` which invokes a Python `BFSVisitor` callback for every node and edge. In dense graphs, these callbacks accumulate on the call stack.

3. **Quadratic memory operations.** `numpy.insert()` copies the entire coordinate array on every single point insertion, causing O(n^2) total memory operations and heavy GC pressure.

### Why incdbscan-rs is immune

| Concern | Python | Rust |
|---|---|---|
| Recursion | BFS via visitor callbacks, GC cycle tracing | Zero recursion -- all traversals are iterative loops |
| Memory model | Circular `Object` references, cyclic GC | `u64` IDs in `HashMap` and `petgraph` -- no reference cycles, no GC |
| Spatial index | KD-tree rebuild + numpy array copy per insert | Flat `Vec` with O(1) append -- no copies |
| Stack growth | Proportional to graph size via callbacks | Constant -- heap-allocated `VecDeque` for BFS |
| Python-side memory | 14 MB at batch 10, growing ~1.4 MB/batch | 13 KB flat -- all data lives in Rust heap |

The Rust version has no call stack growth proportional to data size. Every graph traversal uses `while let Some(node) = queue.pop_front() { ... }` with a heap-allocated queue.

## Architecture

```
src/
├── lib.rs              # PyO3 module + IncrementalDBSCAN pyclass
├── engine.rs           # Pure-Rust IncrementalDbscan entry point
├── types.rs            # ObjectId (u64), ClusterLabel (i64), constants, hash function
├── distance.rs         # Minkowski distance family (p=2 optimized with squared distance)
├── object.rs           # ObjectData struct (id, count, neighbor_count, core status)
├── spatial_index.rs    # Brute-force spatial index (Vec-based, O(1) insert, O(n) query)
├── labels.rs           # LabelHandler (bidirectional HashMap mapping)
├── objects.rs          # Central manager (petgraph StableGraph + spatial index + labels)
├── inserter.rs         # Insertion algorithm (creation / absorption / merge)
├── deleter.rs          # Deletion algorithm (split detection, border reassignment)
└── bfs_split.rs        # Multi-source BFS for cluster split detection
```

### Key design decisions

- **`petgraph::StableGraph`** instead of a plain graph. Stable node indices survive node removal, which is critical since we store `NodeIndex` values in hash maps.
- **No neighbor set on objects.** The Python version stores `obj.neighbors` as a set. Rust queries `graph.neighbors(node_idx)` directly, avoiding duplicated state and circular references.
- **`DeletedObjectInfo` pattern.** Python accesses a deleted object's neighbors after deletion (the object persists in memory via GC). Rust snapshots neighbor data into a struct before removal.
- **Brute-force spatial index.** O(1) insert + O(n) query per insert beats Python's O(n log n) tree rebuild + O(log n + k) query, because the rebuild dominates. Upgradeable to grid-based spatial hashing for very large datasets.
- **Feature-gated PyO3.** PyO3 bindings are behind the `extension-module` Cargo feature, so `cargo test` runs pure Rust tests without requiring a Python interpreter.

## Running tests

### Rust unit tests (24 tests)

```bash
cargo test
```

Tests cover: distance calculations, hashing, spatial index operations, label management, object data structures.

### Python tests (36 tests)

```bash
pip install incdbscan-rs[dev]
pytest
```

Tests cover: construction, noise, cluster creation, absorption, merge, duplicates, deletion, 2-way/3-way splits, reinsert, multi-dimensional (1D-50D), distance metrics, sklearn cross-validation, stress testing.

## Differences from Python incdbscan

| Feature | Python incdbscan | incdbscan-rs |
|---|---|---|
| Distance metrics | Any sklearn metric | Minkowski family only (p=1, 2, inf, or any p >= 1) |
| `delete()` return value | Returns `self` | Returns `list[bool]` (whether each point was found) |
| Warnings for missing objects | `IncrementalDBSCANWarning` | `NaN` in labels, `False` in delete results |
| Dependencies | numpy, scikit-learn, rustworkx, sortedcontainers, xxhash | numpy (Python side only) |
| Minimum Python | 3.9 | 3.9 |

## License

BSD-3-Clause. See [LICENSE](LICENSE).

Based on the [incdbscan](https://github.com/DataOmbudsman/incdbscan) project by Arpad Fulop.
