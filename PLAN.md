# GraphGen C++ Speedup Plan

Prioritized speedup plan based on inspection of `generator.cpp`, `graph_wrapper.h`, `undirected_graphs.h`, `instance.h`, and `matrix.h`. Ordered by expected impact per effort. No code changes made yet.

> **Reading order & consistency notes.** The document is written in three layers that grew over time:
> 1. **Sections 0–5** — V1 profiling analysis. Diagnoses per-tier speedups against the current Boost-based codebase.
> 2. **Sections R1–R15** — refactor plan. **R5 Level 2 (drop Boost for CSR) and R15 (flatten the object hierarchy) are the current recommendation.** The "Suggested refactor order" section further down bundles R5 Level 2 + R15 + Tier 3E as one integrated rewrite.
> 3. **IMPROVEMENTS I1–I9** — opportunities the refactor unlocks.
>
> **Stale-under-R5-Level-2 sections** (Boost-specific tuning that becomes moot once Boost is dropped for CSR):
> - **Section 3B** (`boost::vecS` / `boost::hash_setS` vs `boost::setS`) — irrelevant after R5 Level 2.
> - **Section 3D** (buffer reuse) — already flagged as superseded by 3E.
> - **Section 5's `boost::sorted_erdos_renyi_iterator` bullet** — irrelevant after R5 Level 2.
> - **"Suggested order" step 5 (Tier 3B)** — skip if following the refactor order.
> - **R4 (header split + PCH)** — already flagged as less relevant once Boost is dropped.
> - **R2, R3** — already struck through as superseded by R15.
>
> **Boost speedup discussions in Sections 2E, 3A, 5, R4, R5** describe *why* Boost is expensive and motivate R5 Level 2. They remain accurate as diagnosis; the *fix* they prescribe is R5 Level 2 rather than in-place Boost tuning. Individual sections that could be misread as "tune Boost" are flagged inline.
>
> **Stage-class count is three, not four.** R15 as originally written shows four pipeline stage classes (`GraphSampler`, `TaskComputer`, `ScratchpadBuilder`, `Tokenizer`). **I10-Followup deletes `ScratchpadBuilder`** and moves scratchpad construction into `TaskComputer::run`, which now returns both `Task` and `Scratchpad` in one call. Any diagram, class-count table, code sample, or file-layout listing below that shows `ScratchpadBuilder` as a standalone stage is superseded — read it with `ScratchpadBuilder`'s content understood as private methods on `TaskComputer`. Key structural references (stage-class hierarchy diagram, `Worker` declaration, file layout, standalone `ScratchpadBuilder` class definition) have been updated inline; prose that still says "the four stages" is historical narrative rather than a design claim.

## 0. Profile before you cut

You'll waste time optimizing the wrong thing without numbers. Pick whichever of these is convenient — none of them require code changes except the in-source option.

### Cross-platform, zero-setup

- **In-source timers.** You already have `time_before/time_after` blocks commented out in `graph_wrapper.h` and `generator.cpp`. Re-enable them per phase (`graph gen`, `johnson`, `tokenize`, `pack`) and print per-batch totals. Cheapest way to answer "which phase dominates?".
- **Python-side wall-clock.** From Python, `time.perf_counter()` around calls like `generator.erdos_renyi_n(...)` in a loop gives instances/sec. Sweep `batch_size`, `num_nodes`, `p` and see how throughput scales — that alone often points at the bottleneck (e.g. quadratic scaling with N ⇒ Johnson; linear-in-batch overhead ⇒ Python packaging).
- **cProfile / py-spy on the Python driver.** Confirms whether time is inside the C++ call at all or in surrounding Python. `py-spy record -o out.svg -- python your_script.py` produces a flamegraph and works on macOS + Linux without recompiling.

### Sampling profilers (need a build, but portable)

- **`perf` (Linux) / `Instruments → Time Profiler` (macOS) / `Windows Performance Recorder`.** All three are sampling profilers; the workflow is the same: build with `-O2 -g -fno-omit-frame-pointer`, run the Python driver, attach the profiler, and look at where C++ time is spent. On macOS, `xcrun xctrace` is the CLI form of Instruments.
- **`perf record -g` + `perf report`** on Linux is the gold standard and produces the same information as Instruments. Add `-fno-omit-frame-pointer` to `install.sh` for legible stacks.
- **[FlameGraph](https://github.com/brendangregg/FlameGraph)** turns either `perf` or `dtrace` output into an SVG. Cross-platform once you have the stack samples.

### Allocation / cache profilers (also portable)

- **`heaptrack`** (Linux) or **`valgrind --tool=massif`** — quantify how much time is malloc/free, which is a real concern here given the `unique_ptr<vector<vector<int>>>` pattern.
- **`Instruments → Allocations`** on macOS is the same idea.
- **`perf stat -e cache-misses,cache-references,instructions,cycles`** (Linux) gives you IPC and cache-miss rate; useful to confirm the `vecS` vs `setS` and flat-matrix wins in Tier 3.

### What to look for first

- If the shortest-paths / distance-matrix phase dominates → Tier 2A.
- If malloc/free or `operator new` shows up near the top → Tier 3.
- If Python-side time (`package_for_model`, pybind glue) shows up → Tier 4A.
- If a single core is pinned while others idle during a training run → Tier 4B.

## 1. Compiler flags (small, free)

In `install.sh`:

- Add `-march=native -ffast-math -fno-math-errno -flto` on both branches. `-march=native` alone commonly buys 10–30% on tight numeric loops.
- Consider re-enabling something equivalent to `-Ofast` on GCC via `-O3 -ffast-math`. (You lost fast-math when I switched clang from `-Ofast` to `-O3`.)
- Once stable, drop the bounds-check `throw` in `Matrix::operator()` under `NDEBUG` — it adds a branch in every 2D indexing call inside hot tokenization loops.

### Debuggability caveats for these flags

Not all "fast" flags are equal for debugging. Ranked from least to most disruptive:

- **`-march=native`** — no debug impact. Stack traces, breakpoints, watchpoints unchanged. Only real cost is portability: the resulting `.so` will `SIGILL` on a machine with an older CPU. Fine for a personal dev machine.
- **`-fno-math-errno`** — no debug impact in practice. Only matters if code reads `errno` after math calls (this codebase doesn't).
- **`-flto`** — mostly safe, but functions get inlined across translation units so:
  - A function you set a breakpoint in may not exist as a distinct symbol.
  - Locals can appear `<optimized out>` in gdb/lldb.
  - Link times get noticeably longer.
- **`-ffast-math`** — this is the risky one. It's really ~7 sub-flags, including:
  - **NaN/Inf semantics become undefined.** `isnan(x)` may be compiled to `false` unconditionally. If your code ever produces a NaN (bug or otherwise), branches meant to catch it silently disappear, and printed values may not match values used by branches. Genuinely confusing to debug when it happens.
  - **Reassociation** — `(a+b)+c` may become `a+(b+c)`. Regression tests that bit-compare outputs will diverge.
  - **Reciprocal replacement** — `x/y` becomes `x * (1/y)`. Off by a few ULP.
  - **Denormal flushing** — very small floats snap to zero.

  A middle-ground alternative that keeps most of the perf without the NaN traps:
  `-fno-math-errno -fno-signed-zeros -fno-trapping-math`.

  For this specific codebase the risk is low (only `euclidean_generator` does non-trivial float math), but keep it in mind if you later add code that inspects `isnan(...)` or does bit-exact comparisons.

### Recommended: split into three build modes

Rather than one blanket set of flags in `install.sh`, keep three modes and pick per task:

- **Release** (training runs, what `install.sh` ships today):
  ```
  -O3 -DNDEBUG -march=native -ffast-math -fno-math-errno -flto
  ```
- **RelWithDebInfo** (profiling and perf debugging — this is the mode you want when running Tier 0):
  ```
  -O2 -g -fno-omit-frame-pointer -DNDEBUG -march=native
  ```
  No `-ffast-math`, no `-flto`. Fast enough to reflect real behavior, but legible stacks and line numbers in the profiler.
- **Debug** (correctness bugs, rarely needed once things work):
  ```
  -O0 -g -fno-omit-frame-pointer
  ```

A `BUILD_MODE=release|profile|debug` env var switched on inside `install.sh` is the low-friction way to expose this.

## 2. Algorithmic — the big wins

### 2A. Replace `johnson_all_pairs_shortest_paths` with multi-source unweighted BFS

Every graph in your generator uses **uniform edge weights of 1** (`make_edge_weights` writes 1 to every edge before every Johnson call). Johnson is designed for weighted graphs with possible negative weights; on unweighted graphs it's massive overkill.

- Replace `johnson<D>()` in `graph_wrapper.h::get_distances` with N BFS calls (or a single multi-source BFS if you eventually want that). Complexity drops from ~O(VE + V² log V) to O(V(V+E)); for sparse graphs (Erdős–Rényi with p ~ 1/N, trees, path-star) this is often a 5–20× win on the shortest-paths phase.
- Bonus: you can then delete `make_edge_weights` entirely and drop `EdgeWeightProperty` from the graph type, which shrinks per-edge memory and speeds up graph construction.

#### What if we later want non-uniform weights?

The BFS swap is only sound while all weights are 1. If you plan to add weighted graphs later, don't lock yourself into unweighted BFS — dispatch by regime instead. The right algorithm depends on what weights are allowed:

| Weight regime | Best algorithm | All-pairs complexity |
|---|---|---|
| All weights = 1 (today) | Multi-source BFS | O(V·(V+E)) |
| Non-negative real / integer weights | Dijkstra from each source (binary heap) | O(V·(V+E) log V) |
| Small non-negative integer weights (`0..W`) | Dial's algorithm / bucket-Dijkstra | O(V·(V+E) + V·W) |
| Possibly negative, no negative cycles | Johnson (what you have now) | O(V·E + V²·log V) |
| Dense with negative weights | Floyd–Warshall | O(V³) |

Recommendations that keep future weighted support cheap:

1. **Keep `EdgeWeightProperty` in the type, but check at runtime.** Add a `bool has_uniform_weights` flag on `GraphWrapper`. In `get_distances`, dispatch: if uniform → BFS; else → Dijkstra (Boost has `dijkstra_shortest_paths`); else negative → Johnson. This is a 20-line change that keeps every regime accessible.
2. **Prefer Dijkstra over Johnson as the default weighted path.** Johnson only wins when negative weights are possible. For non-negative weights, `V` calls to `dijkstra_shortest_paths` are strictly faster than Johnson (Johnson runs Bellman–Ford *then* `V` Dijkstras).
3. **If you know your weight range is small and integer-valued**, Dial's algorithm (bucket-Dijkstra with `W+1` buckets) is ~2–3× faster than heap-Dijkstra and simple to implement. Boost doesn't ship it, but it's ~50 lines.
4. **Keep BFS as a specialization**, not the only implementation, so the fast path stays fast while the general path remains available.

Short version: the plan is still to replace Johnson with BFS *for the current uniform-weight case*, but wrap it in a dispatch layer so adding a `weighted=True` flag later is a small, local change.

#### Future extension: k-bounded distances

An additional planned change (independent of the weighted-vs-unweighted question): compute distances only up to some `k` steps rather than the full V×V matrix. Every entry with true distance `> k` becomes a single sentinel (call it `INF_K` or just `-1`), so downstream tasks (khops, bounded-reach reasoning) get a cheaper input.

This actually strengthens the case for **writing our own BFS / Dijkstra rather than calling Boost**, because:

- Boost's `breadth_first_search` and `dijkstra_shortest_paths` don't natively take a distance/hop cutoff. You can bolt one on with a visitor that `throw`s to exit early, but it's awkward and slower than just writing the loop.
- A hand-rolled bounded BFS is trivial: expand from source, stop as soon as the current frontier's level exceeds `k`. Constant-factor faster than Boost even at `k = ∞`, and asymptotically faster whenever `k` is much less than the graph diameter.
- The output matrix can shrink from dense V×V `int` to either (a) an `int8_t` matrix if `k < 128` (4× less memory, better cache), or (b) a sparse representation (per-source list of `(target, distance)` pairs with `distance ≤ k`) when `k` is small compared to V. Both are much cheaper to allocate, fill, and later pack for the model.

Design implication for 2A:

- Add a `distance_bound` parameter (default: unbounded / `INT_MAX`) to the `get_distances` / BFS entry point.
- Have the same dispatch layer choose the right algorithm:
  - **Bounded + unweighted** → bounded multi-source or per-source BFS (stop expanding past level `k`).
  - **Bounded + non-negative weights** → truncated Dijkstra (pop from the heap; when the popped distance exceeds `k`, break).
  - **Unbounded** cases fall through to the algorithms already listed in the table above.
- The `int8_t` / sparse-output decision can be gated on `k` at call time so the tokenizer sees the smallest matrix that still faithfully represents "reachable within k".

Doing 2A with the bound baked into the interface from the start avoids a second rewrite later.

### 2B. ~~Rework `floyd_warshall_frydenlund`'s connectivity tracking~~ (deferred — not used in practice)

Skipping for now. `floyd_warshall_frydenlund` isn't on the hot path in production usage, so the union-find rewrite doesn't need to happen for speed. Notes preserved in case that changes:

- Current implementation tracks components as `vector<shared_ptr<set<int>>>` with per-merge `set::insert` and iteration — heavy allocation, log-N inserts, and cache-hostile pointer chasing.
- If it ever becomes a hot path, replace the component tracking with union-find (path compression + union by rank, backed by `vector<int>`). Same asymptotics for the outer edge stream; inner constant drops by orders of magnitude and the double loop becomes vectorizable.

### 2C. Erdős–Rényi connectivity fixup

`erdos_renyi_generator` recomputes `connected_components` and rebuilds `map<int, list<int>>` in a loop until the graph is one component. Two issues:

- Recomputing components from scratch after every added edge is O(V+E) per iteration.
- `map<int, list<int>>` + `list_to_vector` for random sampling is doing extra allocations.

Fix: use union-find while adding random cross-component edges — you know exactly which components you're merging, so there's no need to re-run Boost's `connected_components` at all. Also index components by a dense `vector<vector<int>>` instead of `map<…, list<…>>`.

### 2D. Euclidean generator

`euclidean_generator` does `pow((*positions_ptr)[i][k] - (*positions_ptr)[j][k], 2)` and then a `sqrt`, per edge candidate.

- Replace `pow(dx, 2)` with `dx*dx` (compilers usually catch this with `-ffast-math`, but explicit is safer).
- Compare `dist_sq < radius*radius` instead of `sqrt(dist_sq) < radius`.
- Store positions as flat `vector<float>` of size `N*dim`, not `vector<vector<float>>`.

Also the algorithm is O(N²·dim); a kd-tree / spatial hash would drop it to ~O(N log N) for large N. Worth it only if N ≥ a few thousand.

### 2E. Random tree generation is doing per-node distribution ctors and per-edge Boost allocations

**The problem.** `random_tree_generator` in [undirected_graphs.h#L419-L427](undirected_graphs.h#L419-L427) constructs a fresh `std::discrete_distribution<int>` (or `std::binomial_distribution<>`) *inside* the per-node inner loop:

```cpp
if (probs.has_value() && !probs.value().empty()) {
    std::discrete_distribution<int> dist(probs->begin(), probs->end());  // ← per node
    num_children = dist(gen) + 1;
} else {
    num_children = d;
    if (bernoulli_p > 0.) {
        std::binomial_distribution<> dist(d, bernoulli_p);               // ← per node
        num_children = dist(gen);
    }
}
```

For each node in the tree, this pays:
- **`discrete_distribution` ctor**: O(K) work plus a heap allocation for its internal CDF vector.
- **`binomial_distribution` ctor**: BTPE parameter setup for large `n`; for small `d` still an object copy per iteration.
- **`boost::add_edge`**: two list-node allocations (undirected `adjacency_list<vecS, vecS, undirectedS>`).
- **`std::queue<int>` push/pop**: deque-based, allocates in 512-byte chunks.

For a 100-node tree that's ~100 needless distribution allocations plus ~200 boost-edge allocations. Profilers will show `malloc`/`operator new` and the boost allocator dominating this function.

**Does pre-sampling a bulk buffer help?** Yes, but the reason is subtle. Individual sample draws cost the same whether you make them one at a time or in a batch. The win comes from *what pre-sampling forces you to do*: **build the distribution once before the loop**. That's the actual lever. So:

- **Just hoisting the ctor out of the loop** gets you ~90% of the "pre-sampling" win with zero buffer overhead. Build the `discrete_distribution` (or `binomial_distribution`) once before the outer `while`, then call `dist(gen)` inside. Smallest possible change. Estimated **~2–5× speedup on the sampling work alone**.
- **Pre-sampling into a buffer** (draw `num_nodes` children-counts up front into a `std::vector<int>`, then iterate over the buffer) gets you the same speedup, *plus* fits the R10 two-phase architecture cleanly — all RNG happens in the precompute phase, the compute phase is deterministic and GIL-free (and therefore parallelisable per R10 / Tier 4B).

Recommendation: **hoist first (as part of R5 Level 2's rewrite), migrate to pre-sampled buffer when R10 lands.** They are on the same code path; doing both in sequence is one edit each.

**But the biggest single win here isn't sampling at all — it's `boost::add_edge`.** That call does two list-node allocations per edge (undirected) and is easily 5–10× more expensive per iteration than either distribution construction. R5 Level 2 (drop Boost for CSR) turns edge emission into `edges_.push_back({parent, child})` into a preallocated `std::vector<std::pair<int,int>>`. For a tree with `num_nodes` nodes, we know we'll have exactly `num_nodes - 1` edges; reserve up front:

```cpp
edges.reserve(num_nodes - 1);   // exact — trees have exactly n-1 edges
```

**Additional micro-wins in the same loop:**

1. **Hoist the `probs.has_value()` branch.** It's the same value for the whole call — either the caller passed weights or didn't. One branch per node × 100 nodes × N batch items × B batches per second = a lot of wasted branch-predictor work. Two separate loop bodies (or a strategy chosen once before the loop) removes it entirely.
2. **`std::queue<int>` → two `std::vector<int>` ping-pong buffers.** `std::queue` is a `deque` under the hood, allocating in chunks. Two `vector<int>`s with `reserve(num_nodes)` do zero allocation after the first tree if the vectors are per-`Worker` scratch (R10 already gives us the scratch home).
3. **Hoist the `end`-selection coin flip.** The `uniform_int_distribution<int>(0, 1)` per depth level is trivial but constructed per level. One `bernoulli_distribution` built once, or just draw a raw `uint32_t` from the RNG and mask its low bit.
4. **Skip `discrete_distribution` for K ≤ 2.** With one or two buckets the sampling degenerates to a coin flip; a direct `if (uniform_real(gen) < probs[0]) ...` inline is faster than the CDF machinery. Micro-win; only worth it if K is nearly always small.

**Ordered speedup ladder** (each item independent of the next; do in the order that matches whichever refactor phase is landing):

| # | Change | Where it lives | Expected win alone | Cumulative |
|---|---|---|---|---|
| 1 | Hoist `discrete_distribution` / `binomial_distribution` ctor out of the per-node loop | Trivial local edit (5 lines) | ~2–5× on the sampling work | ~2–5× |
| 2 | Drop `boost::add_edge` → `edges.push_back({p, c})` with `reserve(n-1)` | R5 Level 2 (already planned) | ~3–10× on the *whole* function | ~10–30× |
| 3 | Replace `std::queue<int>` with two `vector<int>` ping-pong buffers, reused across trees via `Worker` scratch | R10 | ~1.5× | ~15–45× |
| 4 | Pre-sample all children counts into a `vector<int>` in the precompute phase; compute phase is a deterministic tree build | R10 two-phase (already planned) | ~1.2× extra vs #1, plus GIL-free parallelism gains via Tier 4B | Compounds with Tier 4B |
| 5 | Hoist the `probs.has_value()` branch out of the loop | 5-line edit alongside #1 | ~1.1× | — |

**Where this belongs in the refactor timeline.**

- **Do #1 (hoist the ctors) immediately as a one-line fix, independent of everything else.** Zero design decisions required. Ships as a preparatory commit before R5 Level 2 even starts.
- **#2 falls out of R5 Level 2 for free.** When you rewrite tree generation against CSR, `boost::add_edge` doesn't exist to call. This is the biggest single win and it's already planned.
- **#3 and #4 fall out of R10 for free.** `Worker` gives you a home for the scratch vectors; the two-phase split gives you a home for pre-sampled buffers.
- **#5 is a nice-to-have after everything else.** Not blocking.

**Applies beyond `random_tree_generator`.** Grep for `std::discrete_distribution<` and `std::binomial_distribution<` — the same "construct-per-sample" antipattern appears in [directed_graphs.h#L214](directed_graphs.h#L214) and [directed_graphs.h#L233](directed_graphs.h#L233) (`balanced_generator`'s parent selection). The fix there is the same: hoist the constructor when the weights vector is unchanged across iterations, or accept that when the weights *do* change (in `balanced_generator`'s case they do — descendants get zeroed as you walk the graph), the ctor cost is unavoidable and the win has to come from elsewhere. See I5 for the wrapper class that makes this pattern uniform across the codebase.

**Consistency with I5 (`IntRangeSampler`).** I5 proposes wrapping "Python-supplied discrete distribution" into a value type with cached internal state. That class also solves this Tier 2E problem *by construction*: the wrapper is built once (from `GeneratorConfig`) and reused across every call site, so per-node ctors of `discrete_distribution` disappear as a side effect of I5 landing. If both I5 and R5 Level 2 land, this Tier 2E collapses to "done in passing." That's the intended endgame.

## 3. Data structures & allocation churn

### 3A. Drop `vector<vector<int>>` everywhere it's used as a matrix

`distances_ptr`, `graph_ground_truths_ptr`, `node_ranks_ptr` in `graph_wrapper.h` are all `unique_ptr<vector<vector<int>>>`. Every row is a separate heap allocation and iteration touches non-contiguous memory.

You already have `Matrix<int>` in the codebase — use it (or a plain `vector<int>` of size `N*M` accessed as `data[i*M + j]`). Impact: fewer allocations, better cache behavior, roughly a 1.5–3× speedup on the shortest-paths/ground-truth phases in my experience.

### 3B. ~~Use `boost::vecS` (or `boost::hash_setS`) instead of `boost::setS` for edges~~ **moot under R5 Level 2**

> **Consistency note.** This section is entirely about Boost-adjacency-list tuning. Under the current plan (R5 Level 2 — drop Boost for CSR, per "Suggested refactor order") there is no `setS`/`vecS`/`hash_setS` to choose between; edge storage is a flat `vector<int>` (CSR `col_indices`) built once, no dedup structure needed. Keep this section as V1-only historical reasoning; skip if implementing R5 Level 2. Also flagged at [line 1943](#) ("What's not in the order").

`Graph` is declared with `boost::setS` for edges. `setS` stores each vertex's out-edges in an ordered `std::set`, so:

- Iteration is slow (node-per-node pointer chasing),
- Every `add_edge` is O(log deg),
- Duplicate detection is built-in but expensive.

Switching to `boost::vecS` makes iteration contiguous and `add_edge` O(1) amortized. **However, `vecS` does *not* deduplicate** — `add_edge(u, v, g)` will happily create parallel edges. So the switch is only safe if we handle duplicates ourselves. Options, cheapest first:

1. **`boost::hash_setS`** — the drop-in middle ground. Same dedup guarantee as `setS`, but O(1) hashed lookup instead of O(log deg). Iteration is still per-vertex, not contiguous, so it's faster than `setS` but slower than `vecS`. Zero code changes elsewhere. Use this if in doubt.
2. **`vecS` + audit the generators.** All existing generators appear to produce unique edges by construction:
   - `erdos_renyi_generator` uses `sorted_erdos_renyi_iterator`, which emits sorted, unique `(u, v)` pairs.
   - The connectivity fixup samples one node per distinct component, so the added edge can't already exist across components (only the `if (v1 == v2) continue;` self-loop check is needed).
   - `euclidean_generator` iterates `for i, j in i+1..N`, so each pair is visited once.
   - Tree, path-star, and balanced generators build edges deterministically without repeats.

   If the audit holds up, `vecS` is safe and gives the biggest speedup. Worth explicitly asserting no-duplicates in tests to catch regressions.
3. **`vecS` + explicit dedup at construction.** Build the edge list into an `unordered_set<pair<int,int>>` (or a sorted `vector<pair<int,int>>` + `std::unique`) first, then bulk-load into the `vecS` graph via the iterator-pair constructor. Deterministic, keeps `add_edge` calls out of the hot loop, and provides dedup exactly once at build time.
4. **`vecS` + `boost::remove_parallel_edges(g)` as a safety net.** Boost provides this to strip duplicates after the fact. Cheap when duplicates are rare (mostly no-op scans); pointless if you already know construction is clean.

Practical recommendation: start with (1) `hash_setS` for a safe, small win. Once profiling confirms the audit in (2) holds for the generators actually used in training, upgrade to `vecS`. Skip Johnson/BFS-based traversal cost worries — both algorithms benefit from `vecS`'s contiguous edge iteration.

### 3C. Replace `map<int, list<int>>`, `map<int, bool>`, `map<string, int>`

- `get_connected_components_map`: `map<int, list<int>>` → `vector<vector<int>>` indexed by component id (component ids are dense 0..K-1 by construction).
- `make_node_list`: `map<int, bool> node_seen` → `vector<uint8_t> seen(N, 0)` (dense keys, O(1), branch-free).
- Global `dictionary` / `pos_dictionary`: `map<std::string, int>` → `unordered_map<std::string, int>` (or a `flat_hash_map` if you pull in absl / robin_hood). Every tokenizer lookup currently does string comparisons in a red-black tree.

### 3D. Reuse buffers across the batch

The `while` loop in each generator function builds a `GraphWrapper` + `Instance` per iteration and heap-allocates fresh matrices every time. Consider:

- A per-thread scratch arena that owns `distances`, `edge_list`, `node_shuffle_map`, tokenized `Matrix`s, and gets reset (not freed) between instances.
- Or `reserve()` and reuse the vectors on the wrapper. `unique_ptr<...>` recreation is the single most frequent malloc site.

Note: 3D is the "keep the current architecture, just stop reallocating" version. The bigger structural win is Tier 3E below — preallocate the batched output tensors up front and have each instance write directly into its slice. If you're going to do 3E, do it *instead* of 3D, not both.

### 3E. Batch-first output: preallocate the final tensors and write instances directly into them

Today's pipeline is:
1. Build each `Instance` into per-instance `Matrix<int>` buffers (many heap allocations).
2. `batched_instances.add(instance)` collects instances in a per-instance-owned structure.
3. `package_for_model(...)` allocates the final numpy arrays and copies every instance in.

That's *two* full data copies of every instance (into `Matrix`, then into numpy) plus one heap allocation per matrix per instance. If we know the batch shape at the top of the function, we can write every instance directly into the destination numpy buffer and eliminate both the intermediate `Matrix` and the final gather copy.

The blocker is that per-instance shapes (seq length, edge count, task length, thinking tokens, scratch pad length, struct dim, etc.) aren't known before generation. Three ways to handle that:

**Approach A — Two-pass with size prescan (early recommendation; superseded by D below):**
- **Pass 1 (sizing):** For each of the `batch_size` accepted instances, run only the parts of generation that determine layout (sample `num_nodes`, generate the graph, decide task, compute query / graph / task / scratch-pad / thinking-token lengths). Keep a lightweight per-instance record of these sizes. Handle rejections here (regenerate until accepted; only accepted instances contribute to the max).
- Compute `max_seq_len`, `max_struct_dim`, `max_labels`, etc. across the accepted set.
- **Allocate the final numpy arrays once** at `[B, max_seq_len, max_struct_dim]` etc. via `py::array_t<int>({B, max_seq_len, max_struct_dim})` and get raw mutable pointers.
- **Pass 2 (fill):** For each instance, run the expensive tokenization phase writing straight into its slice of the numpy buffer.

**Approach B — Preallocate to a hard upper bound (simplest, wastes memory):**
- Use existing arguments (`max_num_nodes`, `max_edges`, `num_thinking_tokens`, max query length, etc.) to compute a *theoretical* upper bound on each output dim.
- Allocate `[B, max_theoretical_seq, max_struct]` once.
- Every instance writes into its slot; unused tail is left as pad tokens.
- Simplest to implement; only viable if the upper bound isn't wildly larger than typical (otherwise you'll allocate 10× more memory than you use and hurt cache behavior).

**Approach C — Hybrid, size prescan without regenerating (earlier recommendation; superseded by D below):**
- Do a cheap sizing pass that generates only the graph and picks task/scratchpad, computing lengths but *not* tokenizing.
- Allocate final tensors from the true max.
- Then tokenize each instance directly into its slice.
- Similar wins to A but avoids the "regenerate on rejection during sizing" bookkeeping — rejections are cheap because sizing didn't do the expensive tokenization.

**Approach D — Clean precompute/compute split (recommended):**

Every piece of work in the batch loop belongs to exactly one of two phases. This is the design the plan's Phase 2 refactor should target.

*Precompute phase* — for each item in the batch, run all of:

1. `graph_sampler_.run(kind, rng, cfg)` → `SampledGraph`
2. `task_computer_.run(task_kind, scratchpad_kind, graph, rng, cfg)` → `(Task, Scratchpad)` — both produced in one call (per I10-Followup, which merges the former `ScratchpadBuilder` stage into `TaskComputer`; the two remain distinct data classes)
3. `Layout::from(graph, task, scratchpad, cfg)` → `Layout` *(pure static factory; no I/O, no allocation beyond the tiny result struct)*

Each stage class exposes a single `.run(...)` entry point rather than a class-name-derived verb (`.sample` / `.compute` / `.build`), so the three precompute-side pipeline stages read symmetrically at the call site (per I10-Followup: `GraphSampler` + `TaskComputer` — the former `ScratchpadBuilder` is now internal to `TaskComputer`) and the pattern extends cleanly if a new stage is inserted. The tokenizer stage's compute-phase method is deliberately *not* named `.run(...)` — it takes a preallocated row and writes into it rather than producing a fresh value, so it keeps its distinctive `tokenize_into_row(...)` name to flag the difference.

Member naming: the `Worker` data member is `graph_sampler_` (not `sampler_`) so it doesn't collide with the many other things the codebase samples — `IntRangeSampler` for config-driven distributions (I5), path samplers on the shortest-path DAG (I9-Clarification), and `sample_choice_node` inside `ShortestPathTask` — which are all distinct concerns.

Where `graph_sampler_`, `task_computer_` are the stage-class members owned by `Worker` (see R15; I10-Followup merges the former `scratchpad_builder_` into `task_computer_`). Store the four results per accepted item in a `std::vector<PrecomputedItem>`. Handle rejections here — any item that fails `attempt_check` is discarded before it contributes to the batch. Because tokenization hasn't happened yet, rejections are cheap.

*Allocate the batch tensors* — once all `batch_size` items have precomputed, take max over `Layout`s to get batch-wide dimensions. Allocate every `py::array_t` in one place. GIL required for this step; nothing else.

*Compute phase (fill the buffer)* — for each precomputed item, call `tokenizer_.tokenize_into_row(item.graph, item.task, item.scratchpad, item.layout, cfg, out.row(i))`. This is **pure data placement** — it writes tokens into pre-computed positions in the pre-allocated buffer. No new work, no rejections, no branching on unknown sizes. This phase releases the GIL (Tier 4A) and parallelizes trivially (Tier 4B) because every row-slot is disjoint.

Why D beats A and C:

- **Every function belongs to exactly one phase.** No "cheap sizing pass" that duplicates parts of tokenization. No "sample then re-sample" for rejections during sizing. The pipeline reads top-to-bottom with a hard architectural boundary in the middle.
- **The compute phase does zero variable-cost work.** All layout uncertainty is resolved before the tensor is allocated. `tokenize_into_row` is a straight-line data-placement function; its runtime is deterministic given its inputs.
- **Rejection handling is trivial.** Rejection happens inside the precompute phase (a rejected item is discarded before its `Layout` is added to the batch). The compute phase never sees a rejection.
- **Parallelism is trivial.** The compute phase is embarrassingly parallel — every row-slot is disjoint memory, and every input tuple is already computed. Tier 4B becomes a `#pragma omp parallel for` over the range `[0, batch_size)`.
- **Debuggability wins.** When output is wrong, you can freeze the precompute result and print it, then re-run tokenization deterministically to see where placement went wrong. Currently you'd have to re-run generation with the same RNG state — much harder.

**The one cost: memory during precompute.** You hold the whole batch's `SampledGraph + Task + Scratchpad` in memory before the compute phase starts. In practice this is *smaller* than what today's code holds temporarily, because:

- `SampledGraph` = three `vector<int>` of size ~O(E), plus a `vector<int>` of size N for `node_list`, plus optional positions (O(N)).
- `Task` = a small query struct + one or two `vector<int>` targets of size ~O(N).
- `Scratchpad` = one `vector<int>` of size ~O(scratchpad_len).

Total per item: O(N + E + scratchpad_len) ints. Today's code holds five `Matrix<int>` per instance each of size `seq_len × struct_dim`, where `seq_len` is typically dominated by tokenized graph structure. Empirically the precompute working set will be similar or smaller, and it's freed the instant the compute phase completes.

If profiling ever shows the precompute working set becoming a memory bottleneck, fall back to Approach C (which streams precompute and compute per item at the cost of losing trivial parallelism).

**Concrete `Worker::generate_batch` under Approach D:**

```cpp
py::dict Worker::generate_batch(GraphKind kind, const GeneratorConfig& cfg) {
    // ---- Precompute phase ----
    std::vector<PrecomputedItem> precomputed;
    precomputed.reserve(cfg.batch_size);

    int attempts = 0;
    while (static_cast<int>(precomputed.size()) < cfg.batch_size
           && attempts < cfg.max_attempts) {
        SampledGraph g   = graph_sampler_.run(kind, gen_, cfg);
        if (!g.passes_attempt_check(cfg)) { ++attempts; continue; }
        // Per I10-Followup: task_computer_ produces both Task and Scratchpad in one call.
        auto [t, sp]     = task_computer_.run(cfg.task_kind, cfg.scratchpad_kind, g.graph(), gen_, cfg);
        Layout       ly  = Layout::from(g, t, sp, cfg);
        precomputed.push_back({std::move(g), std::move(t), std::move(sp), ly});
        ++attempts;
    }
    GG_CHECK(static_cast<int>(precomputed.size()) == cfg.batch_size,
             "batch fill exhausted attempts");

    // ---- Allocate ----
    BatchLayout batch_layout = BatchLayout::from(precomputed);
    BatchOutputArrays out(cfg, batch_layout);            // acquires numpy buffers

    // ---- Compute phase (fill) ----
    {
        py::gil_scoped_release release;                   // Tier 4A
        // Optionally parallelize here for Tier 4B.
        for (int i = 0; i < cfg.batch_size; ++i) {
            const auto& item = precomputed[i];
            tokenizer_.tokenize_into_row(item.graph, item.task, item.scratchpad,
                                         item.layout, cfg, out.row(i));
        }
    }
    return std::move(out).to_pydict();
}
```

Structure at a glance:

- Everything above `BatchOutputArrays out(...)` is precompute — variable-time work, RNG use, rejection handling.
- Everything below is compute — deterministic time, disjoint writes, GIL released.
- The two phases never share responsibility for the same computation.

Recommendation: **do Tier 3E (approach D) instead of 3D**. It subsumes 3D's win (no per-instance allocations) and adds the "one copy, not two" win, gives you a clean precompute/compute architectural boundary that Phase 2 of the refactor plan can target directly, and it's a prerequisite for the biggest gains in Tier 4. Skip 3D unless 3E is out of scope for the sprint.

Additional wins that come along for the ride, regardless of approach:

- **One data copy instead of two.** The `Matrix<int>` intermediate goes away; instances write straight to numpy memory. For a batch of 256 with typical seq lengths in the thousands, this is a big chunk of memory bandwidth saved.
- **One allocation instead of ~10 per instance.** Every `unique_ptr<vector<vector<int>>>` (distances, ground truths, node ranks, plus per-instance Matrix buffers) collapses into stack-local pointers into the batched buffer.
- **Better cache locality during tokenization.** Writing sequentially into a large flat buffer is very cache-friendly, unlike scattered per-instance allocations.
- **Composes with parallel batch fill (Tier 4B).** Each worker thread owns a disjoint slice of the batched buffer, so there's no synchronization on writes. Merges are memcpy-free.
- **Composes with GIL release (Tier 4A).** The numpy arrays get allocated at the top (needs GIL), then the GIL is released for the entire fill loop (pure raw pointer writes), then reacquired just to build the returned `py::dict`. Currently the GIL is held for the whole call.

Trade-offs and pitfalls to flag:

- **Rejections during Pass 1.** If a graph fails `attempt_check`, you must regenerate before it counts against the batch. Approaches A and C already handle this because Pass 1 is where rejection happens.
- **Determinism / reproducibility.** RNG advance order changes when generation and tokenization are separated in phases. Not a correctness issue, but a fixed seed will produce different outputs than the current code. Document this in the migration.
- **Ragged tokenizations across instances.** The final tensor already has to be padded to max in the batch — that's what the current code does implicitly in `package_for_model`. The prescan just makes that max known before allocation instead of after.
- **Layout must match `package_for_model`'s current output contract** (dict keys, shapes, dtypes). Add a regression test that compares the new path's output to the old path's output on a fixed seed before flipping the default.

#### How to preallocate: `py::array_t<T>` directly, no intermediate `Matrix<int>`

The batched output tensors should be **`py::array_t<T>` allocated up front**, and instances should write directly into their raw buffer via `mutable_unchecked<N>()` or `mutable_data()`. No intermediate `Matrix<int>` for the final output, and no final `memcpy` at package time.

The three pybind11 options, from best to worst for this use case:

1. **Best: allocate `py::array_t<T>` first (numpy owns memory), write into it.**
   ```cpp
   py::array_t<int> arr({batch_size, max_seq, struct_dim});  // requires GIL, cheap
   auto raw = arr.mutable_unchecked<3>();                    // no bounds checks
   {
       py::gil_scoped_release release;
       for (int b = 0; b < batch_size; ++b) {
           for (int s = 0; s < seq_len[b]; ++s) {
               for (int d = 0; d < struct_dim; ++d) {
                   raw(b, s, d) = /* … */;
               }
           }
       }
   }
   return arr;  // zero copies from here to Python
   ```
   Row-major (C-order) by default. `mutable_unchecked<N>()` compiles down to plain `data[i*S1 + j*S2 + k]`, same speed as writing into a raw `int*`.

2. **Also fine (if you need custom allocation): own the buffer in C++, hand it to numpy via `py::capsule`.**
   ```cpp
   auto* buf = new std::vector<int>(B * S * D);
   // fill *buf ...
   py::capsule owner(buf, [](void* p) { delete static_cast<std::vector<int>*>(p); });
   return py::array_t<int>({B, S, D},
                           {sizeof(int)*S*D, sizeof(int)*D, sizeof(int)},
                           buf->data(), owner);
   ```
   Zero-copy. Useful only if you have a reason to want a custom allocator (arena, aligned, mmap-backed). No advantage over option 1 for plain heap allocations.

3. **Avoid: fill a `Matrix<int>` then `memcpy` into a fresh `py::array_t`.** This is the current pipeline shape and the thing 3E is trying to eliminate.

#### Is it "just a cast" from `Matrix<int>` to `py::array_t<T>`?

Almost, but not quite. If `Matrix<int>` were heap-allocated with a `data_ptr()` accessor and a public `shape()`, you could hand it to pybind11 with an owning capsule (option 2 above, parameterized on `Matrix<int>` instead of `std::vector<int>`). Zero data copy. But it requires:

- `Matrix<int>` to be heap-allocated (currently `Matrix` members on `Instance` are stack-embedded).
- A `data_ptr()` public accessor.
- A capsule that owns the whole `Matrix<int>` (not just the underlying vector).

For **batched output** there's no advantage over option 1 — allocate numpy first, done. `Matrix<int>` may still earn its keep as a *per-instance scratch buffer* in intermediate phases (before the tokenized data knows where in the batch tensor it belongs), and if a scratch phase needs `(row, col)` access it's fine to keep it there. But the *final* batched tensor should skip `Matrix<int>` entirely.

#### GIL timing

- `py::array_t<T>({...})` construction touches numpy → **requires the GIL**. Do this at the top of the function while you still have the GIL.
- Once allocated, `.mutable_data()` / `.mutable_unchecked<N>()` return raw pointers that are safe to touch **without the GIL**.
- Release the GIL for the fill loop (Tier 4A). Reacquire at the end to build the returned `py::dict`.

## 4. Concurrency

### 4A. Release the GIL around the C++ batch loop

The whole `while (batched_instances.size() < batch_size)` loop in each `*_n` function runs holding Python's GIL. That means multi-worker PyTorch `DataLoader` still serializes on this C++ code when workers share the interpreter (spawn mode is fine; fork with `num_workers>0` less so).

Wrap the pure-C++ portion with `py::gil_scoped_release`, and only re-acquire in `package_for_model` where you touch `py::dict` / `py::array`. That's often a 2–4× throughput improvement in a training loop that also does Python-side work.

### 4B. Parallelize the batch fill

Each instance in a batch is independent. With per-thread `std::mt19937` seeded deterministically from a master seed, you can fill the batch with a small `std::thread` pool (or `std::for_each(std::execution::par, ...)` if libc++ ever ships it — on macOS, use TBB or a hand-rolled pool). This scales close to linearly with cores because there's no shared state except at the final `batched_instances.add()`, which can push into a per-thread vector and merge at the end.

Caveat: the current `thread_local gen` and `set_seed()` don't cooperate — you'll need to explicitly seed each worker's RNG from the master.

## 5. Smaller cleanups (visible in profiler but not usually top hits)

- `make_edge_list` in `graph_wrapper.h`: construct with `edge_list.reserve(E)` and skip the `make_pair(-1,-1)` prefill if you don't need shuffled placement (or shuffle indices, not entries).
- `list_to_vector` inside a nested loop in `erdos_renyi_generator` — allocates `c1` and `c2` every iteration.
- `Matrix::copy_tok` / `set_tok` do per-cell writes — use `std::copy_n` / `std::fill_n` so the compiler auto-vectorizes them.
- ~~`boost::sorted_erdos_renyi_iterator` with `setS`: consider generating edges into a `vector<pair<int,int>>` and constructing the graph in one shot with the iterator pair (already what you do), but on `vecS` edges — much faster.~~ **Moot under R5 Level 2**: with Boost dropped, ER edge generation is a direct `edges.push_back({u, v})` loop into a preallocated `vector<pair<int,int>>` (see R5 replacement outline). No iterator adapter to tune.
- `pybind11` array construction in `is_in_validation` / `is_in_test`: use `py::array_t<bool>({n})` and write via `mutable_unchecked<1>()` — you're already close, but the `arr[py::make_tuple(py::ellipsis())] = false` init in `is_invalid_example` is a Python-side op; use `std::fill_n` on the raw buffer.

## Suggested order

> **Superseded for the primary path.** The list below is the *in-place-tune-V1* sequence, kept for the case where the big refactor is deferred. **The current recommendation is the "Suggested refactor order" section further down**, which bundles R5 Level 2 + R15 + Tier 3E as one integrated rewrite and makes several steps here moot.

1. Turn on profiling (30 minutes).
2. Tier 1 flags (5 minutes; ~10–20%).
3. Tier 2A (Johnson → bounded BFS behind a dispatch layer) (~half a day; often 3–10× on the shortest-paths phase).
4. Tier 3E (batch-first preallocated tensors, approach D — clean precompute/compute split) — big architectural change but subsumes 3D and unlocks the parallelism wins in 4B. If done first, 4A becomes cheap.
5. ~~Tier 3B (`hash_setS` → `vecS` after audit) — do alongside 3A (flat matrices) if picking up 3A separately.~~ **Moot under R5 Level 2** — with Boost dropped, there is no adjacency-list container to pick.
6. Tier 4A (GIL release around the pure-C++ fill loop) (~15 minutes once 3E is done; ~2× in a real training loop).
7. Tier 4B (parallel batch fill across a thread pool) — biggest core-count win, easiest to reason about after 3E lands.
8. Everything else once you know from the profiler what still matters.

Note: 3D (buffer reuse in the current architecture) is superseded by 3E and should be skipped unless 3E is deferred.

---

# Refactor plan: making C++ friendly for research iteration

Speed alone isn't the only goal — research code has to be **flexible** (new task / generator / scratchpad ideas land quickly), **debuggable** (something's off, you can find it fast), and **fast** (the reason this whole file exists). These three constraints often pull in opposite directions; the ideas below are picked to serve as many of them as possible at once.

## R1. Kill the positional-argument explosion at the pybind boundary

**Motivations (record so future changes stay aligned with intent):**

1. **Single source of truth for arguments across C++ and Python.** Config fields, defaults, and validation should live in exactly one place. Duplicated definitions across `.h` + `.py` reliably drift out of sync and cause silent bugs (e.g. a default flipped in C++ but not in Python, or vice versa) that surface only as degraded training quality — hard to attribute, hard to reproduce.
2. **This is research code and must be easy to modify.** Adding a new experiment knob should be a one-line change on each side (or ideally one line total). Anything that requires touching seven `*_n` functions, pybind bindings, and every Python caller is friction that slows research iteration and biases us against experiments that would otherwise be quick.

Every design choice in R1 (config struct, single kwargs-parsing constructor, `py::class_` + auto-generated `.pyi`) is chosen to serve these two constraints simultaneously. If a future change makes one of the two worse, revisit.

---

Every `*_n` function in `generator.cpp` takes 20+ positional args of very similar types (`bool`, `int`, `string`, `int`, `bool`, `bool`, `int`, ...). Every research change tends to add or reorder args, which breaks call sites and is bug-prone.

- Introduce a single `GeneratorConfig` (or reuse `Args`) that is constructed from `py::kwargs` at the top of every `*_n` function. Every downstream C++ call takes `const GeneratorConfig&` — no positional fan-out.
- On the Python side, mirror it as a `@dataclass` (already close in shape to the current `Args`). Python callers pass keyword args; unknown keys are validated at construction time, not silently ignored.
- Impact: (a) adding a new experiment knob = add one field, one place; (b) call sites at the pybind boundary shrink to `generate(config, kwargs)`; (c) `Args::print()` (already present) gives a canonical reproduction dump for every run.

### The config struct kills the entire `TaskArgs` / `ScratchpadArgs` inheritance hierarchy

This is worth stating explicitly because it's easy to miss: **the only reason `TaskArgs`, `ShortestPathTaskArgs`, `BFSTaskArgs`, `CenterCentroidTaskArgs`, `KhopsArgs`, `ScratchpadArgs`, `BFSScratchpadArgs` exist as an inheritance chain today is to model "different tasks have different argument sets".** Under a flat config struct, that motivation disappears.

Two ways to model per-task fields on a single flat config, in order of preference:

**Option 1 (simplest) — `std::optional<T>` per task-specific field:**

```cpp
struct GeneratorConfig {
    // shared fields
    int min_num_nodes;
    int max_num_nodes;
    TaskKind task_kind;                    // enum, from R11

    // task-specific fields (only some are populated per config)
    std::optional<int>   distance_bound;   // shortest_path / BFS
    std::optional<int>   khops_k;          // khops
    std::optional<float> center_p;         // center
    // ...
};
```

The C++ constructor validates in one place: "if `task_kind == Khops` then `khops_k` must be set". Instead of five subclasses each parsing their own args, there's one struct and one validation function. Extending: adding a new task = adding a `TaskKind::MyNewTask` enum entry + any new optional fields it needs + one more branch in the validator. **Zero new classes, zero inheritance.**

**Option 2 (structured) — `std::variant<...>` for the task-specific piece:**

```cpp
struct BfsParams          { int distance_bound; };
struct ShortestPathParams { int distance_bound; bool weighted; };
struct KhopsParams        { int k; int max_num_hops; };
struct CenterParams       { float p; };

struct GeneratorConfig {
    int min_num_nodes;
    int max_num_nodes;
    std::variant<BfsParams, ShortestPathParams, KhopsParams, CenterParams> task_params;
};
```

This is a bit more type-safe — you literally can't have `khops_k` set when `task_kind == BFS` because they live in different variant alternatives. Dispatch via `std::visit` (from R11). Slightly more machinery than Option 1, but the compiler enforces "each task carries exactly its fields, no more no less". Prefer this when a task has ~3+ specific fields; Option 1's `optional` sprawl gets unwieldy past that point.

**Same thing for scratchpads.** `ScratchpadArgs` / `BFSScratchpadArgs` collapse to either an optional field cluster or a `std::variant<BfsScratchpadParams, DfsScratchpadParams, NoneScratchpadParams>`. `ScratchpadKind` enum picks the variant.

**Concrete before/after class count for the argument layer:**

| Today (inheritance) | Under R1 (flat config) |
|---|---|
| `Args` | `GeneratorConfig` |
| `TaskArgs` (base, virtual) | *deleted* |
| `ShortestPathTaskArgs` | one variant alternative (or optional fields) |
| `BFSTaskArgs` (extends ShortestPath) | one variant alternative |
| `CenterCentroidTaskArgs` | one variant alternative |
| `KhopsArgs` | one variant alternative |
| `ScratchpadArgs` (base, virtual) | *deleted* |
| `BFSScratchpadArgs` | one variant alternative |
| `TokenizationArgs` | fields on `GeneratorConfig` |
| `PosArgs` | fields on `GeneratorConfig` |
| **10 classes, 3 inheritance chains** | **1 struct + N POD variant alternatives** |

**This is what R15 was pointing at.** The `Args` inheritance chain was modeling variation that a config struct + enum/variant models more directly, with:
- No virtual destructors
- No `unique_ptr<TaskArgs>` indirection
- No allocator on the hot path
- Compiler-checked exhaustiveness (the compiler complains at every switch/visit you didn't update when a new variant is added)
- No polymorphic dispatch cost in the tokenizer / task dispatch layer
- Trivially copyable / moveable — pass `const GeneratorConfig&` everywhere, no ownership questions

**Same logic extends to `Task`/`ScratchPad` themselves** — those hierarchies exist to model "different tasks do different work". R15 replaces them with a `TaskComputer` stage class that dispatches internally on `TaskKind` / `ScratchpadKind` and produces both `Task` and `Scratchpad` in one call (per I10-Followup, which merged the originally-planned standalone `ScratchpadBuilder` into `TaskComputer`; see R15 for the full design). The config struct kills the **argument** hierarchies; R15 kills the **runtime object** hierarchies. Same principle, applied at two layers.

### Concrete walkthrough: adding a new required arg

Take the k-bound (`distance_bound: int`) from Tier 2A as a running example. Here's what changes today vs under R1.

**Today (positional arg explosion):** to add `distance_bound` you must edit
1. Every `*_n` function signature in `generator.cpp` (seven of them, each ~20 args long) — decide where in the arg list it goes.
2. The `Args(...)` constructor's positional list in `args.h`.
3. Seven `.def("...", &fn, py::arg(...), ...)` pybind bindings at the bottom of `generator.cpp`.
4. Every downstream user that reaches for it (`GraphWrapper::get_distances`, `Instance` pass-through).
5. Every Python caller — any positional call breaks silently if the arg is inserted mid-list.
6. No Python-side schema exists, so there's no one place to see the list of valid knobs.

Any inconsistency across those layers (pybind default disagrees with C++ default disagrees with a Python caller's default) becomes a silent training-quality bug.

**Under R1 (config-struct pattern):**

1. **C++ struct — one field, one place.**
   ```cpp
   struct GeneratorConfig {
       int min_num_nodes;
       int max_num_nodes;
       float p            = -1.0f;
       std::string task_type       = "shortest_path";
       std::string scratchpad_type = "none";
       int distance_bound = -1;              // NEW: -1 = unbounded
       int batch_size   = 256;
       // ...

       explicit GeneratorConfig(const py::kwargs& kw);   // parse + validate
   };
   ```
2. **C++ constructor — one line, all validation lives here.**
   ```cpp
   GeneratorConfig::GeneratorConfig(const py::kwargs& kw) {
       parse_and_set_arg(kw, "min_num_nodes", min_num_nodes, /*required*/);
       // ...
       parse_and_set_arg(kw, "distance_bound", distance_bound, -1);   // NEW
       if (distance_bound == 0)
           throw std::invalid_argument("distance_bound must be > 0 or -1 for unbounded");
   }
   ```
3. **The seven `*_n` functions collapse to a single templated entry** that R2 already wants:
   ```cpp
   template <typename D>
   py::dict batch_generate(const std::string& graph_kind, py::kwargs kw) {
       GeneratorConfig config(kw);
       auto sampler = get_graph_sampler<D>(graph_kind, config);
       return run_batch_loop<D>(config, *sampler);
   }
   ```
4. **Pybind bindings become trivial and stop changing per new arg:**
   ```cpp
   m.def("erdos_renyi_n",
         [](py::kwargs kw){ return batch_generate<boost::undirectedS>("erdos_renyi", kw); });
   // ... one line per graph kind, unchanged when args are added
   ```
5. **Python dataclass mirror — one line.**
   ```python
   @dataclass
   class GeneratorConfig:
       min_num_nodes: int
       max_num_nodes: int | None = None
       task_type: str = "shortest_path"
       distance_bound: int = -1              # NEW
       batch_size: int = 256
       def to_kwargs(self) -> dict:
           return {k: v for k, v in self.__dict__.items() if v is not None}
   ```
   Call sites:
   ```python
   cfg = GeneratorConfig(min_num_nodes=32, distance_bound=3, task_type="shortest_path")
   batch = generator.erdos_renyi_n(**cfg.to_kwargs())
   ```

**Summary:** adding a required arg becomes a one-line change on each side (C++ struct + Python dataclass) plus wherever the value is *used* — which is the actual work, not boilerplate.

### Where "required" is enforced

Three layers, pick per-arg:

| Layer | How to enforce | When to use |
|---|---|---|
| Python `@dataclass` | No default on the field | IDE / linter flags missing args at call sites; fast local feedback. |
| C++ `GeneratorConfig` ctor | `parse_and_set_arg` variant that throws when the key is absent | Fails fast with a clear Python exception no matter who calls the C++ (tests, notebooks, other languages). |
| Downstream user | `if (config.foo < 0) throw ...` | Only when the value is required for a specific code path, not the API as a whole. |

Typical setup: **required at both the dataclass and C++ constructor layers**, with cross-field validation (e.g. "task=`khops` implies `khops_k` present") in the C++ constructor. The dataclass catches typos; the C++ constructor catches malformed configs from any source.

### Single source of truth: avoid the C++ struct + Python dataclass duplication

The straightforward version of R1 has two definitions to keep in sync — the C++ struct and the Python `@dataclass`. This is fine at first but grows into a maintenance burden as the config grows. Options to collapse to a single definition, from lightest to heaviest tooling:

**Option A — Expose the C++ struct as a pybind11 class (recommended default).** No extra tooling; pybind11 can turn `GeneratorConfig` into a Python class directly.

```cpp
py::class_<GeneratorConfig>(m, "GeneratorConfig")
    .def(py::init<py::kwargs>())                    // GeneratorConfig(**kwargs)
    .def_readwrite("min_num_nodes",  &GeneratorConfig::min_num_nodes)
    .def_readwrite("distance_bound", &GeneratorConfig::distance_bound)
    // ... one line per field
    .def("__repr__", &GeneratorConfig::to_string)
    .def("to_dict",  &GeneratorConfig::to_dict);
```

Python usage becomes:

```python
cfg = generator.GeneratorConfig(min_num_nodes=32, distance_bound=3)
batch = generator.erdos_renyi_n(cfg)
```

- Adding a field: one line in the C++ struct + one `.def_readwrite` line. No Python file to edit.
- Truly single source of truth.
- Attribute access, `__repr__`, `to_dict` all work like a dataclass.
- Not literally a `@dataclass`, so tools that *require* one (Hydra, OmegaConf, some pydantic paths) need a small wrapper.
- IDEs / mypy can't introspect the compiled `.so` for autocomplete — fix by hand-writing a small `generator.pyi` stub file (or auto-generating it from the C++ struct with a ~30-line script). Two files edited per field, but both trivial and one is optional.

**Option B — Schema-driven codegen (best when the config grows).** Write the schema once in YAML or a Python dict; a small codegen script emits both the C++ header and the Python dataclass at build time.

```yaml
# config.yaml
GeneratorConfig:
  min_num_nodes:  {type: int, required: true}
  max_num_nodes:  {type: int, default: -1}
  distance_bound: {type: int, default: -1}
  # ...
```

A ~50-line `gen_config.py` emits `generated/config.h` and `generator_config.py`. Hook it into `install.sh` before `g++` runs.

- Truly one source of truth — not "one struct plus one mirror I promise to keep in sync".
- Full control over emitted code (validation, `to_dict`, `__repr__`, docs, `.pyi`).
- Easy to add more outputs later (JSON schema, other-language bindings).
- Adds a build step; generated files show up in diffs.

**Option C — C++ reflection (`boost::pfr` / `boost::describe`).** Auto-bind every field of the struct in one loop, no per-field `.def_readwrite`. Collapses Option A to just the struct definition, but introduces non-trivial template code and Boost dependency.

**Option D — protobuf / cap'n proto / flatbuffers.** Overkill for a research project of this size. Only reach for these if configs also need to be serialized cross-language or persisted long-term.

**Recommended path:**

1. **Start with Option A** (pybind class + `def_readwrite`). Zero new tooling, immediate benefit, works today. Hand-write a `.pyi` stub for IDE hints (or skip until you miss them).
2. **Escalate to Option B** if the config grows past ~40 fields or you find yourself editing the struct + bindings + `.pyi` more than a few times a week. The codegen script is small and pays for itself quickly.
3. **Never Option D** unless requirements change dramatically.

### Decision: Option A + auto-generated `.pyi` via `pybind11-stubgen`

Locked-in choice: Option A (pybind class), with the `.pyi` stub **auto-generated** from the compiled `.so` rather than hand-maintained. This gives single-source-of-truth-in-C++ *and* IDE autocomplete for free.

**Tool: `pybind11-stubgen`** (there are alternatives — `mypy stubgen`, custom scripts — but `pybind11-stubgen` is the standard and understands pybind11 signatures better than the generic tools).

**Setup:**

```bash
# One-time install into the project venv:
.venv/bin/python -m pip install pybind11-stubgen
```

**Add a post-build step to `install.sh`:**

```bash
# After the g++ compile + setup.py install steps:
"${PYTHON}" -m pybind11_stubgen generator \
    --output-dir . \
    --exit-code
```

`--exit-code` makes stub generation fail the build if the module doesn't introspect cleanly, so type-signature regressions get caught early. Output lands at `generator.pyi` (or `generator/__init__.pyi` depending on your layout).

**Every new config field flows through automatically:**

1. Add the field to the C++ `GeneratorConfig` struct.
2. Add one `.def_readwrite("field_name", &GeneratorConfig::field_name)` line in bindings.
3. Run `bash install.sh` — the `.pyi` regenerates from the newly compiled module.
4. IDEs pick it up on next reload.

**Commit or gitignore the `.pyi`?**

Recommend **commit it**. Reasons:

- Collaborators who don't rebuild locally still get IDE hints and mypy support.
- The `.pyi` acts as a documented, review-able schema of every arg — code review sees exactly what changed in the config surface per PR.
- File is small, doesn't churn much, and diffs are informative.

Add `!*.pyi` to `.gitignore` if a broader glob would otherwise exclude it.

**Tip for docstrings:** pybind11 lets you attach `.def_readwrite("field", &..., "Short description...")` — those propagate into the `.pyi` as `# Short description...` comments and into `help(cfg)` on the Python side. Worth doing for anything non-obvious; it becomes free-tier API documentation.

**Failure modes to watch:**

- If `pybind11-stubgen` runs before the module is importable, it silently produces an empty stub. `--exit-code` catches this; also run stubgen only after `setup.py install` succeeds.
- Some complex types (nested `std::variant`, custom smart pointers) can show up as `typing.Any` in the stub. Fine for most cases; if IDE hints get vague, add explicit type conversions in the bindings.

## R2. ~~Strategy/factory pattern for tasks, scratchpads, and generators~~ **superseded by R15**

*Original suggestion below; kept for historical context. R15 does this better with less machinery.*

> Currently `Instance<D>`'s constructor is a cascading `if / else if / else if` on `task_type` and `scratchpad_type`, and `generator.cpp` has ~7 near-identical `*_n` functions differing mostly in which `graph->make_*` call they issue.
>
> - Register tasks and scratchpads in a table: `unordered_map<string, unique_ptr<TaskFactory>>`. Adding a new task = write one class + one `register("my_task", ...)` line. No touching `Instance<D>`'s constructor.
> - Same pattern for graph generators: a `GraphSampler` interface with `void sample(mt19937&, GraphWrapper<D>&, const Args&)`. Then a single templated batch loop in `generator.cpp` takes a `GraphSampler*` and drives it, replacing the 7 duplicated `while (batched_instances.size() < batch_size)` blocks.
> - Impact: (a) new ideas ship in isolated files; (b) the hot loop is written once, so speedups from Tier 3E and Tier 4 apply everywhere; (c) rewinding a change becomes deleting one file, not surgery on a mega-function.

**Why superseded:** R15 collapses the whole `Args`/`Instance`/`Task`/`ScratchPad` inheritance chain into flat data classes + one class per pipeline stage (`GraphSampler`, `TaskComputer`, `Tokenizer` — three stages per I10-Followup, which merged the originally-planned `ScratchpadBuilder` into `TaskComputer`), each dispatching internally on an enum (R11). No factory registry, no virtual dispatch, no strategy interface — a fresh new sampler is one new private method + one enum entry + one switch case, all in the same file. Strictly less machinery for the same win.

## R3. ~~Phase-separate `Instance`~~ **superseded by R15**

*Original suggestion below; kept for historical context.*

> `Instance<D>`'s constructor mixes graph construction, distance computation, task selection, scratchpad selection, tokenization, and packaging into one call. That makes each phase hard to test in isolation and hard to debug when the output is wrong.
>
> - Split into explicit phases: `build_graph`, `compute_distances_or_ranks`, `select_task`, `select_scratchpad`, `tokenize`, `pack_into_batch_slot`. Each takes the previous phase's output and returns its own.
> - Each phase gets a unit test (see R7). Debugging becomes: "which phase's output first went wrong?" — often findable in minutes instead of hours.
> - Bonus: the phase boundaries are exactly where the Tier 3E prescan pass would split (sizing phases vs fill phases), so this refactor and 3E are complementary, not competing.

**Why superseded:** R15 achieves phase separation as a *natural consequence* of deleting `Instance` entirely. Each pipeline stage becomes its own class (`GraphSampler`, `TaskComputer`, `Tokenizer` — three stages per I10-Followup, which merged the originally-planned `ScratchpadBuilder` into `TaskComputer`) with a single public method returning a data-class product; `Worker::generate_batch` orchestrates them in sequence. Tier 3E Approach D further splits execution into an explicit precompute phase and a compute phase with a hard boundary between them. No `Instance` class to phase-separate because there is no `Instance` class.

## R4. Split headers from implementations (compile time = iteration speed)

Right now every non-trivial class lives in a header (`instance.h`, `graph_wrapper.h`, `tasks.h`, `scratch_pads.h`, `undirected_graphs.h`, ...). Every touch to any of them recompiles `generator.cpp` from scratch — a ~30–60 second penalty per iteration.

- Move non-template code into `.cpp` files. Keep templates (`Graph<D>`, `GraphWrapper<D>`, `Instance<D>`) header-only, but pull as much shared code as possible out into non-templated helpers in `.cpp`s.
- Consider explicit instantiation of the two directedness variants (`GraphWrapper<boost::undirectedS>` and `GraphWrapper<boost::directedS>`) in a single `.cpp` so the templates only compile twice, not once per translation unit.
- Add precompiled headers (Boost includes are the biggest offender). `pybind11/pybind11.h` and the boost graph headers together are the top compile-time cost.
- Impact: turnaround per code change often drops from ~45s to ~5s. That directly turns into more research cycles per day.

### Is R4 the reason compile times are slow? (No — but it interacts with the real cause.)

Compile time is dominated by **what headers get parsed**, not by how the code is split into `.cpp` files. For this project, the rough breakdown per full build is:

1. **Boost Graph headers** — the single biggest offender by a wide margin. `adjacency_list.hpp`, `johnson_all_pairs_shortest.hpp`, `floyd_warshall_shortest.hpp`, `connected_components.hpp`, and the erdos_renyi generator together are legendary for compile time. Typically 20–40s on their own.
2. **pybind11 headers** — `pybind11/pybind11.h` + `stl.h` + `numpy.h`. ~5–10s of template-heavy content.
3. **Templates instantiated twice** (`<boost::undirectedS>` and `<boost::directedS>`) — small on its own but multiplied through Boost's templates.
4. **STL and your own code** — <10% of compile time.

Effect of each planned change on compile time:

| Change | Effect on compile time |
|---|---|
| **R4 alone (split .h/.cpp)** | ~no change to a *full* build. Enables incremental builds — change one `.cpp` and only that TU rebuilds. But with one `.cpp` and ten headers, touching any header still triggers a full rebuild. |
| **R4 + precompiled headers (PCH)** | 3–5× faster full builds, near-instant incremental builds. PCH precompiles Boost + pybind11 once. **Biggest ROI for effort spent** with Boost still in the codebase. |
| **R5 Level 1 (Boost wrapper)** | ~5–10% faster. You still transitively include Boost from the backend header. Not the main win. |
| **R5 Level 2 (drop Boost entirely for CSR)** | **Single biggest structural improvement.** Full builds drop from ~45s to ~5s. Every edit becomes fast without needing PCH tricks. R4 then becomes an optional convenience rather than a compile-time necessity. |

Bottom line: R4 helps *incremental* compile time, but if what hurts is the absolute rebuild wait, **the two things that actually move the needle are PCH and R5 Level 2 (dropping Boost)**.

**Recommended sequence if compile time is what's hurting:**

1. **Add a precompiled header** to `install.sh` (~30 min of work). Precompile `boost/graph/*` + `pybind11/*` once, reuse across builds.
2. **R5 Level 2** (drop Boost for CSR). Absolute rebuild time drops from ~45s to ~5s.
3. **R4** (split .h/.cpp) as a convenience once there are multiple `.cpp` files worth splitting.

**Measure first if you want to confirm.** Two commands will tell you exactly where the time goes:

```bash
# Per-header parse time (clang):
clang++ -std=c++20 -H generator.cpp 2>&1 | head -80

# Per-file compile timing (clang, produces a Chrome-tracing JSON):
clang++ -std=c++20 -ftime-trace -c generator.cpp
# Then load generator.json into https://ui.perfetto.dev/
```

On GCC use `-ftime-report` for the same information in text form. Almost always shows Boost Graph at the top for projects like this.

## R5. Reduce or isolate the Boost dependency

> **Key rationale (to be preserved as a comment in the refactored code):**
>
> The only place CSR is genuinely weak is **frequent dynamic mutation** — adding/removing edges after the graph is built. **Our workflow doesn't do this: we generate, then compute.** Edge lists are the right structure during generation, CSR is the right structure during compute. A single `.freeze()` step at the end of generation converts one to the other.
>
> This is the entire justification for why Boost Graph isn't needed here. Boost's `adjacency_list` is designed to support dynamic mutation throughout the graph's lifetime, which is why it pays for `setS` per-vertex sets (or template-heavy alternatives), fights compile times, and produces the error messages it does. Once you accept the generate-then-freeze pattern, CSR fits the workflow *better* than Boost does, not worse. Preserve this reasoning as a header comment in the graph backend so future contributors don't reintroduce Boost or dynamic-graph patterns without a clear reason.
>
> **Additional reason to ditch Boost: compile time.** Boost Graph headers are the top compile-time cost in this project (see R4 for the breakdown). Full rebuilds are ~45s today; dropping Boost for CSR brings them under ~5s. That's a ~9× improvement in edit-compile-test iteration speed — for research code where every experiment involves recompiling, this alone often pays for the refactor. Also carry this reasoning as a comment in `graph_backend.h` so someone doesn't later reintroduce a Boost dependency "because it was easier for this one algorithm" and undo the win.

Boost Graph is doing a lot for you (adjacency list, iterators, Johnson) but also costing a lot (compile time, error message walls, `setS` overhead, opaque template errors). Two levels of decoupling to consider:

- **Level 1 (light):** Put every direct Boost call behind a thin wrapper interface — `graph_backend.h` with `add_edge`, `edges`, `neighbors`, `num_vertices`. The rest of the code touches only that. Then Boost can be swapped for something simpler later without touching all callers.
- **Level 2 (heavier, but big payoff):** Replace Boost with a hand-rolled CSR (compressed sparse row) graph — three `vector<int>`s: `row_offsets`, `col_indices`, and optionally `edge_weights`. Custom BFS/Dijkstra sit on top in <100 lines each. Advantages: compiles in milliseconds, trivially debuggable (three int vectors you can print), and outperforms `adjacency_list<setS, vecS, ...>` easily. Downside: you lose Boost's algorithms library — but you were going to write your own bounded BFS anyway (Tier 2A + k-bound). This one refactor absorbs Tier 2A, Tier 3B, and half of Tier 5.

Recommend Level 1 as a first step; escalate to Level 2 if compile times or Boost's error messages keep costing you research time.

### How hard is replacing Boost Graph, really?

Short answer: **less scary than it sounds** for this codebase, because the Boost surface actually in use is small and CSR is a direct drop-in for every algorithm you're likely to add.

**Actual Boost usage in this project (audit):**

- Types: `adjacency_list<setS, vecS, un/directedS, no_property, EdgeWeightProperty>`, `graph_traits<>::edge_iterator`.
- Functions: `add_edge`, `num_vertices`, `num_edges`, `edges`, `source`, `target`.
- Property maps: `get(edge_weight, g)`.
- Algorithms: `connected_components`, `johnson_all_pairs_shortest_paths`, `floyd_warshall_all_pairs_shortest_paths`.
- Generators: `sorted_erdos_renyi_iterator`.

That's it. Most tree/path-star/balanced/khops code already builds edges manually via `add_edge` and doesn't lean on Boost's algorithm library at all.

**CSR replacement:**

```cpp
struct CsrGraph {
    int N;                          // num vertices
    std::vector<int> row_offsets;   // size N+1
    std::vector<int> col_indices;   // size E (2E for undirected)
    std::vector<int> edge_weights;  // optional, aligned with col_indices
    bool directed;
};
```

Build once from an accumulated `vector<pair<int,int>>` edge list via a one-pass degree-count + prefix-sum. ~30 lines. Then every algorithm operates on plain integer vectors.

**Reimplementation effort per algorithm (all well under 100 lines):**

| Algorithm | Approx lines | Notes |
|---|---|---|
| BFS (bounded / unbounded) | ~30 | Queue + level array. Same code as Tier 2A. |
| Multi-source BFS | ~30 | Seed queue with all sources. |
| Dijkstra (bounded) | ~40 | `std::priority_queue`. |
| Union-find connected components | ~40 | Path compression + rank. |
| Floyd–Warshall | ~20 | Dense APSP if you ever want it. |
| Erdős–Rényi generation | ~15 | Bernoulli per pair, or geometric-skip for sparse `p`. |
| Random tree, path-star, balanced | 0 | Already custom. Just point them at CSR builder. |

Total: **300–500 lines to fully replace current Boost usage**, most of which overlaps with code you're about to write anyway for Tier 2A.

**How well does CSR scale to future work?**

CSR is *the* standard representation in HPC and research graph libraries (NetworKit, GraphBLAS, SNAP, iGraph internals). Every algorithm you might add has a well-known CSR implementation.

| Extension | Effort on CSR | Effort on Boost `adjacency_list` |
|---|---|---|
| Weighted edges (int/float) | Add `edge_weights` array | Change property map type; some template pain |
| Directed vs undirected | Store one-sided vs symmetric edges | Change template parameter |
| Node attributes | Parallel `vector<T>` indexed by node | Property maps |
| Edge attributes | Parallel `vector<T>` indexed by edge | Property maps |
| Heterogeneous / typed edges | Add `edge_type` array | Complex property setup |
| Bipartite | Partition marker per node | Same |
| Very large sparse | Excellent (cache-optimal) | Depends heavily on `OutEdgeList` |
| Adding a new algorithm | Write it against 3 `vector<int>`s | Learn Boost visitor/iterator concepts |

The one place CSR is genuinely weak is **frequent dynamic mutation** — adding/removing edges after the graph is built. This codebase doesn't do that: generate → freeze into CSR → compute. Edge lists are the right structure during generation, CSR during compute; a single `.freeze()` step converts one to the other. **This is the core rationale for the whole refactor — carry it verbatim as a comment on the CSR type in the eventual `graph_backend.h`.**

**Alternatives if you'd rather use a library:**

1. **[NetworKit](https://networkit.github.io/)** — modern C++ graph analytics, pip-installable Python bindings, much cleaner API than Boost, CSR-like internals. The natural "library alternative" if you want algorithms without writing them yourself.
2. **[igraph C](https://igraph.org/c/)** — mature, comprehensive C API, wraps well with pybind. Bigger dependency.
3. **[GraphBLAS / LAGraph](https://graphblas.org/)** — algebraic graph algorithms via sparse linear algebra. Very fast on modern hardware; unusual mental model. Overkill unless scaling to enormous graphs.
4. **Roll your own CSR** — max flexibility, zero deps, most upfront work. Best for research code you want to fully understand and control.

**Concrete effort estimates:**

- **Level 1 (abstraction wrapper)** — 1–2 days. No new dependencies. Boost stays for now.
- **Level 2 (full CSR replacement, incremental)** — 3–5 days spread out. New algorithms go to CSR; existing ones migrate when touched; delete Boost once nothing depends on it.
- **NetworKit swap** — 2–3 days to wire up + relearn API. Less code owned, but you inherit a real dependency.

**Free wins that come with CSR:**

- Compile time drops by 30–60s per `generator.cpp` rebuild (Boost graph headers are the top offender).
- Error messages become legible ("index out of range on `col_indices`") instead of hundred-line template instantiation traces.
- Debugging becomes trivial: `print(row_offsets)`, `print(col_indices)`, and you can *see* the graph. You cannot meaningfully print a `boost::adjacency_list`.
- No more `setS` vs `vecS` vs `hash_setS` decision fatigue.

**Updated recommendation:**

1. **Do Level 1 first** (thin abstraction wrapper). Cheap, immediate insulation from Boost, keeps future migration flexible.
2. **Roll your own CSR incrementally, starting with Tier 2A's bounded BFS.** Every new algorithm goes to CSR. Migrate old ones opportunistically.
3. **Only consider NetworKit** if you later want algorithms (community detection, centrality, spectral methods) that aren't worth writing from scratch. Unlikely for the current "generate + distances + tokenize" workflow.

## R6. Debug macros and rich assertions

Research bugs are usually "the output isn't what I expected" rather than "the process crashed". Cheap tooling that makes those bugs findable:

- A `GG_ASSERT(cond, msg)` macro that's a no-op under `NDEBUG` and prints `file:line`, the failing expression, and any locals you pass under debug builds. Replace scattered `throw std::invalid_argument(...)` with `GG_CHECK(cond, msg)` at API boundaries (always-on) and `GG_ASSERT(cond, msg)` for inner invariants (debug-only, zero release cost).
- A `debug_dump()` method on `Instance`, `GraphWrapper`, `Task`, `ScratchPad` that writes to a stream. Same content in every class, always usable when you need to see what's happening.
- The `Matrix<T>::operator()` bounds check should be `GG_ASSERT`-gated so it disappears in release (see Tier 1) but still catches OOB accesses in debug builds.

## R7. Two-track testing: reference implementation + fast implementation

The single most common failure mode when you swap Johnson for BFS (or `setS` for `vecS`, or rewrite tokenization) is a subtle correctness regression that doesn't crash — the model just trains slightly worse. Defence in depth:

- Keep a **slow reference implementation** for every algorithm you optimize (both in C++ under `#ifdef GG_TEST` or in Python for cross-check). Give them clear names: `johnson_reference`, `bfs_reference`.
- Add a regression test that runs a fixed seed through both paths and byte-compares the output. `generator_test.cpp` already exists; expand it.
- Every optimization commit lands with "reference and fast versions produce the same output on N seeds". That's the *only* thing that lets you refactor aggressively in research code without regressing quality.

## R8. Data-oriented `GraphWrapper` (separates task-specific state from generic graph state)

`GraphWrapper<D>` currently carries both regular-graph state (`g_ptr`, `distances`, `edge_list`, ...) and khops-specific state (`khops_k`, `khops_prefix_length`, ...) side by side. The comment `/* khops graphs / This really should be typed */` is exactly the smell.

- Move khops fields out into a separate `KhopsWrapper` (or a `variant<RegularWrapper, KhopsWrapper>` on the instance).
- Same fix pattern applies to `Instance<D>`'s different task fields that go unused in most branches.
- Impact: less confusing state, no accidental "khops_k left over from previous instance" bugs, smaller structs = better cache behavior.

### Design note: khops and regular graphs do not need symmetric implementations

Khops and regular graph generation are fundamentally different pipelines:

- Regular graph generators produce a *graph object* (nodes, edges, distances), then tasks/scratchpads consume it. Everything flows: sample graph → compute distances → build task → tokenize.
- Khops generation is much closer to *token stream generation with a graph structure implied by the sequence*. There's no separate "graph then distances" phase in the same shape — the khops_k, khops_prefix_length, and segment_lengths *are* the primary output, and the graph analog is a byproduct.

The current code forces both into `GraphWrapper<D>` and a shared `Instance<D>` pipeline, which is why khops fields sit awkwardly next to regular-graph fields and the constructor branches on task type. That symmetry is aspirational, not real. **Don't force symmetric implementations where the actual data flow diverges.**

Concrete design freedoms this unlocks:

- Two **separate wrapper types** (`RegularGraphWrapper<D>`, `KhopsWrapper`) with no shared base class if none is needed. Duck-typed by whichever pipeline consumes them.
- Two **separate factories / dispatch tables** in R2 — a `GraphSampler` factory for regular generators, a `KhopsSampler` factory for khops. Trying to fit them both under one interface just creates a lowest-common-denominator API that satisfies neither.
- Optionally two **separate pybind entry points** — `regular_batch_generate(...)` and `khops_batch_generate(...)` — instead of a single dispatch that internally routes. Callers usually know which one they want anyway.
- **Only share what's genuinely shared** — the returned batched tensor format (numpy dict), the tokenization vocabulary/dictionary lookups, the config-parsing helpers. Those are the natural seams. Everything else (distance computation, task selection, scratchpad handling) can differ.

Small architectural cost of splitting: possibly some duplicated boilerplate in the batch fill loop (Tier 3E). Mitigation: make the batch fill loop a template parameterized over the sampler + instance types, so both pipelines share the framework but keep their own semantics. If duplication ever grows past ~50 lines, factor those into a helper — but do it *because* real duplication showed up, not preemptively.

Rule of thumb to record for future contributors: **shared abstractions should follow shared behavior, not the other way around.** If someone finds themselves adding an `if (is_khops)` branch inside a "regular" code path, that's the signal to split, not to widen the abstraction.

## R9. Documented invariants

Small, high-value: put a comment block at the top of `graph_wrapper.h`, `instance.h`, and each task header stating:

- The shape and semantics of every non-trivial member (`distances_ptr[i][j]` = ?, sentinel = ? for unreachable, indexing order = node-shuffle order or original order?).
- The order phases must be called in.
- Which fields are only valid after which phase.

Research code accumulates undocumented invariants faster than any other kind, and these are exactly the invariants that get violated when someone adds a new task quickly. The `include_nodes_in_graph_tokenization` / `is_direct_ranking` branching in `Instance` already hints at implicit ordering; write it down.

## R10. Kill module-level globals — split into `SharedContext` + per-thread `Worker`

The current code has a pile of module-level globals in `generator.cpp`:

- `seed_`, `gen` (RNG state).
- `dictionary`, `dictionary_num_special`, `dictionary_num_extra`, `dictionary_max_vocab`, `dictionary_extra_after_symbol`, `pos_dictionary`.
- `sample_int_partition` (the integer-partition cache).
- `validation_hashes`, `test_hashes`.

Every one of these is a hidden input to every generator function. That has three costs — one per each of our criteria:

- **Flexibility:** running two experiments in one process (different dictionaries, different seeds) is impossible without careful save/restore. Comparing configurations side-by-side requires a subprocess.
- **Debuggability:** state persists across tests. Any test that fails and leaves globals dirty causes cascading failures in later tests, and reproducing a bug requires knowing the full call history.
- **Speed:** the batch-fill parallelization plan in Tier 4B is blocked by these — you can't share the RNG or the dictionaries across threads without locks, and the `thread_local gen` workaround creates the reproducibility hazard already flagged.

### The right split: shared read-only context + per-thread worker

The globals fall into **two distinct lifetime/ownership classes** that should not be lumped into one object:

| Class | What lives here | Sharing model |
| --- | --- | --- |
| **Shared, immutable-after-build** | dictionaries, `pos_dictionary`, `validation_hashes`, `test_hashes`, `sample_int_partition`, config | One instance, shared read-only across all worker threads, no synchronization needed |
| **Per-worker, mutable** | RNG state, current seed, scratch buffers, per-thread counters | One instance per Python thread / worker, owned by that worker |

The design:

```cpp
// Built once, on the main thread, then handed to workers as const&.
// Immutable after construction — no mutation methods.
class SharedContext {
public:
    SharedContext(GeneratorConfig config,
                  Dictionary dict, Dictionary pos_dict,
                  HashSet validation_hashes, HashSet test_hashes,
                  SampleIntPartition int_partition);

    const GeneratorConfig& config() const noexcept { return config_; }
    const Dictionary&      dict()   const noexcept { return dict_; }
    // ... all accessors return const references. No setters. No mutation.

private:
    GeneratorConfig     config_;
    Dictionary          dict_;
    Dictionary          pos_dict_;
    HashSet             validation_hashes_;
    HashSet             test_hashes_;
    SampleIntPartition  int_partition_cache_;  // fully populated in ctor — see wrinkle below
};

// One per Python thread. Owns its own RNG. Holds a shared_ptr to the context.
class Worker {
public:
    Worker(std::shared_ptr<const SharedContext> ctx, uint64_t seed);

    py::dict erdos_renyi_n(py::kwargs kwargs);
    py::dict khops_n(py::kwargs kwargs);
    // ...

private:
    std::shared_ptr<const SharedContext> ctx_;  // shared, read-only
    std::mt19937_64                       gen_; // per-worker RNG
    uint64_t                              seed_;
};
```

Python-side usage (matches your threaded `DataLoader` workflow):

```python
# Main thread: build the shared context once.
ctx = generator.SharedContext(config=cfg,
                              dictionary=..., pos_dictionary=...,
                              validation_hashes=..., test_hashes=...)

# Each worker thread: construct its own Worker with a distinct seed.
def worker_init_fn(worker_id: int):
    global _worker
    # Deterministic per-worker seed derived from the master seed (see below).
    worker_seed = generator.derive_worker_seed(master_seed=42, worker_id=worker_id)
    _worker = generator.Worker(ctx, seed=worker_seed)

def generate_batch():
    return _worker.erdos_renyi_n(**cfg.to_kwargs())
```

### Why this is correct (and lock-free)

Sharing a `const SharedContext&` across threads is safe **without any synchronization** as long as the data is genuinely not mutated during generation. C++ guarantees that concurrent reads of `const` data are race-free. No mutex, no atomic, no perf cost from sharing.

Each `Worker` has its own RNG, so:

- **No RNG contention.** Zero synchronization on the hot path.
- **Per-thread reproducibility.** Given the same `(master_seed, worker_id)`, worker N produces the same sequence every run, regardless of what other workers are doing. This is the property that `thread_local gen` failed to give you.
- **Reproducibility across worker counts.** If the seeding scheme is a function of `worker_id` alone (not "next available seed"), then a 4-worker run and an 8-worker run give the same per-worker output for `worker_id ∈ {0..3}`.

### Deterministic worker seeding

Don't just do `worker_seed = master_seed + worker_id` — nearby seeds correlate in some RNGs. Use `std::seed_seq` or a splittable RNG:

```cpp
uint64_t derive_worker_seed(uint64_t master_seed, uint32_t worker_id) {
    std::seed_seq seq{
        static_cast<uint32_t>(master_seed),
        static_cast<uint32_t>(master_seed >> 32),
        worker_id
    };
    std::array<uint32_t, 2> out;
    seq.generate(out.begin(), out.end());
    return (static_cast<uint64_t>(out[1]) << 32) | out[0];
}
```

Expose this on the module so Python can derive worker seeds itself — this makes seeding transparent and testable rather than "magic that happens inside `Worker`".

### The one wrinkle: `sample_int_partition` is a *cache*

Right now `sample_int_partition` is populated lazily on first use. A lazy cache **cannot** be shared read-only across threads without a mutex. Options considered:

1. **Pre-warm in `SharedContext` constructor.** During construction (which happens on one thread, before workers are spawned), compute all partitions you'll ever need up to some `max_n`. After that point, the field is truly `const` and races are impossible. **This is the chosen approach.**
2. ~~**Move it into `Worker`** (each worker has its own cache).~~ **Ruled out — the cache is memory-intensive and per-worker duplication is prohibitive.**
3. **Guard with a `std::shared_mutex` (or per-key `std::once_flag`)** as a fallback for the rare miss after warmup. Only needed if the key set can't be fully enumerated in advance; otherwise skip it.

#### Making option 1 work: what does "pre-warm" require?

Pre-warming turns the cache into a plain `const` lookup table. That requires knowing, at `SharedContext` construction time, the full set of keys that will ever be queried during generation. Two cases:

- **Bounded key set (preferred).** If the keys are functions of a known-in-advance parameter (e.g., "partitions of every integer up to `max_n`"), compute them all in the constructor. The field becomes a `const std::vector<...>` (or `const std::unordered_map<...>`) and is trivially shareable.

  ```cpp
  SharedContext::SharedContext(GeneratorConfig config, /* ... */)
      : config_(std::move(config)),
        /* ... */,
        int_partition_cache_(precompute_partitions(config_.max_n)) {}
  ```

  Add an explicit `max_n` (or equivalent bound) to `GeneratorConfig` if it isn't there already. Failing a lookup after construction should be an assertion, not a "let me go compute that" — see R6.

- **Unbounded / data-dependent key set (fallback).** If a key might genuinely appear at runtime that wasn't foreseeable at construction, use option 3: wrap the cache in a `std::shared_mutex` with the read-mostly pattern:
  - `shared_lock` for the common case (hit): all workers read concurrently, no contention.
  - `unique_lock` only on the rare miss: one worker computes and inserts.

  This is not free — even uncontended `shared_lock` has some cost — but the miss rate should approach zero after warmup, making it acceptable. Still prefer option 1 whenever the key set can be characterized up front.

#### Diagnostic recommendation

Before implementing this: **instrument the current lazy cache** (add a counter for miss-after-first-N-calls) to see empirically whether the key set is actually bounded in practice. If, after the first few thousand generations, no new keys ever appear, then option 1 is definitely correct and you can size `max_n` from the observed range. If new keys keep appearing indefinitely, that's a signal that the cache's key structure needs rethinking — possibly the "cache" is doing something that would be cleaner as an on-the-fly computation with no memoization at all.

Bonus outcome: whether the cache actually needs to be a cache at all, or if the values it's memoizing could just be precomputed once and stored as a plain lookup table indexed by an integer — which is faster than any hash map lookup, and needs no synchronization primitives whatsoever.

### Migration cost

Moderate but mechanical:

1. Introduce `SharedContext` and `Worker` classes; move the globals into them.
2. Convert every current top-level function into a method on `Worker`.
3. Update pybind bindings: `py::class_<SharedContext>`, `py::class_<Worker>`. The V1 top-level module functions are deleted in the same cutover (per the Monitored execution plan's Step 2) — no thin-wrapper shim, no `_v2` suffix, no coexistence. `git checkout v1.0-submission` is the recovery path if V1 is ever needed again.
4. Change your Python `DataLoader` `worker_init_fn` to construct one `Worker` per thread.

Wins across the three criteria:

- **Flexibility:** multiple contexts coexist (different dictionaries, different seeds). Perfect for A/B experiments in one process. Multiple workers per context is the *native* mode, not an afterthought.
- **Debuggability:** every test creates a fresh `Worker` (and typically a fresh `SharedContext`) — no shared state, no test-ordering coupling. Bug reports become "here's the `(master_seed, worker_id)` that repro's it" — one line.
- **Speed:** parallel batch fill (Tier 4B) becomes trivial and correct. `const` sharing needs no locks; per-worker RNG needs no locks. The GIL release from Tier 4A gets its full parallelism benefit.

### Interaction with PyTorch DataLoader

R10's `SharedContext` + per-`Worker` split is exactly the shape PyTorch's `DataLoader` expects when using `num_workers > 0`. Making DataLoader work well (rather than accidentally fighting it) needs three things, all of which are already in the plan:

1. **GIL release during generation.** DataLoader with `num_workers > 0` forks worker processes, so the GIL isn't shared — but with `num_workers = 0` (single-process) DataLoader relies on the C++ side releasing the GIL for `pin_memory=True` to overlap host→GPU copies. Provided by Tier 4A.
2. **Fork-safe `SharedContext`.** DataLoader forks after Dataset construction, so `SharedContext` (dictionaries, config) must survive `fork()` and be usable read-only in every child. This is automatic if `SharedContext` is a plain `std::shared_ptr<const T>` with no threads, no file handles, no CUDA state — which is the R10 design. Assertion: no mutable state in `SharedContext` after construction.
3. **Worker-count-independent reproducibility.** Different `num_workers` values must produce identical batches from the same seed, otherwise "reproduce this run on 4 cores" doesn't work on 8 cores. Provided by I6 (per-item seeding derived from `(base_seed, batch_index, item_index)`, not from a worker-local counter).

With these three, `DataLoader(dataset, num_workers=4, prefetch_factor=2, pin_memory=True)` works out of the box. **Do not build a separate C++/Python prefetch wrapper** — it duplicates DataLoader's built-in prefetching, and can fight it (thread oversubscription, deadlocks on the GIL, worker fork races). DataLoader owns async orchestration; our C++ side owns per-call throughput and thread-friendliness.

One gotcha to write into the migration doc: **construct the `Worker` inside `worker_init_fn`, not in `Dataset.__init__`**. Otherwise every DataLoader worker inherits the same `Worker` object across the fork boundary and races on its RNG state. The correct pattern is:

```python
def worker_init_fn(worker_id):
    info = torch.utils.data.get_worker_info()
    info.dataset.worker = generator.Worker(shared_context, worker_id=worker_id)

loader = DataLoader(dataset, num_workers=4, worker_init_fn=worker_init_fn, ...)
```

Each worker process constructs its own `Worker` after fork; `SharedContext` is inherited read-only and shared safely.

## R11. Type-safe dispatch: enum classes and `std::variant` instead of string comparisons

Task type, scratchpad type, and generator kind are all handled as `std::string` today:

```cpp
if (args.task->task_type == "shortest_path") { ... }
else if (args.task->task_type == "bfs" || args.task->task_type == "BFS") { ... }
else if (args.task->task_type == "dfs" || args.task->task_type == "DFS") { ... }
// ...
```

Costs:

- **Flexibility:** adding a new task requires string-typos-not-caught-by-anything. `"shortst_path"` compiles fine and silently falls through to no-op.
- **Debuggability:** the double `"bfs" || "BFS"` case-handling is a smell — someone will forget a spelling. Case-sensitivity bugs are among the hardest to spot.
- **Speed:** string comparison per instance (small, but nonzero) and no compiler exhaustiveness check → missing branches are runtime bugs, not compile errors.

Two-part fix:

1. **Parse strings to enum classes at the boundary.**
   ```cpp
   enum class TaskType { ShortestPath, BFS, DFS, Center, Centroid, KHops, KHopsGen };
   TaskType parse_task_type(std::string_view s);   // does the case-normalization once
   ```
   Every internal dispatch is on the enum. Typos in Python become "unknown task_type" at parse time with a clear message listing valid options.

2. **Use `std::variant` for polymorphic task/scratchpad instances, and `std::visit` for dispatch.**
   ```cpp
   using TaskInstance = std::variant<ShortestPathTask, BFSTask, KHopsTask, KHopsGenTask, CenterTask>;

   TaskInstance instance = make_task(task_type, ...);
   std::visit([&](auto& task){ task.tokenize(dict, node_map, pos_dict, gen); }, instance);
   ```
   The compiler now enforces exhaustiveness — add a new task type to the variant and forget to handle it in one `visit` and you get a compile error, not a silent wrong-answer. No `unique_ptr` polymorphism, no virtual dispatch, no per-task heap allocation.

Wins:

- **Flexibility:** adding a task = write a class + one line in the enum + one line in the variant. `std::visit` covers dispatch automatically.
- **Debuggability:** compile errors instead of runtime silent-skip. IDEs can enumerate valid task types.
- **Speed:** `std::variant` is stack-allocated; `std::visit` compiles to a small switch table. Faster than `unique_ptr<Task>` + virtual dispatch, and much faster than string comparison.

Same treatment for `ScratchpadType`, `GraphKind`, `PartitionMethod`, etc.

## R12. Modern C++ hygiene

Individually small, collectively significant. All three criteria benefit:

- **Drop `using namespace std;` from headers.** It's in `matrix.h`, `graph_wrapper.h`, `undirected_graphs.h`, and elsewhere. It pollutes every translation unit, creates hard-to-diagnose ambiguity errors, and is the number one C++ style rule for a reason. Keep it in `.cpp` files only if you must.
- **Wrap the library in a namespace.** `namespace graphgen { ... }` gives IDE navigation a hook, prevents symbol collisions, and signals "this is a library, not scripts".
- **Return by value, not `unique_ptr<T>`.** `unique_ptr<vector<vector<int>>>` is a pointer to a pointer to a pointer. `Matrix<int>` (or a flat buffer) by value is faster (RVO / move) and clearer. Ownership becomes obvious.
- **`std::optional<int>` instead of `-1` sentinels.** `GraphWrapper::start = -1` and `::end = -1` are magic. `std::optional<int> start` says exactly what it means and can't be misread.
- **`std::string_view` / `std::span` for read-only parameters.** `const std::string&` and `const std::vector<T>&` in headers force callers to hand you exact types; `string_view` / `span` accept any contiguous source with zero copies.
- **`[[nodiscard]]` on functions that return status or a required value.** Ignoring a return value that mattered becomes a compile warning.
- **`noexcept` on functions that can't throw, `constexpr` on ones that can be evaluated at compile time.** Enables tail-call optimization and constant folding respectively.
- **C++20 designated initializers** for aggregate types. `GeneratorConfig{.min_num_nodes = 32, .distance_bound = 3}` is unambiguous and future-proof against field reordering.
- **`std::mdspan` (C++23) as the primary multi-dim view type — recommended, not optional.** Every 2D slot in the post-refactor codebase is a `std::vector<T>` (or `py::array_t<T>`) + a non-owning `std::mdspan<T, std::extents<int, std::dynamic_extent, std::dynamic_extent>>` view, rather than an owning 2D class like V1's `Matrix<T>`. Concrete uses: `RowView` slices into batch tensors (`BatchOutputArrays`), `DistanceMatrix` from `CsrGraph::all_pairs_distances` (I3), the per-source `d` / `dag_out_deg` / `N_v` tables cached in `SharedContext` per I9-Clarification, and any per-instance tokenizer scratch that still wants `(row, col)` indexing. `view(i, j)` reads cleaner than `data[i * cols + j]`, generates identical assembly, and interops natively with `py::array_t<T>::mutable_unchecked<2>()` (already an mdspan-shaped concept in pybind11). Toolchain: Apple clang 16+/libc++ 18+, GCC 14+, MSVC 19.40+ ship `std::mdspan` directly. On older toolchains use the header-only [Kokkos/NVIDIA `mdspan` reference implementation](https://github.com/kokkos/mdspan) (single header, C++17+); it drops into the existing `-std=c++20` flag with no other changes.
- **Prefer `unique_ptr` over `shared_ptr` unless you actually share.** The `shared_ptr<set<int>>` in `floyd_warshall_frydenlund` is expensive and unnecessary; single-owner semantics are usually what you want.

None of these individually justify a refactor. Together they turn "C++ code someone else wrote" into "C++ code you can navigate confidently".

## R13. Tooling and testing infrastructure

The three criteria all depend on tooling that catches regressions and enforces consistency, not on developer discipline. This is what separates a research codebase that stays maintainable from one that ossifies.

- **Real test framework instead of `assert` in `tests.h`.** [doctest](https://github.com/doctest/doctest) or [Catch2](https://github.com/catchorg/Catch2) — single-header, no build integration required, rich failure output showing exactly which values differed. Same header can drive C++ tests and pybind11-exposed Python tests, so R7's "reference vs fast" comparison lives naturally in the test file.
- **`clang-format` config + editor integration.** Consistent style eliminates a huge class of trivial diffs and code review friction. Commit `.clang-format` and let the editor auto-format on save.
- **`clang-tidy` config for common bugs.** Catches uninitialized variables, unused code, dangerous casts, missing `override`, and dozens of other issues. Run as part of `install.sh` or a `lint.sh` script.
- **Sanitizers in the debug build mode** (from Tier 1). Add `-fsanitize=address,undefined,leak` to the debug flags. AddressSanitizer catches use-after-free, out-of-bounds, leaks; UBSan catches signed overflow, misaligned access, invalid enum values, and more. Slower to run, but often finds bugs that would take hours to reproduce otherwise. Free with clang and gcc.
- **CMake instead of hand-rolled `install.sh`.** As soon as you have multiple `.cpp` files (from R4) or want PCH, cross-platform builds, incremental compilation, or CI, CMake will save you time. Modern CMake with pybind11 is well-documented (`find_package(pybind11 REQUIRED)`, `pybind11_add_module(...)`), and it's what everyone else in the ecosystem uses.
- **CI even if it's just GitHub Actions running `install.sh` + a small smoke test.** Catches "works on my machine" regressions. Free for public repos.
- **Compile-time budget check.** `-ftime-trace` (clang) or `-ftime-report` (gcc) in a nightly build. If compile times regress past a threshold, complain loudly. Prevents slow compile creep.
- **Reproducibility test in CI.** Fixed seed + fixed config → byte-compare against a saved reference output. Any refactor that changes output silently gets caught. Complements R7.

## R14 (~~optional, longer horizon~~ **not applicable — recorded as a considered-and-rejected alternative**)

Original suggestion: "consider replacing hand-rolled C++ where possible with numpy/numba equivalents". **After discussion, this is explicitly ruled out for this project.** The reasoning matters and is worth recording so future contributors don't re-litigate it.

### Why C++ (not numpy/numba) is the right choice here

1. **Threading model.** Generation runs inside a PyTorch `DataLoader`-style pipeline, and the goal is to parallelize batch generation across worker threads. Python-level threading is blocked by the GIL for anything that isn't a numpy kernel or a C-extension holding a released GIL. numpy operations release the GIL implicitly for large arrays, but not predictably — you can't reliably build a threaded batch pipeline out of numpy calls. C++ with `py::gil_scoped_release` around the fill loop (Tier 4A) gives us **explicit, deterministic GIL control**. That is the right primitive for parallel batch generation.
2. **Cohesion.** Splitting generation across "some in C++, some in numpy, some in numba, some in Python" scatters what's fundamentally one algorithm (sample graph → compute distances → tokenize → package) across four languages with four sets of debugging tools, four sets of type conventions, and four places to synchronize when the algorithm changes. Keeping it in one C++ module means one place to change when a research idea lands, one place to profile, one place to debug. The maintenance cost of the polyglot approach dwarfs whatever perf you'd claw back from numpy in any individual step.

So: **keep everything in C++**. The rest of this plan (perf tiers 1–4, refactor R1–R13) is scoped accordingly. If specific numerical pieces later show up as clear numpy wins during profiling (e.g. a dense-matrix operation that numpy would beat a hand-rolled loop at), they can be handled case-by-case — but as a design principle, keep the pipeline in one language.

### Corollary: invest in the C++ side, don't apologize for it

Because C++ is the right choice, treat it as a first-class codebase, not a "necessary evil":

- The refactors in R1–R13 aren't "making C++ tolerable" — they're making it a genuinely pleasant research tool.
- Tooling investment (R13) is especially worth doing because it compounds. Every hour spent on `clang-format`, `clang-tidy`, sanitizers, and doctest is repaid across every future edit.
- The Boost replacement (R5 Level 2) becomes more clearly worth it under this framing — it takes the biggest source of C++ pain (compile times, error messages) out of the picture entirely.

## R15. Flatten the object hierarchy — fewer levels of `A owns B owns C` between the pybind entry point and the algorithm

**This is the biggest architectural improvement in the plan.** It supersedes R2 and R3.

> **Note on stage-class count.** R15 as written below shows four pipeline stage classes (`GraphSampler`, `TaskComputer`, `ScratchpadBuilder`, `Tokenizer`). **I10-Followup collapses this to three** by merging `ScratchpadBuilder` into `TaskComputer`. Structural references have been updated inline; prose descriptions of "the four stages" reflect the original design and are correct as history but not as the current target.

**Design style: OO, not procedural.** Per user preference, behaviour lives in classes with methods, not in free functions over POD. What we flatten is the **object ownership hierarchy** — the chain of `A owns B owns C owns D` between the pybind entry point and the code doing the actual graph work. Each pipeline stage becomes its own class owned directly by `Worker` (one hop). Variation across kinds is expressed with enums + `std::variant` + internal `switch` dispatch inside each stage class.

Inheritance flattening happens as a *side effect* of this reorganization — the V1 `Task`/`Args`/`ScratchPad` inheritance chains disappear because they were modelling closed variation, which we handle with enums instead. But the primary target is composition depth, not inheritance depth. See Rule 4 in the Design Rules section below.

### The current call chain

Today, from `generator.erdos_renyi_n(**kwargs)` down to the actual work:

```
generator.erdos_renyi_n(**kwargs)                            (pybind)
  → Args::parse_and_set_arg(...)                             (owns TaskArgs, ScratchpadArgs, TokenizationArgs, PosArgs)
      → TaskArgs → ShortestPathTaskArgs → BFSTaskArgs        (inheritance chain, 5 classes)
      → ScratchpadArgs → BFSScratchpadArgs                   (inheritance chain)
  → BatchedInstances<D>(args)                                (owns vector<Instance>)
     for i in batch:
      → GraphWrapper<D>(...)                                 (owns Boost graph + node_list + edge_list + shuffle_map + ...)
      → Instance<D>(gen, graph, args, dict, pos_dict)        (owns GraphTokenizer, Task*, ScratchPad*, 5 Matrices, positions_ptr)
           → new GraphTokenizer(...)                         (heap allocation)
           → new BFSTask() / new ShortestPathTask() / ...    (heap allocation, polymorphic)
           → new BFSScratchPad() / new DFSScratchPad()       (heap allocation, polymorphic)
           → task->do_work()                                 (finally the algorithm)
      → batched_instances.add(instance)                      (copies the Instance into a vector)
  → batched_instances.package_for_model(dict, pos_dict)      (flattens everything into py::dict)
```

Count: **~5 owning classes** and **3 inheritance chains** between the pybind entry and the actual graph algorithm. Each Instance heap-allocates 7+ objects (`GraphWrapper`, `GraphTokenizer`, `Task`, `ScratchPad`, `positions_ptr`, plus five `Matrix<int>` members that own their own vectors). Debugging a wrong output means chasing through 5 layers of ownership indirection.

### The flat design

Everything reduces to a small set of **flat classes** (no inheritance) — data classes for the pipeline's intermediate products (`SampledGraph`, `Task`, `Scratchpad`, `Layout`) and **one class per pipeline stage** (`GraphSampler`, `TaskComputer`, `Tokenizer` — three stages, per I10-Followup, which merged the former `ScratchpadBuilder` into `TaskComputer`) that owns the behaviour for that stage. `Worker` composes them.

```cpp
py::dict Worker::generate_batch(GraphKind kind, const GeneratorConfig& cfg) {
    // ---- Precompute phase: variable-cost work, rejections happen here ----
    std::vector<PrecomputedItem> precomputed;
    precomputed.reserve(cfg.batch_size);
    while ((int)precomputed.size() < cfg.batch_size) {
        SampledGraph g   = graph_sampler_.run(kind, gen_, cfg);
        if (!g.passes_attempt_check(cfg)) continue;
        // Per I10-Followup: task_computer_ produces both Task and Scratchpad in one call.
        auto [t, sp]     = task_computer_.run(cfg.task_kind, cfg.scratchpad_kind, g.graph(), gen_, cfg);
        Layout       ly  = Layout::from(g, t, sp, cfg);
        precomputed.push_back({std::move(g), std::move(t), std::move(sp), ly});
    }

    // ---- Allocate: once, from batch-wide max layout ----
    BatchOutputArrays out(cfg, BatchLayout::from(precomputed));

    // ---- Compute phase: deterministic-cost, disjoint writes, GIL released ----
    {
        py::gil_scoped_release release;
        for (int i = 0; i < cfg.batch_size; ++i) {
            const auto& it = precomputed[i];
            tokenizer_.tokenize_into_row(it.graph, it.task, it.scratchpad,
                                         it.layout, cfg, out.row(i));
        }
    }
    return std::move(out).to_pydict();
}
```

`graph_sampler_`, `task_computer_`, `tokenizer_` are `Worker` data members (three stage classes per I10-Followup, which merged the former `scratchpad_builder_` into `task_computer_`). Each is a small, self-contained class with one public method plus per-kind private methods; internal dispatch is a `switch` on the enum. `Layout::from` and `BatchLayout::from` are static factory methods on the respective data classes.

**Two phases with a hard architectural boundary between them:**
- Everything above `BatchOutputArrays out(...)` is *precompute* — variable-time, RNG use, rejection handling.
- Everything below is *compute* — deterministic-time, disjoint writes, GIL released, trivially parallelizable for Tier 4B.
- See Tier 3E Approach D for the full rationale.

**Depth from `generate_batch` to the actual algorithm: one function call per phase stage.** That's the entire target.

What gets deleted:

| Deleted | Replaced by | Class count change |
| --- | --- | --- |
| `Args`, `TaskArgs`, `ShortestPathTaskArgs`, `BFSTaskArgs`, `CenterCentroidTaskArgs`, `KhopsArgs`, `ScratchpadArgs`, `BFSScratchpadArgs`, `TokenizationArgs`, `PosArgs` (10 classes, several with inheritance) | One flat `GeneratorConfig` struct (R1) with `std::optional`/`std::variant` per-task fields; `TaskKind`/`GraphKind`/`ScratchpadKind` enums (R11) | 10 → 1 |
| `GraphWrapper<D>` (owns Boost graph + 6 vectors + methods) | `SampledGraph` (plain struct: CSR arrays from R5 Level 2, node_list, edge_list, shuffle_map, optional positions) | 1 → 1 (but flat) |
| ~~`Task`~~, `ShortestPathTask`, `BFSTask`, `CenterTask`, `KHopsTask`, `KHopsGenTask` (6 classes, virtual dispatch) | `Task` data class (name reused after V1's polymorphic `Task` is deleted) + `TaskComputer` class (one public method, `switch`-dispatch to private per-kind methods) | 6 → 1 data + 1 stage class |
| `ScratchPad`, `BFSScratchPad`, `DFSScratchPad` (3 classes with virtual methods) | `Scratchpad` data class (tagged with `ScratchpadKind`); construction lives inside `TaskComputer` per I10-Followup (no standalone `ScratchpadBuilder` class) | 3 → 1 data class (0 new stage class) |
| `GraphTokenizer` (owns tokenization state per instance) | `Tokenizer` stage class that writes directly into pre-allocated batch tensors | 1 → 1 (thin) |
| `Instance<D>` (owns everything above) | *nothing — deleted entirely* | 1 → 0 |
| `BatchedInstances<D>` (owns `vector<Instance>` + packaging logic) | `BatchOutputArrays` (thin wrapper over the preallocated `py::array_t`s from Tier 3E) + `Worker::generate_batch()` | 1 → 1 (thin) |

**Net: ~23 classes (with 3 inheritance chains, ~15 virtual methods, ownership depth of ~5) → ~13 classes (0 inheritance, 0 virtual methods, ownership depth of 2).** The important number here is **ownership depth**, not class count — what actually hurts today isn't that there are 23 classes, it's that reaching the algorithm requires descending through 5 levels of `owns`.

### Why classes-per-stage (not free functions, not a mega-`Worker`)

Three design forces here that pull in different directions:

1. Behaviour should live with the data it operates on (favours classes).
2. Class hierarchies should be flat (favours no inheritance).
3. `Worker` shouldn't become a god-class (favours splitting the pipeline into multiple classes).

The resolution is **one class per pipeline stage**. Not "free functions over POD" (violates 1), not "everything is a method on `Worker`" (violates 3), not "one class per graph/task kind with a common base" (violates 2).

Each stage class (`GraphSampler`, `TaskComputer`, `Tokenizer` — three stages per I10-Followup) has:

- **A single public method**, e.g. `TaskComputer::compute(TaskKind, ...)`.
- **Private per-kind methods**, e.g. `compute_bfs(...)`, `compute_shortest_path(...)`.
- **Internal `switch` dispatch** on the enum inside the public method.
- **Constructor injection** for any dependencies (`SharedContext&`, config) — passed once at construction, not on every call.
- **Optional scratch buffers** as private members, reused across items in a batch (real perf win vs free functions, which can't cache anything between calls).
- **Zero inheritance, zero virtual methods.**

What this buys:

- **Testability.** Each stage is a testable unit — construct one, call its public method with crafted inputs, check the result. Signatures are short because dependencies are held as members rather than passed on every call.
- **Locality.** Everything that touches a task's algorithm lives inside `TaskComputer` (one file, easy to grep). Adding a new task = new private method + new switch case, all in one place.
- **Exhaustiveness.** `-Wswitch-enum` catches missing cases at compile time — the same guarantee as `std::visit`, without the polymorphism.
- **Scratch reuse.** `TaskComputer` can own a `std::vector<int>` distances buffer that persists across `compute()` calls within a batch. Free functions would either reallocate per call or force the caller to pass scratch space in.

What `Worker` is *not*: a mega-class that inlines all pipeline logic. `Worker::generate_batch` orchestrates — it holds instances of the stage classes and calls them in order. All actual algorithms live in the stage class they belong to.

### What about polymorphism? Isn't inheritance sometimes right?

Inheritance is right when you have an open set of types that will be extended by *code you don't control*. That's not the case here — every task, scratchpad, and graph kind lives in this codebase and is added by you. For a closed set of variants, `switch` on an enum inside a stage class beats virtual dispatch on every axis: faster (predictable branches vs indirect call), compiler-checked for exhaustiveness (`-Wswitch-enum`), zero heap allocation, no polymorphic destructors to trace through in a debugger.

### Wins across the three criteria

- **Flexibility.** Adding a new task is: (1) add a `TaskKind::MyNewTask` enum entry, (2) add a private `compute_my_new_task` method to `TaskComputer`, (3) add the switch case in `TaskComputer::compute`. All three changes live in one file. Compiler complains at every switch you didn't update (`-Wswitch-enum`) — no silent runtime fallthroughs like today's `if (task_type == "my_new_task")` chain.
- **Debuggability.** Concrete, non-templated, non-polymorphic classes. A failing test produces a stack trace with exactly the right nesting depth — one frame per stage class, not five polymorphic hops. Ownership is trivial: `Worker` owns the stage classes; each stage class owns only its scratch buffers; `SampledGraph`/`Task`/`Scratchpad` own their arrays. No `unique_ptr` chains, no virtual destructors.
- **Speed.** No per-item heap allocation for pipeline objects (only the graph *data* allocates, and that's held only briefly across the precompute/compute phase boundary per Tier 3E approach D). No virtual calls in the hot loop. Stage classes can cache scratch buffers across items in a batch (concrete perf win over the free-function alternative). The whole batch iteration inlines cleanly because there are no polymorphic barriers.

### Migration cost

High — this is the biggest single refactor in the plan. But it's mostly **mechanical** once the `SharedContext`/`Worker` split (R10) and the `GeneratorConfig` struct (R1) are in place:

1. Introduce data classes (`SampledGraph`, `Task`, `Scratchpad`, `Layout`) and stage classes (`GraphSampler`, `TaskComputer`, `Tokenizer` — three stages per I10-Followup, which merges scratchpad construction into `TaskComputer`) plus `BatchOutputArrays` as flat, non-inheriting types.
2. **Cut over.** Delete V1 wholesale (see the Monitored execution plan's Step 2 for the file list) and stand up the new pipeline as a stub that raises `GG_CHECK(false, ...)` for every kind. No side-by-side coexistence period; `git checkout v1.0-submission` is the recovery path if anything V1-only turns out to be worth resurrecting.
3. Port one graph kind end-to-end (Erdős–Rényi is simplest): add `GraphSampler::sample_erdos_renyi` + `TaskComputer::compute_shortest_path` + a case in each stage class's public `switch`. `Worker::generate_batch` orchestrates the stage classes.
4. Verify with invariant tests (path validity, distance monotonicity) and analytic ground truth where available. If a numeric-parity spot-check against V1 is wanted, run V1 manually from a `git worktree add ../graphgen-v1 v1.0-submission` sibling directory — no in-tree V1.
5. Port the rest one graph kind at a time, each one landing on the same new pipeline. No dead V1 wrappers to sweep up at the end (they were removed in step 2).

R6 (assertions) and R7 (test harness — doctest + pytest wiring, plus invariant/ground-truth patterns; see the Monitored execution plan for R7's role after the cutover) are prerequisites; without them this refactor can't be verified. That's why they come first in the suggested order.

## Proposed post-refactor class hierarchy

This is the concrete design that Phases 1–2 build toward. Everything below is a single flat header structure — no inheritance except where explicitly noted (there is exactly one place: nowhere). All types live under `namespace graphgen`.

### At a glance

```
graphgen::
├── Config layer (R1 + R11) ── plain-old-data, no inheritance
│   ├── GeneratorConfig                        (top-level; passed by const& everywhere)
│   ├── enum class TaskKind                    (BFS, ShortestPath, Center, Khops, KhopsGen)
│   ├── enum class GraphKind                   (ErdosRenyi, PathStar, Balanced, Euclidean, RandomTree)
│   ├── enum class ScratchpadKind              (None, BFS, DFS)
│   ├── std::variant<Bfs|ShortestPath|Center|Khops>Params  (task-specific fields)
│   └── std::variant<None|Bfs|Dfs>ScratchpadParams         (scratchpad-specific fields)
│
├── Threading layer (R10) ── one class each
│   ├── SharedContext                          (const-shared across all workers; no mutation methods)
│   └── Worker                                 (one per Python thread; owns RNG + shared_ptr<const SharedContext>)
│
├── Graph representation (R5 Level 2) ── ONE common class for every graph kind
│   └── CsrGraph                               (three vectors: row_offsets, col_indices, weights + bool directed)
│
├── Pipeline data (R15) ── flat classes, no inheritance
│   ├── SampledGraph                           (CsrGraph + node_list + shuffle_map + optional positions)
│   ├── Task                                   (query [source-side] + targets [target-side] — flat data class)
│   ├── Scratchpad                             (flat data class tagged by ScratchpadKind)
│   ├── Layout                                 (per-item length fields + Layout::from(...) static factory)
│   ├── PrecomputedItem                        (SampledGraph + Task + Scratchpad + Layout — held between phases)
│   └── BatchOutputArrays                      (owns Tier 3E preallocated py::array_t)
│
└── Pipeline stage classes (R15 + I10-Followup) ── three classes, no inheritance, no virtuals
    │   Precompute phase (variable-cost, RNG, rejections happen here):
    ├── GraphSampler                           .run(GraphKind, rng, cfg) → SampledGraph
    ├── TaskComputer                           .run(TaskKind, ScratchpadKind, graph, rng, cfg) → (Task, Scratchpad)
    │       (scratchpad construction lives here per I10-Followup; former standalone `ScratchpadBuilder` deleted)
    │       (Layout is computed via Layout::from(...) static factory — no separate class needed)
    │   Compute phase (deterministic, GIL-released, parallelizable):
    └── Tokenizer                              .tokenize_into_row(graph, task, sp, layout, cfg, out_row)
```

### Design rules

Four rules the whole design follows:

1. **Behaviour lives in classes.** Not free functions over POD. Each stage of the pipeline is its own class with a single public entry point; the algorithm variants for that stage are private methods on that class. Data types are also classes (with methods on them where appropriate) rather than raw structs sprinkled with helper functions.
2. **Minimize inheritance; use it only when it earns its keep.** No base classes purely to share a signature or group related types. Variation between "kinds" is expressed via enums + `std::variant` + internal `switch` dispatch, which is compiler-checked for exhaustiveness and eliminates heap allocation. Inheritance is fine when it *does* pay for itself — e.g., a genuine open extension point where third parties add subclasses, or a case where the shared interface has real semantic weight (not just "these are both tasks"). The current V1 code fails this test: `Task`/`ShortestPathTask`/... inheritance exists to model closed variation, which enums do better. Keep the option available for cases where it actually earns its cost.
3. **`CsrGraph` is the *single* representation for every graph kind.** All sampling classes produce a `CsrGraph`; all algorithms consume a `CsrGraph`. Directedness is a runtime `bool` inside the class, not a template parameter. One class, one implementation of BFS/Dijkstra/etc., every graph kind wired through it.
4. **Flatten the object ownership hierarchy \u2014 short paths from the entry point to the algorithm.** This is the *primary* goal of R15. In V1 the ownership chain from the pybind entry point down to the algorithm is roughly `generator entry \u2192 Args \u2192 BatchedInstances \u2192 Instance \u2192 GraphWrapper \u2192 Boost graph` (with parallel branches into `GraphTokenizer*`, `Task*`, `ScratchPad*`, five `Matrix<int>` members, and a `positions_ptr`). Debugging a wrong output means chasing five levels of `owns`. In the target design the chain is just `Worker \u2192 stage class \u2192 algorithm` \u2014 two hops. The rule: **any composition chain longer than three levels between the entry point and the code doing real work needs a strong justification.** Don't create intermediate owning classes purely to aggregate other classes; either promote members up to a flatter owner, or drop the aggregation entirely.

### Class-by-class notes

#### `GeneratorConfig` — one flat struct at the pybind boundary

**Replaces:** `Args`, `TaskArgs`, `ShortestPathTaskArgs`, `BFSTaskArgs`, `CenterCentroidTaskArgs`, `KhopsArgs`, `ScratchpadArgs`, `BFSScratchpadArgs`, `TokenizationArgs`, `PosArgs` (**10 classes with 3 inheritance chains → 1 struct**).

```cpp
struct GeneratorConfig {
    // Shared fields
    int      min_num_nodes;
    int      max_num_nodes;
    int      batch_size    = 256;
    int      min_vocab, max_vocab;

    // Kind selectors (dispatch enums, from R11)
    GraphKind      graph_kind;
    TaskKind       task_kind;
    ScratchpadKind scratchpad_kind = ScratchpadKind::None;

    // Kind-specific parameters (std::variant, from R11)
    TaskParams        task_params;         // std::variant<BfsParams, ShortestPathParams, ...>
    ScratchpadParams  scratchpad_params;   // std::variant<NoneScratchpadParams, BfsScratchpadParams, ...>

    // Tokenization / positional-encoding fields (all shared)
    bool   is_causal        = true;
    bool   include_nodes_in_graph_tokenization = false;
    int    distance_bound   = -1;
    // ... other flat fields

    // One constructor: parse+validate from py::kwargs. Fails fast on unknown keys.
    explicit GeneratorConfig(const py::kwargs& kw);
};
```

**Exposed to Python via pybind class + auto-generated `.pyi`** (R1 Option A + `pybind11-stubgen`). Adding a field is one line in the struct + one `.def_readwrite` line + rebuild → `.pyi` regenerates → IDE hints update.

#### `TaskParams` — `std::variant` of per-task POD structs

```cpp
struct BfsParams          { int distance_bound; };
struct ShortestPathParams { int distance_bound; bool weighted; };
struct CenterParams       { float p; };
struct KhopsParams        { int k; int max_num_hops; };
using TaskParams = std::variant<BfsParams, ShortestPathParams, CenterParams, KhopsParams>;
```

Each alternative carries *exactly* its own fields — you can't have `khops_k` set when `task_kind == BFS`. Dispatch via `std::visit`. Same pattern for `ScratchpadParams`.

#### `SharedContext` — const, one instance, shared across all worker threads

**Replaces:** all module-level globals (`seed_`, `gen`, `dictionary`, `pos_dictionary`, `sample_int_partition`, `validation_hashes`, `test_hashes`).

```cpp
class SharedContext {
public:
    // Populates all fields once, on the main thread, before any Worker is spawned.
    // sample_int_partition is fully pre-computed up to config.max_num_nodes (R10 wrinkle).
    SharedContext(GeneratorConfig cfg,
                  Dictionary dict, Dictionary pos_dict,
                  HashSet validation_hashes, HashSet test_hashes);

    // All accessors return const references. NO mutation methods after construction.
    const GeneratorConfig& config()    const noexcept { return cfg_; }
    const Dictionary&      dict()      const noexcept { return dict_; }
    const Dictionary&      pos_dict()  const noexcept { return pos_dict_; }
    const HashSet&         val_hashes()  const noexcept { return validation_hashes_; }
    const HashSet&         test_hashes() const noexcept { return test_hashes_; }
    const SampleIntPartition& int_partition() const noexcept { return int_partition_cache_; }

private:
    GeneratorConfig     cfg_;
    Dictionary          dict_;
    Dictionary          pos_dict_;
    HashSet             validation_hashes_;
    HashSet             test_hashes_;
    SampleIntPartition  int_partition_cache_;   // pre-warmed, const after construction
};
```

- Owned by a `std::shared_ptr<const SharedContext>` on the Python side.
- Immutable after construction → safe to share `const` across any number of threads with **zero synchronization**.

#### `Worker` — one per Python thread, owns RNG + pipeline stage classes

**Replaces:** the `thread_local gen` hack; also serves as the orchestrator across the three pipeline stages (per I10-Followup; formerly four, until `ScratchpadBuilder` was merged into `TaskComputer`).

```cpp
class Worker {
public:
    Worker(std::shared_ptr<const SharedContext> ctx, uint64_t seed);

    // The single entry point for the whole pipeline.
    py::dict generate_batch(GraphKind kind, const GeneratorConfig& cfg);

    // Separate entry point for khops per R8 (different data shape, own pipeline).
    py::dict generate_khops_batch(const GeneratorConfig& cfg);

    uint64_t seed() const noexcept { return seed_; }

private:
    // State
    std::shared_ptr<const SharedContext> ctx_;
    std::mt19937_64                       gen_;
    uint64_t                              seed_;

    // Pipeline stage classes — constructed once per Worker, reused across batches.
    // Each holds a reference to *ctx_ and (optionally) scratch buffers.
    // Three stages per I10-Followup (formerly four; scratchpad construction moved inside TaskComputer).
    GraphSampler         graph_sampler_;
    TaskComputer         task_computer_;   // produces (Task, Scratchpad)
    Tokenizer            tokenizer_;
};
```

- Each Python thread constructs its own `Worker` with a distinct seed via `derive_worker_seed(master_seed, worker_id)`.
- The `SharedContext` pointer is shared across all workers; the RNG is not; the stage classes are per-worker but small and stateless-ish (they may hold scratch buffers reused across items in a batch).
- **No locks anywhere in the hot path.**

#### `CsrGraph` — the *one* graph representation for every graph kind

**Replaces:** `GraphWrapper<D>` templated on `boost::directedS` / `boost::undirectedS`, plus every Boost `adjacency_list` in the codebase. **This is the class the user specifically called out** — it lives here once, not once per graph kind.

```cpp
class CsrGraph {
public:
    // --- Construction: the only place directedness matters. ---
    // Doubling of undirected edges happens inside from_undirected_edges;
    // callers pass the logical edge list either way.
    static CsrGraph from_directed_edges(
        int n,
        std::span<const std::pair<int,int>> edges,
        std::span<const int> weights = {});          // weights empty ⇒ unweighted

    static CsrGraph from_undirected_edges(
        int n,
        std::span<const std::pair<int,int>> edges,
        std::span<const int> weights = {});

    // --- Queries. Algorithms use these; kind of graph is irrelevant here. ---
    int  num_nodes()           const noexcept { return num_nodes_; }
    int  num_directed_edges()  const noexcept { return (int)col_indices_.size(); }
    bool is_weighted()         const noexcept { return !edge_weights_.empty(); }
    int  degree(int u)         const noexcept { return row_offsets_[u+1] - row_offsets_[u]; }
    std::span<const int> neighbours(int u) const noexcept {
        return {&col_indices_[row_offsets_[u]], (size_t)degree(u)};
    }
    std::span<const int> edge_weights_of(int u) const noexcept;  // empty if unweighted

    // --- Algorithms (single implementation, called by every graph kind) ---
    BfsResult      bfs(int source, int distance_bound = -1) const;
    DijkstraResult dijkstra(int source) const;                    // asserts is_weighted()
    // ... connected_components(), etc.

private:
    int                 num_nodes_;
    std::vector<int>    row_offsets_;   // size num_nodes + 1
    std::vector<int>    col_indices_;   // size num_directed_edges (2·E for undirected input)
    std::vector<int>    edge_weights_;  // parallel to col_indices; empty ⇒ unweighted
};
```

**Design choices worth calling out:**

- **Directedness is expressed at construction, not stored.** Two named constructors (`from_directed_edges` / `from_undirected_edges`) do the two possible things (insert each edge once vs. twice). After that, algorithms only walk `neighbours(u)` — they *never need to know* whether the CSR was built from directed or undirected input. The graph's arrays already encode the answer. No `bool directed_` field is stored, because nothing reads it after construction. If a future algorithm needs it (e.g., a transpose query), add it then.
- **Weightedness is expressed by whether `edge_weights_` is empty.** An empty `vector<int>` is 24 bytes of zeros — that's the entire cost for unweighted graphs. Algorithms that require weights (`dijkstra`) assert `is_weighted()`; ones that don't (`bfs`) ignore the field entirely.
- **`std::span` for zero-copy views.** `neighbours(u)` returns a view into `col_indices_` without copying — a pointer and a length, 16 bytes on the stack.
- **All algorithms** (BFS, Dijkstra, connected-components, Floyd–Warshall) are members of `CsrGraph` (or free-standing functions in `csr_graph.h` — either works). Written once. Same code works whether the graph came from Erdős–Rényi, path-star, Euclidean, or khops.
- Boost is not linked. Compile times drop under 5s (from ~45s).

**Why not template on `Directed` and/or `Weighted`?** This is worth a paragraph because the templating instinct is strong here (Boost does it) and picking wrong would recreate exactly the V1 pain.

Templating on `Directed` gives you `CsrGraph<true>` and `CsrGraph<false>`. That template parameter now propagates through `SampledGraph<Directed>`, every algorithm signature (`bfs<Directed>(...)`), and — critically — through `Worker::generate_batch<Directed>(...)`, which is the pybind entry point. To bind it to Python you'd need four instantiations if you also template on `Weighted`, plus a Python-side dispatcher that maps `(directed, weighted)` strings to the right instantiation. **This is precisely the pattern of the current V1 code** (`GraphWrapper<D>` propagating through `Instance<D>` and `BatchedInstances<D>`), and it's the root cause of the `undirected_graphs.h` / `directed_graphs.h` split plus the whole `convert_undirected_to_directed` function.

Templates buy you compile-time type safety, which matters when the choice between variants determines which *algorithm* you need to call. Here, it doesn't — BFS/Dijkstra don't care about directedness (they walk outgoing edges either way), and weightedness is a per-algorithm precondition that's easier to express as `assert(is_weighted())` than as a template constraint. Zero-cost runtime dispatch (a `switch` in `Worker::generate_batch` on `GraphKind`) is the right tool here; template propagation isn't.

Templates would be right if this were a generic graph library where third parties bring their own graph representations. It isn't — you own every algorithm and every call site.

#### `SampledGraph` — CSR + auxiliary generation-time data

`CsrGraph` is the pure graph. Generation produces additional data (node labels, shuffle maps, Euclidean positions, etc.) that lives with the graph but isn't part of it. That's `SampledGraph`:

```cpp
struct SampledGraph {
    CsrGraph                          graph;              // the shared CSR representation
    std::vector<int>                  node_list;          // labels assigned to nodes (dictionary IDs)
    std::vector<int>                  node_shuffle_map;   // for randomized node ordering
    std::optional<std::vector<std::array<float,2>>> positions;  // Euclidean only; nullopt otherwise
};
```

- Same design principle as `TaskParams`: use `std::optional` for fields only some graph kinds populate.
- Algorithms take `const CsrGraph&` (they don't need `node_list` or positions). Tokenization takes `const SampledGraph&`.

#### `Task` and `Scratchpad` — plain data, tagged by kind

**Replaces:** V1's `Task`, `ShortestPathTask`, `BFSTask`, `CenterTask`, `KHopsTask`, `KHopsGenTask` (6 classes) and `ScratchPad`, `BFSScratchPad`, `DFSScratchPad` (3 classes).

**Naming note:** the new plain-data struct reuses the name `Task` after V1's polymorphic `Task` class hierarchy is deleted. There is no coexistence period — V1 is removed wholesale in the Monitored execution plan's Step 2 cutover, so the name is available in `graphgen::` from that point on with no `::Task` (V1) sibling to disambiguate against.

**A `Task` contains both sides of the sample:**
- **Query** — source-side, the model *input*. E.g., "what is the shortest path from A to B?" — the pair `(A, B)`.
- **Targets** — target-side, the model *output*. E.g., the actual shortest path.

Both are computed together in `compute_task` (choosing which query to ask *is* part of running the task). Both are needed by `tokenize_into_row` (query goes in the source-side of the sequence; targets go in the target-side).

```cpp
struct Task {
    TaskKind kind;                                    // which fields are populated

    // --- Query (source-side / model input) ---
    // Small-cardinality parameters; the tokenizer formats these into the
    // source part of the sequence during the compute phase.
    struct Query {
        std::optional<int>              start_node;   // shortest_path, BFS-from-source
        std::optional<int>              end_node;     // shortest_path
        std::optional<std::vector<int>> source_set;   // multi-source BFS, center query
        std::optional<int>              k;            // khops
        // ... one std::optional per query-side field
    } query;

    // --- Targets (target-side / model output) ---
    // The correct answer(s) the model must produce. Tokenized into the
    // target part of the sequence.
    struct Target {
        std::optional<std::vector<int>> path;         // shortest_path, BFS
        std::optional<std::vector<int>> ranks;        // node-ranking tasks
        std::optional<std::vector<int>> centers;      // center task
        std::optional<std::vector<int>> distances;    // any task using a distance vector
        // ... one std::optional per target-side field
    } target;
};

struct Scratchpad {
    ScratchpadKind                    kind;
    std::optional<std::vector<int>>   trace;         // BFS / DFS trace, tokenized as thinking steps
    // ...
};
```

The `Query` / `Target` sub-struct split makes the model's I/O boundary a first-class part of the type. When you write `tokenize_into_row`, you can literally see which fields become source tokens (from `task.query`) and which become target tokens (from `task.target`).

Simpler than `std::variant` here because the fields overlap heavily across tasks (multiple tasks produce a `path` target, multiple use `start_node` in the query). If overlap ever stops being the norm — say a new task type carries a completely disjoint set of fields — upgrade the relevant sub-struct to `std::variant<...>`.

#### `BatchOutputArrays` — Tier 3E preallocation

**Replaces:** `BatchedInstances<D>::package_for_model` and the piecemeal `py::array_t` construction currently done at the end.

```cpp
class BatchOutputArrays {
public:
    // Constructor allocates all output tensors up front, one per output field, sized
    // for the whole batch. Uses per-batch shapes from the precompute phase's Layout,
    // with config.max_seq_len etc. as hard upper bounds (Tier 3E approach D).
    explicit BatchOutputArrays(const GeneratorConfig& cfg);

    // Returns a "row view" — writable slices of every array for row i. The Tokenizer
    // stage class writes into these directly, no intermediate allocation.
    RowView row(int i);

    py::dict to_pydict() &&;   // move-out; caller returns this from generate_batch
private:
    py::array_t<int>   tokenized_inputs_;
    py::array_t<int>   tokenized_targets_;
    // ... other output arrays
};
```

- Owns the NumPy buffers. Every batch allocates once, at the top of `Worker::generate_batch`.
- `RowView` is a tiny non-owning aggregate of `std::mdspan` views — one 2D view per output field, sliced to the row-`i` region of the corresponding numpy buffer. Per-cell writes read as `row.tokenized_inputs(seq_pos, dim) = tok;` rather than raw offset arithmetic. Zero heap allocation per pipeline-stage call, identical codegen to hand-rolled index math (mdspan's `LayoutRight` is constexpr stride-2 access), and constructed directly from `py::array_t<int>::mutable_unchecked<2>()`. See R12's `std::mdspan` bullet for toolchain notes.

#### Pipeline stage classes

The algorithms live here. One class per stage; each class has a single public method (dispatched internally by `switch` on the kind enum) plus private per-kind methods. No inheritance, no virtual methods, no shared base class.

```cpp
class GraphSampler {
public:
    explicit GraphSampler(const SharedContext& ctx) : ctx_(ctx) {}

    SampledGraph sample(GraphKind kind, std::mt19937_64& rng,
                        const GeneratorConfig& cfg);

private:
    SampledGraph sample_erdos_renyi(std::mt19937_64& rng, const GeneratorConfig& cfg);
    SampledGraph sample_path_star   (std::mt19937_64& rng, const GeneratorConfig& cfg);
    SampledGraph sample_balanced    (std::mt19937_64& rng, const GeneratorConfig& cfg);
    SampledGraph sample_euclidean   (std::mt19937_64& rng, const GeneratorConfig& cfg);
    SampledGraph sample_random_tree (std::mt19937_64& rng, const GeneratorConfig& cfg);
    // ... one private method per GraphKind

    const SharedContext& ctx_;                     // dictionaries, etc.
    std::vector<std::pair<int,int>> edge_scratch_; // reused across items in a batch
};

class TaskComputer {
public:
    TaskComputer() = default;

    Task compute(TaskKind kind, const CsrGraph& g, std::mt19937_64& rng,
                 const GeneratorConfig& cfg);

private:
    Task compute_bfs           (const CsrGraph& g, std::mt19937_64& rng, const GeneratorConfig& cfg);
    Task compute_shortest_path (const CsrGraph& g, std::mt19937_64& rng, const GeneratorConfig& cfg);
    Task compute_center        (const CsrGraph& g, std::mt19937_64& rng, const GeneratorConfig& cfg);
    Task compute_khops         (const CsrGraph& g, std::mt19937_64& rng, const GeneratorConfig& cfg);
    // ... one private method per TaskKind

    // Scratch buffers reused across items in a batch (real perf win vs free functions).
    std::vector<int> distances_scratch_;
    std::vector<int> parents_scratch_;
};

// SUPERSEDED BY I10-Followup: no standalone ScratchpadBuilder stage class.
// The content below is retained as reference for the *methods themselves* (their signatures
// and per-kind dispatch), but they now live as *private members of TaskComputer*, not as their
// own class. TaskComputer::run(TaskKind, ScratchpadKind, ...) returns (Task, Scratchpad) in one call.
// See the I10-Followup section under IMPROVEMENTS for the rationale.
class ScratchpadBuilder {
public:
    ScratchpadBuilder() = default;

    Scratchpad build(ScratchpadKind kind, const CsrGraph& g, const Task& task,
                     std::mt19937_64& rng, const GeneratorConfig& cfg);

private:
    Scratchpad build_none(...);
    Scratchpad build_bfs (...);
    Scratchpad build_dfs (...);
};

class Tokenizer {
public:
    explicit Tokenizer(const SharedContext& ctx) : ctx_(ctx) {}

    // Compute phase: pure data placement into a pre-allocated RowView.
    // GIL-released and safe to parallelize.
    void tokenize_into_row(const SampledGraph& g, const Task& task,
                           const Scratchpad& sp,   const Layout& layout,
                           const GeneratorConfig& cfg, RowView out);

private:
    // Per-task private helpers, all internal to the compute phase.
    void write_shortest_path(const Task& task, const Layout& layout, RowView out);
    void write_bfs           (const Task& task, const Layout& layout, RowView out);
    // ...

    const SharedContext& ctx_;  // dictionary lookups happen here
};
```

Adding a new task/graph/scratchpad kind:

1. Add enum entry.
2. Add case in the relevant stage class's public method.
3. Add per-kind private method on that stage class.
4. Add per-kind params to the variant if needed.

Compiler flags every switch you didn't update (with `-Wswitch-enum` / `-Werror=switch`), so silent runtime fallthroughs like today's `if (task_type == "shortest_path")` chain become compile errors.

Why no `IPipelineStage` base class? Because there's nothing to share — the stage classes have different signatures (`sample` takes no graph, `compute` takes a graph, `tokenize_into_row` writes to a row view). A base class would force artificial signature unification via `std::variant` inputs or `void*`, buying nothing over a plain method call. This is the case R2 was originally aimed at; R15 replaces it because there is no useful abstraction to extract.

### File layout

```
include/graphgen/
├── config.h              // GeneratorConfig, kinds, param variants
├── shared_context.h      // SharedContext, Dictionary, HashSet
├── worker.h              // Worker
├── csr_graph.h           // CsrGraph + neighbours() view + BFS/Dijkstra methods
├── sampled_graph.h       // SampledGraph
├── task.h                // Task (with Query/Target sub-classes) + Scratchpad + Layout
├── graph_sampler.h       // GraphSampler class
├── task_computer.h       // TaskComputer class (produces both Task and Scratchpad per I10-Followup)
├── tokenizer.h           // Tokenizer class + RowView
├── batch_output.h        // BatchOutputArrays
└── check.h               // GG_CHECK / GG_ASSERT (R6)

src/
├── graph_sampler.cpp     // dispatch + all sample_<kind> private methods
├── task_computer.cpp     // dispatch + all compute_<kind> private methods + all build_<sp_kind>_scratchpad private methods (per I10-Followup)
├── tokenizer.cpp
├── worker.cpp
├── shared_context.cpp
└── bindings.cpp          // pybind11 module entry, py::class_<>, m.def(...)

tests/
├── reference/            // slow reference implementations (R7)
└── fast_vs_reference/    // doctest tests comparing outputs
```

- Each stage class gets its own `.cpp` → parallel compilation, small recompiles per change.
- **Trade-off vs the free-function alternative:** one file per stage class instead of one file per algorithm. Adding a new task means editing `task_computer.h` + `task_computer.cpp` rather than creating a new file — slightly less isolation between algorithms, but keeps related code together (the dispatcher and every per-kind method are all one grep away).
- No headers include Boost. Headers include only `<vector>`, `<optional>`, `<variant>`, `<span>`, `<memory>`, `<random>`, and pybind11 (in `bindings.cpp` and the header that declares pybind types).

### Class-count summary (V1 → post-refactor)

| Concern | V1 count | Post-refactor count | Delta |
|---|---:|---:|---:|
| Argument classes (with inheritance) | 10 | 1 struct + a few POD variant alternatives | **−10** |
| Task classes (with inheritance) | 6 | 1 `Task` data class + 1 `TaskComputer` stage class | **−4** |
| Scratchpad classes (with inheritance) | 3 | 1 `Scratchpad` data class (construction lives inside `TaskComputer` per I10-Followup; no separate stage class) | **−2** |
| Graph classes | 1 (`GraphWrapper<D>` templated on directedness) | 1 (`CsrGraph`, non-templated) + 1 (`GraphSampler` stage class) | +1 (but flat) |
| Instance / batching | 2 (`Instance<D>`, `BatchedInstances<D>`) | 1 (`BatchOutputArrays`) | **−1** |
| Tokenization | 1 (`GraphTokenizer`) | 1 (`Tokenizer` stage class) | 0 |
| Threading / state | 0 (globals) | 2 (`SharedContext`, `Worker`) | +2 |
| **Total** | **~23 with heavy inheritance** | **~13 classes, all flat** | **~40% reduction in count; 100% reduction in inheritance** |

**The important number is the second one.** Class count drops modestly (23 → ~13) because we're keeping OO structure. What drops dramatically:

- **Ownership depth: from ~5 (`generator entry → Args → BatchedInstances → Instance → GraphWrapper → Boost graph`, with parallel branches into `Task*`/`ScratchPad*`/`GraphTokenizer*`) to 2 (`Worker → stage class → algorithm`).** This is the number that actually determines how many layers of indirection you thread through when debugging a wrong output.
- **Inheritance depth: from 2 (`BFSTaskArgs → ShortestPathTaskArgs → TaskArgs`) to 0** — a side effect of moving closed variation from virtual dispatch to enums.
- **Virtual-method count: from ~15 to 0.**

Every class in the new design is concrete, non-templated (except `std::variant` alternatives), and can be understood in isolation.

### What stays templated, what doesn't

- **`Worker`, `SharedContext`, `CsrGraph`, `SampledGraph`, `Task`, `Scratchpad`, `GeneratorConfig`, `GraphSampler`, `TaskComputer`, `Tokenizer`** (three stage classes per I10-Followup; `ScratchpadBuilder` merged into `TaskComputer`): not templated. Compile once.
- **`std::variant` visitors** inside each stage class: template on the visitor callable, as usual for `std::visit`. Kept internal to each stage class's `.cpp`.
- **BFS / Dijkstra / etc.**: not templated on graph type — they take `const CsrGraph&` (they can be either methods on `CsrGraph` itself or private members of `TaskComputer`; leaning toward the former for reusability). Written once. Zero template noise in headers.

The current `template<typename D>` propagation across `Instance<D>`, `GraphWrapper<D>`, `BatchedInstances<D>` disappears entirely.

## Suggested refactor order

### Key insight: the "big three" are one integrated rewrite, not three sequential steps

**R5 Level 2** (drop Boost, use CSR), **R15** (flatten the object hierarchy — fewer ownership levels between pybind entry and algorithm), and **Tier 3E** (preallocate batch tensors and fill them in place) all rewrite the same code paths from opposite angles. Trying to sequence them means writing the same code two or three times:

- If you do R15 first over Boost graphs, you'll rewrite every algorithm again when R5 Level 2 drops Boost and changes the graph type.
- If you do Tier 3E first without R15, you're wiring `py::array_t` allocation into `Instance`/`BatchedInstances`, then throwing that wiring away when R15 deletes those classes.
- If you do R5 Level 2 first without R15, you replace Boost with CSR inside the existing `GraphWrapper`/`Instance` chain — then rewrite it all again when R15 flattens.

**Tier 2A** (Johnson → BFS) has the same problem: writing BFS against Boost's graph type is throwaway work if we're about to swap Boost for CSR. Better to write BFS *once*, directly over the CSR representation.

The right approach: land the prerequisites, then do all the "big rewrite" work as a single integrated pass, one graph kind at a time, verified against the reference implementation (R7) at each step.

### Phases

**Phase 0 — Prerequisites (must exist before any rewrite).**
- **R6** — `GG_CHECK` / `GG_ASSERT` macros. Cheap; touches nothing else.
- **R7** — reference-vs-fast test infrastructure. **Absolutely required** — without this the Phase 2 rewrite cannot be verified against the V1-submission behaviour. Use `git checkout v1.0-submission` output as the reference.
- **R13** — tooling. Wire in doctest, clang-format, clang-tidy, sanitizers, migrate to CMake. Compounds across every future edit.

**Phase 1 — Add new types alongside the old ones (no algorithm rewrites yet).**
Everything in this phase is additive. Old code paths keep working during the transition; new types get built up in isolation.
- **R1** — `GeneratorConfig` struct (Option A: pybind class + auto `.pyi`). Kills the `TaskArgs`/`ScratchpadArgs` inheritance chain immediately. Can start being used at the pybind boundary before anything else changes.
- **R10** — `SharedContext` + `Worker` skeleton. Move the module-level globals into `SharedContext`; every existing top-level function initially just delegates to a "default worker".
- **R11** — `TaskKind` / `GraphKind` / `ScratchpadKind` enums + `std::variant` for task-specific params. Just type definitions; no dispatch changes yet.
- **R12** — modern C++ hygiene, opportunistically applied while touching files for R1/R10/R11 (drop `using namespace std;`, wrap in `namespace graphgen`, `[[nodiscard]]`, etc.).

**Phase 2 — The integrated rewrite (this is where the big wins are).**

**Do the shared infrastructure ONCE, up front, before any per-kind porting:**

- **`CsrGraph`** (R5 Level 2): the single graph representation used by every graph kind and every algorithm. Written once, in `csr_graph.h`. See the "Proposed post-refactor class hierarchy" section above.
- **Algorithms on `CsrGraph`** (Tier 2A, plus I3): `bfs`, `distance_bounded_bfs`, `dijkstra`, `distance_bounded_dijkstra` (if needed), `all_pairs_distances`, `distance_bounded_all_pairs_distances`, `connected_components`, etc. Each ~30–40 lines. Written once, called by every graph kind's `TaskComputer::compute_<kind>` private method.
- **`SampledGraph`, `Task`, `Scratchpad`, `Layout`, `BatchOutputArrays`**: flat data-class definitions (see R15's class hierarchy section). Written once.
- **Pipeline stage classes** (`GraphSampler`, `TaskComputer`, `Tokenizer` — three stages per I10-Followup): declared once with empty `switch` bodies in their public methods. Per-kind private methods and switch cases added as each graph kind is ported. `TaskComputer` gets both `compute_<task_kind>` and `build_<scratchpad_kind>_scratchpad` private method families.
- **`Worker`** skeleton: holds instances of the three stage classes; `generate_batch` allocates `BatchOutputArrays`, releases GIL, loops, dispatches to the stage classes.

**Then port one graph kind at a time.** Erdős–Rényi first (simplest). V1 has already been deleted at Phase 2's opening cutover (see the Monitored execution plan's Step 2), so each per-kind step is *purely additive to the new pipeline* — there is no "keep the old alive" toggle. Verification uses invariant / analytic tests per kind; if numeric-parity spot-checks against V1 are wanted, run V1 manually from a `git worktree add ../graphgen-v1 v1.0-submission` sibling directory.

For each `GraphKind` (Erdős–Rényi → path-star → balanced → Euclidean → random-tree → khops):

  1. **Write `sample_<kind>`** — the corresponding private method on `GraphSampler`, producing a `SampledGraph` (which contains the *shared* `CsrGraph` type — not a per-kind graph class). Uses `CsrGraph::from_directed_edges` or `CsrGraph::from_undirected_edges` to convert its accumulated edge list into CSR.
  2. **Add cases** to the pipeline dispatchers for this kind's `TaskKind` / `ScratchpadKind` combinations. The task algorithms themselves (BFS etc.) are already written and work uniformly across all graph kinds — no per-kind algorithm code.
  3. **Tokenization case** in `tokenize_into_row` — writes directly into the pre-allocated `RowView` for this row (Tier 3E). No intermediate allocation.
  4. **Verify** with invariant tests (path validity, distance monotonicity, hardness bounds) and analytic ground truth where available (e.g., Euclidean grids in Section 2C have Manhattan-distance ground truth). Optional manual `git worktree` spot-check against `v1.0-submission` if a numeric-parity sanity check is wanted.
  5. **R8**: if this graph kind is khops, expose it as `Worker::generate_khops_batch` rather than adding to the regular `GraphKind` enum (different data shape, own pipeline).

By the end of Phase 2, `Args`, `Instance`, `BatchedInstances`, V1's polymorphic `Task`, `ScratchPad`, `GraphWrapper`, and every Boost include are already gone (they were deleted at the Phase 2 opening cutover). `Tier 4A` (GIL release) is already in place because `Worker::generate_batch`'s skeleton included it from the start. `R3` (phase separation) is automatic (each stage class's public method is a phase, and Tier 3E Approach D enforces a hard precompute/compute boundary at the `Worker::generate_batch` level).

**Phase 3 — Multi-threading.**
- **Tier 4B** — parallel batch fill across worker threads, using the per-thread `Worker` from R10 and the GIL release from Phase 2. Each worker owns its own RNG; the `SharedContext` is `const`-shared. Deterministic per-worker seeds via `derive_worker_seed(master_seed, worker_id)`.

**Phase 4 — Opportunistic tuning (ongoing).**
Order by profiler evidence, not by list position. These are independent, low-priority, and safe to skip until profiling says otherwise:
- **Tier 3A** — flat matrices for hot buffers.
- **Tier 3B** — `hash_setS` vs `vecS` for adjacency (Boost-specific; only relevant if any Boost path survives, which it shouldn't after Phase 2).
- **Tier 3C** — `unordered_map` → `flat_hash_map` where profiling shows a win.
- **Tier 2C** — Erdős–Rényi via union-find.
- **Tier 2D** — squared Euclidean distances.
- **R4** — header split + PCH. Do when compile times start hurting; less relevant if Phase 2 dropped Boost, which is the dominant source of compile time.
- **R9** — documented invariants. Apply as a pass at the end of each Phase 2 iteration.

### What's not in the order

- **R2, R3** — superseded by R15; skip. Their content is absorbed into Phase 2.
- **R14** — considered and rejected; keep the pipeline in C++ (see R14 for reasoning).
- **Tier 2B** — `floyd_warshall_frydenlund` is not used in practice; deferred indefinitely.

---

## IMPROVEMENTS — opportunities the refactor unlocks

The refactor isn't just a rewrite of the current behaviour — it opens the door to a few capabilities that are hard or impossible in V1's shape. Recording them here so they don't get lost.

### I1. Every graph kind should support directed *and* undirected

**Current state.** V1 has each graph kind hard-wired to one directedness:

- `erdos_renyi_generator`, `euclidean_generator`, `random_tree_generator` → undirected only (declared in `undirected_graphs.h`, `boost::undirectedS`).
- `path_star_generator`, `balanced_generator` → directed only (declared in `directed_graphs.h`, `boost::directedS`).

There is a whole `convert_undirected_to_directed` function precisely because the directedness lives in the *type* rather than in the *data*.

**What the refactor unlocks.** Once `CsrGraph` is a single non-templated type with `from_directed_edges` / `from_undirected_edges` as separate constructors (see the `CsrGraph` section under "Proposed post-refactor class hierarchy"), *every* `sample_<kind>` method on `GraphSampler` can produce either variant with a single line change at the CSR construction site. The algorithms downstream (BFS, Dijkstra, ...) don't care. So the API grows from "5 kinds" to "5 kinds × 2 directednesses = 10 kinds" for free.

Concretely:

- Add a `bool directed` (or `Directedness`) field to `GeneratorConfig`.
- Each `GraphSampler::sample_<kind>` reads `cfg.directed` and picks the corresponding CSR constructor:
  ```cpp
  return cfg.directed
      ? CsrGraph::from_directed_edges(n, edges, weights)
      : CsrGraph::from_undirected_edges(n, edges, weights);
  ```
- `convert_undirected_to_directed` is deleted — the caller just asks for whichever they want at construction.

**Rejection cases.** A few kinds have a directedness that's semantically fixed (a *path*-star graph is inherently directed — pointing away from the center). For those, the sampler ignores `cfg.directed` (or asserts it matches), documented per kind. But the *default* stance flips from "one directedness per kind" to "both, unless the kind requires otherwise."

### I2. Decouple internal directedness from tokenized directedness

**The insight.** From the *language model's* point of view, what makes a graph "directed" isn't the presence or absence of `(v, u)` tokens in the input sequence — it's the **asymmetry of the distance ground truths**. If shortest-path from `A` to `B` is 3 hops but from `B` to `A` is unreachable, that's the signal the model has to learn to represent as a directed relationship. Whether the tokenized edge list happens to include the reverse edges is a separate question.

**What this enables.** The graph representation used for algorithms (which needs to be fast, cache-friendly, and complete) can be **decoupled** from the graph representation used for tokenization (which is a choice about what the model sees). Specifically:

- Build the CSR as *directed* internally — even for logically-undirected graphs, we can store each edge as two directed edges (which is what undirected CSR already does anyway).
- Compute distances / ground truths against this directed CSR. For an undirected input, forward and reverse distances are equal by construction. For a directed input, they can differ — and *that asymmetry* is what carries the directedness signal to the model.
- At **tokenization time**, decide what the model sees:
  - **`emit_forward_and_reverse_edges=true`** → tokenize all directed edges (both `(u, v)` and `(v, u)` for undirected inputs; only `(u, v)` for one-way directed inputs).
  - **`emit_forward_and_reverse_edges=false`** → emit each undirected edge once, or (for genuinely directed inputs) only emit forward edges. The model has to *learn* that the missing reverse edges either exist (undirected case) or don't (directed case) from the distance-target asymmetry.

The second option is the more research-interesting one: it lets you train a model that sees a canonical edge listing and has to infer edge symmetry from the answer targets. That's a genuinely new axis of experiment that V1 can't express because "directedness" is baked into the graph type from the beginning.

**Where this lives in the design.**

- Add a tokenization-time flag to `GeneratorConfig` — something like `TokenizeEdgesAs { AllDirected, ForwardOnly, CanonicalUndirected }`. Cardinality 3 enum, not templated.
- Add the branch inside `Tokenizer::write_edge_list` (private method on the `Tokenizer` stage class, called from `tokenize_into_row`).
- No changes needed to `CsrGraph`, algorithms, or `GraphSampler`. This is a tokenization-layer decision, kept entirely inside `Tokenizer`.

**Why this is only expressible after the refactor.** In V1, the graph type (`boost::directedS` vs `boost::undirectedS`) determines both the algorithm behaviour and the tokenization behaviour — they're the same code path. Separating "what the CSR looks like" from "what the model sees" requires a graph representation that carries structural information independently of a fixed directedness label. That's exactly what the non-templated `CsrGraph` provides: the graph *is* the CSR arrays; interpretation ("emit both directions?") is a downstream choice made per-tokenization.

### I3. Distance-bounded distance-matrix computation

**Current state.** V1 always computes full all-pairs distances via `floyd_warshall_all_pairs_shortest_paths` (O(N³)) or `johnson_all_pairs_shortest_paths` (O(N·E·log N + N²·log N)). Both compute *every* pair of distances, even when the task only needs distances up to some small bound. `GeneratorConfig::distance_bound` already exists in V1 as a field, but it's used only at tokenization time — the underlying distance computation ignores it and produces the full N×N matrix regardless.

**What we want.** A *distance-bounded* variant of each distance primitive. When called, the algorithm returns as soon as every pair's distance is known to be either (a) ≤ `max_distance_bound`, or (b) > `max_distance_bound`. Distances exceeding the bound are stored as a sentinel (V1's existing "unreachable" marker, `-1` / `INT_MAX`) rather than their true value.

**API shape — separate distance-bounded functions, matching parameter name.**

Rather than overloading the existing primitives with a `max_distance_bound = -1` default, expose the bounded variants as their own named functions. Call sites read more explicitly (`distance_bounded_bfs(source, 4)` vs `bfs(source, 4)` — the former is unambiguous about intent), and each variant can pick the algorithm that's actually best for its case without runtime branching on the parameter.

```cpp
class CsrGraph {
    // ---- Unbounded (full APSP / full BFS): ----
    BfsResult      bfs(int source) const;
    DistanceMatrix all_pairs_distances() const;   // full APSP; picks best algorithm for the case

    // ---- Distance-bounded variants: ----
    BfsResult      distance_bounded_bfs(int source, int max_distance_bound) const;
    DistanceMatrix distance_bounded_all_pairs_distances(int max_distance_bound) const;
    //   Entries with true distance > max_distance_bound are stored as UNREACHABLE.
    //   Preferred implementation: repeated single-source BFS with early exit when
    //   the frontier is empty at depth max_distance_bound + 1.
};
```

`DistanceMatrix` is a `std::vector<int>` + `std::mdspan<int, std::extents<int, std::dynamic_extent, std::dynamic_extent>>` pair (per R12's `std::mdspan` recommendation), so `distances(u, v)` is the access syntax at every call site — no `distances[u * N + v]` arithmetic anywhere. Same pattern for the per-source arrays that I9-Clarification caches in `SharedContext` (`d(src, node)`, `dag_out_deg(src, node)`, `N_v(src, node)`).

Parameter name is `max_distance_bound` throughout — matching V1's existing `GeneratorConfig::distance_bound` terminology so the concept has one name across the whole codebase.

**Semantic note: `max_distance_bound` is a hop count, not a path weight.** On weighted graphs this distinction is load-bearing. There are two things "distance-bounded" could mean:

1. **Hop-bounded** — only consider paths that traverse ≤ `max_distance_bound` *edges*. Independent of edge weights.
2. **Weight-bounded** — only consider paths whose *sum of edge weights* is ≤ `max_distance_bound`.

**We mean (1).** `max_distance_bound` counts *connections* (edges traversed), not accumulated path length. On an unweighted graph the two definitions coincide (every edge has weight 1). On a weighted graph they diverge: a 2-hop path with weights `10 + 10` = 20 is *within* a hop bound of 3 but *outside* a weight bound of 5.

Concrete consequences:

- **`distance_bounded_bfs`**: unambiguous — BFS enumerates paths by hop depth, so "stop expanding after depth `max_distance_bound`" is a natural early exit. Weights are ignored (BFS doesn't use them).
- **`distance_bounded_all_pairs_distances`**: the returned matrix stores hop counts (unweighted graph) or shortest-path weights among *paths of ≤ `max_distance_bound` hops* (weighted graph). Pairs not reachable within that hop count → sentinel.
- **`distance_bounded_dijkstra`** (if we add it): still hop-bounded — implemented as Dijkstra where each popped node also carries its hop depth, and expansion stops at depth `max_distance_bound`. The returned distances are still weight-sums (that's what Dijkstra returns), but only over paths that fit the hop budget.

If we ever need weight-bounded semantics — e.g., "give me all nodes within total path weight ≤ K" — that's a separate primitive (`weight_bounded_*`) with its own name, not an overloading of `max_distance_bound`. Keeping the two concepts under different names prevents the class of bugs where a caller assumes one meaning and gets the other.

**Why this matters for the language-model side.** The bound controls **how many hops the downstream language model is required to reason over** — either as an inference-time capacity (the model must resolve reachability / shortest paths of at most that hop count) or as a supervision-time cap (targets only exist for pairs within that hop count; farther pairs are collapsed to the "unreachable / out of scope" sentinel). Both framings are about *hop count*, because the reasoning depth we're studying is measured in graph steps. Weight-bounding would answer a different research question — "how much path cost can the model tolerate?" — which isn't what any current V1 task or downstream experiment is about.

**Why this is worth doing.**

- **Correctness.** Several tasks in V1 already only care about distances up to a bound (BFS-with-distance-bound, k-hop tasks, tokenization schemes that clamp distance values into a small vocabulary). Computing more than that is *wasted work* whose only purpose is to be thrown away at tokenization.
- **Speed.** For sparse graphs, distance-bounded APSP is dramatically cheaper than unbounded:
  - **Repeated BFS with early termination**: O(N · (N + E)) unbounded, but O(N · (N_b + E_b)) distance-bounded — where `N_b` and `E_b` are only the nodes and edges reachable within `max_distance_bound` hops from any given source. For small bounds this is often near-linear in `N` on sparse graphs.
  - **Unweighted graphs**: repeated BFS from each source with an early exit when the frontier is empty at depth `max_distance_bound + 1` beats Floyd–Warshall by a large factor for any bound `< N`.
  - **Weighted graphs**: distance-bounded Dijkstra (stop popping when the top of the heap has key > `max_distance_bound`) is straightforward.
- **Tokenization vocabulary.** Distances above `max_distance_bound` all get the same "far" or "unreachable" token anyway. Storing their exact values in the distance matrix is dead information — it costs memory to hold and compute to produce.

**Implementation notes.**

- **Preferred primitive is repeated single-source BFS (unweighted) or Dijkstra (weighted).** Floyd–Warshall doesn't parallelize the "stop early" idea cleanly and is O(N³) regardless of bound.
- **Sentinel convention.** Reuse V1's existing convention (`-1` for unreachable) and extend it: `-1` means "either unreachable or > `max_distance_bound`". If a caller ever needs to distinguish the two, add a separate query; nothing in the current pipeline does.
- **Non-breaking.** The unbounded versions (`bfs`, `all_pairs_distances`) keep their exact V1 semantics; new call sites that opt into a bound reach for the `distance_bounded_*` variants explicitly.
- **Interaction with `Task`/`Layout`.** The task specifies its `max_distance_bound` in its `Query` fields (see the `Task` section). `TaskComputer::compute_<kind>` reads it and calls `graph.distance_bounded_all_pairs_distances(max_distance_bound)` (or the BFS variant). Layout allocation is unaffected — the distance matrix is still N×N; only the *values* are bounded.
- **Plumb through the existing `distance_bound`.** V1's `GeneratorConfig::distance_bound` is currently a tokenization-only knob; extend its meaning to also drive `max_distance_bound` in the distance computation. Every generated batch that already declared a bound gets the speedup automatically (the task's `compute_<kind>` reads `cfg.distance_bound` and dispatches to the bounded variant when it's set).
- **Naming keeps a single vocabulary.** `distance_bound` (config field), `max_distance_bound` (algorithm parameter), `distance_bounded_*` (function names) — all three use the same "distance-bounded" concept name, so a reader who understands one understands all three.

**Not free.** Adding the bounded variants requires re-implementing distance computation from scratch (which R5 Level 2 requires anyway — Boost is going away). The cost is one BFS + one distance-bounded-Dijkstra implementation, both ~30 lines. The saving compounds across every batch generated for a distance-bounded task, which is most of them.

### I4. Complete the graph center / centroid tasks (unfinished in V1)

**Current state.** V1 has a `CenterTask` class fully written in `tasks.h` (sampling logic, tokenization, everything) — but the entire class body is wrapped in `/* ... */`. The dispatcher branch in `instance.h` recognizes `task_type == "center"` and `task_type == "centroid"` but its body is comments only. So V1 *accepts* the argument names and silently does nothing. Not a bug per se; just an incomplete feature.

**What the task is** (from the commented-out V1 implementation, matching the docstring in the tokenization helper):

Given a graph `G` and a *query set* `Q ⊆ V` of nodes, compute the node(s) that are "most central" relative to `Q`:

- **Center of `Q`**: node `v` minimizing `max_{q ∈ Q} distance(v, q)`. The 1-center problem restricted to `Q` — minimize the *worst-case* distance to any query node.
- **Centroid of `Q`**: node `v` minimizing `Σ_{q ∈ Q} distance(v, q)`. The 1-median problem restricted to `Q` — minimize the *total* distance to the query set.

The output is the set of all nodes achieving the minimum (there may be ties). The query `Q` is either provided by the caller or sampled uniformly from `V` at task-construction time, with size in `[min_query_size, max_query_size]`.

Tokenization (per V1's sketch):
- Source-side: `/  q1 q2 ... qk  ?`  (marker, query nodes, marker).
- Target-side: `=  o1 o2 ... om  .` (marker, output nodes, marker).

**How it fits the new design.** Cleanly. The plan already anticipates this task — `TaskKind::Center` is listed in the enum sketch, `CenterParams { float p; }` appears in the `TaskParams` variant, and the `Task::Target::centers` field is already declared. All that's missing is:

1. **Enum entries.** Decide whether center and centroid are one `TaskKind::Center` with a `CenterMode { Center, Centroid }` sub-enum in `CenterParams`, or two separate `TaskKind::Center` / `TaskKind::Centroid`. **Recommendation: one `TaskKind::Center` + `CenterMode` sub-enum**, because the two variants share ~95% of the code (only the aggregation function differs — `std::max` vs `std::accumulate`); duplicating the switch case buys nothing.

   Concrete shape:
   ```cpp
   enum class CenterMode : uint8_t { Center, Centroid };

   struct CenterParams {
       CenterMode mode                = CenterMode::Center;
       int        min_query_size      = 2;
       int        max_query_size      = -1;   // -1 ⇒ N
       // If the caller wants to fix the query set, they can populate
       // Task::Query::source_set directly and skip sampling.
   };
   ```

2. **`Task::Query::source_set`.** Already listed in the `Query` sub-struct. Holds `Q` — either sampled or caller-provided.

3. **`Task::Target::centers`.** Already listed in the `Target` sub-struct. Holds the set of nodes minimizing the aggregation.

4. **`TaskComputer::compute_center` private method.** ~40 lines. Structure:
   ```cpp
   Task TaskComputer::compute_center(const CsrGraph& g, std::mt19937_64& rng,
                                     const GeneratorConfig& cfg) {
       const auto& p = std::get<CenterParams>(cfg.task_params);
       Task t; t.kind = TaskKind::Center;

       // (1) Sample Q if not provided.
       t.query.source_set = sample_query(g, rng, p);   // uniform without replacement

       // (2) For each q in Q, BFS from q to fill a row of distances.
       //     Result: |Q| × N distance matrix (much cheaper than full APSP).
       auto d = distances_from_sources(g, *t.query.source_set);

       // (3) For each candidate v, aggregate d[:, v] by max or sum.
       //     Pick nodes with minimum aggregation. Ties → all included.
       t.target.centers = argmin_aggregated(d, p.mode);
       return t;
   }
   ```
   The `distances_from_sources` helper is a wrapper around `|Q|`-many `CsrGraph::bfs` calls (or `distance_bounded_bfs` from I3 if `cfg.distance_bound` is set), so it composes naturally with the distance primitives already planned.

5. **`Tokenizer::write_center` private method.** Translates the V1 tokenization scheme (`/ q1 q2 ... ? = o1 o2 ... .`) directly. Also existed in the V1 commented-out block; port it as-is.

6. **`Layout::from` case for `TaskKind::Center`.** Query length = `|Q| + 2` markers; target length = `|outputs| + 2` markers. Straight substitution into the existing per-kind size computation.

**Why this is worth doing now (during the refactor, not later).**

- **Zero extra design work.** Every hook — enum entry, variant alternative, query/target field, tokenization slot — is *already* in the plan's Task sketch. The refactor was designed around this task existing; completing it costs less than adding a brand-new task.
- **The V1 implementation is a reference.** The commented-out `CenterTask` class has correct sampling logic, correct aggregation, correct tokenization. We port it (adapted to the new stage-class shape and the new distance primitives) rather than designing from scratch.
- **Interacts well with I3.** For small query sets (`|Q| ≪ N`), running `|Q|` distance-bounded BFSes from each `q ∈ Q` is far cheaper than a full APSP, and the distance bound flows through from `cfg.distance_bound`.
- **Fills a hole in the task catalogue.** The V1 code documents and accepts `center`/`centroid` at the config level (see `get_generator_module.py`), so users may already have configs that mention them and get silent no-ops. Making the task work eliminates a silent-failure mode.

**Not free.** Implementation cost is roughly:
- ~40 lines for `TaskComputer::compute_center` (mostly a translation of V1's `CenterTask` constructor).
- ~20 lines for `sample_query` and `argmin_aggregated` helpers.
- ~40 lines for `Tokenizer::write_center` (direct port of V1's `tokenize`).
- One `Layout::from` case.
- One `distances_from_sources` helper on `CsrGraph` (or inlined into `TaskComputer` as ~5 lines calling `bfs` in a loop).
- R7 reference-vs-fast test entries.

Total ≈ 150 lines of new code, plus tests. Nothing algorithmically hard — the primitives (BFS, distance bound) are already planned; this task just composes them.

**Naming note.** V1's `CenterCentroidTaskArgs` was one class covering both modes. The new design keeps that unification (`CenterParams` with `CenterMode`), so the R1 argument-class replacement table remains accurate — no need to update it.

### I5. Standardise Python-supplied discrete distributions with an `IntRangeSampler` class

**The pattern.** In several places the caller passes a `vector<float>` of weights from Python that the C++ side turns into a `std::discrete_distribution<int>`, then samples an integer in a range `[lo, hi]` — typically to pick a path length, a k-hop count, a branching factor, etc. Every call site does this by hand, and they all differ subtly:

| Call site | Python name | Raw type | Sampled to produce | Fallback when absent |
|---|---|---|---|---|
| `args.h` (`TaskArgs::task_sample_dist`) | `task_sample_dist` | `optional<vector<float>>` | (parsed only, stored raw) | `nullopt` |
| `args.h` (`TaskArgs::probs`) | `probs` | `optional<vector<float>>` | (parsed only, stored raw) | `nullopt` |
| `ShortestPathTask` ctor (`tasks.h` ~L217) | `task_sample_dist` | `optional<vector<float>>` | `discrete_distribution<int>` over path lengths, offset by `min_path_length` | build `vector<float>(span, 1.0f)` → uniform `discrete_distribution` |
| `BFSTask` ctor (`tasks.h` ~L460) | `task_sample_dist` | `optional<vector<float>>` | Same — *copy-pasted* from `ShortestPathTask` | Same — *copy-pasted* uniform fallback |
| `make_khops` (`graph_wrapper.h` ~L161) | `task_sample_dist` | `optional<vector<float>>` | `discrete_distribution<int>` over k, offset by `min_khops` | `uniform_int_distribution<int>(min_khops, max_khops)` — **different fallback shape than the tasks above** |
| `random_tree_generator` (`undirected_graphs.h` ~L420) | `probs` | `optional<vector<float>>` | `discrete_distribution<int>` over branching factor, offset by `+1` | `binomial_distribution(d, bernoulli_p)` (or fixed `d`) — **entirely different fallback**, not a uniform equivalent |

Historical reference (already dead code but shows the same pattern with yet another normalisation policy):
- `old_code/generator.cpp` L60–L69: global `set_task_sample_dist` normalises weights to sum to 1.
- `old_code/generator.cpp` L1131–L1169: `random_tree_n` *throws* if `sum(probs) != 1`.

**The problems this causes.**

1. **Duplicated logic.** The "if `task_sample_dist` provided use `discrete_distribution`, else build a uniform" fallback appears verbatim in `ShortestPathTask` and `BFSTask` — copy-pasted. If we ever change the semantics (e.g. add validation, or switch to a cached distribution) we have to remember every call site.
2. **Silently inconsistent fallbacks.** The task ctors' "absent ⇒ uniform" is implemented as a `discrete_distribution` with equal weights (sampling gives values in `[0, span)`). `make_khops`'s "absent ⇒ uniform" is a `uniform_int_distribution` directly (values in `[lo, hi]`). Same-named argument (`task_sample_dist`), same conceptual meaning ("uniform if absent"), two different objects — one is a categorical distribution over bins, the other is a native integer-range distribution. In principle they produce the same result, but the discrepancy makes it look like the answer to "what does `task_sample_dist=None` mean here?" depends on which task you're in.
3. **Latent bug in `make_khops`.** `graph_wrapper.h` line 164 reads:
   ```cpp
   khops_max_k = task_sample_dist.value()[task_sample_dist.value().size() - 1] + min_khops;
   ```
   This uses the last *weight value* as `max_k`. It almost certainly should be `task_sample_dist->size() - 1 + min_khops` — the index of the last bucket, not the weight stored there. Undetected because for typical inputs the last weight happens to be a small number that looks plausible. A wrapper class with a proper `hi()` accessor makes this impossible to write.
4. **No normalisation policy.** V1 used to throw on non-normalised probs (old code). Current code silently accepts any non-negative weights because `discrete_distribution` auto-normalises. Not necessarily wrong, but the decision is invisible — no comment, no doc, no test.
5. **Repeated construction of `discrete_distribution`.** Every task construction rebuilds the distribution from the raw vector. `std::discrete_distribution`'s constructor is `O(N)` and precomputes a CDF; when the same weights are used across a whole batch we're doing that work per-item. A wrapper class can hold the distribution as a member and reuse it — small win, but free once the class exists.
6. **pybind boilerplate duplicated in `args.h`.** Every distribution field has the same `contains → is_none → cast<py::list>().empty() → cast<vector<float>>` dance. One helper cleans this up.

**The proposed class.** A single value type wrapping "sample an integer in `[lo, hi]` from optional weights". Full range-semantics spec is in the subsection below; the class sketch here matches it.

```cpp
namespace graphgen {

class IntRangeSampler {
public:
    // Named constructors — every one preserves the inclusive-inclusive [lo, hi] invariant.
    static IntRangeSampler single(int v);
        // Degenerate range [v, v]. Sampling always returns v.

    static IntRangeSampler uniform(int lo, int hi);
        // Uniform over [lo, hi]. Requires lo <= hi; asserts.

    static IntRangeSampler weighted(int lo, std::vector<float> weights);
        // hi = lo + weights.size() - 1. Requires weights non-empty,
        // all(w >= 0), sum(w) > 0. Asserts.

    static IntRangeSampler from_py(int lo, int hi,
                                   std::optional<std::vector<float>> weights);
        // pybind entry point. If weights is nullopt/empty → uniform(lo, hi).
        // If weights is present and non-empty → asserts
        //     weights.size() == hi - lo + 1
        // then weighted(lo, weights). Never silently adjusts hi.

    // Sampling. Always returns a value in [lo(), hi()].
    int operator()(std::mt19937_64& rng) const;

    // Accessors. Naming chosen to make "inclusive" impossible to forget.
    int  lo()   const noexcept;   // smallest possible value (inclusive)
    int  hi()   const noexcept;   // largest  possible value (inclusive)
    int  span() const noexcept;   // hi - lo + 1  (number of possible values, always ≥ 1)
    bool is_weighted() const noexcept;

    // For reproducibility / debug dumps.
    const std::vector<float>& weights() const noexcept;   // empty ⇒ uniform

private:
    int lo_;
    int hi_;
    std::vector<float> weights_;                                   // empty ⇒ uniform
    mutable std::discrete_distribution<int> discrete_;             // built once
};

} // namespace graphgen
```

**Name choice.** Candidates considered:
- `GivenDistro` (user's suggestion) — good, but the class is used even when the caller *didn't* give a distribution (the uniform fallback path). "Given" is misleading.
- `Categorical` / `CategoricalInt` — mathematically accurate but overloaded jargon.
- `WeightedRange` — reads well but doesn't say "sampler".
- **`IntRangeSampler`** — recommended. Matches the codebase's existing `SampleIntPartition` naming. Says exactly what the object does: samples an int in a range. Doesn't lie when weights are absent (it's still a sampler, just uniform).
- `PyDist` / `PyDiscreteDist` — leaks the origin. The object is a C++ value; it doesn't know or care that its weights came from Python. Better hidden in `from_py`.

Do *not* introduce a shorter alias like `using Sampler = IntRangeSampler;`. The bare name `Sampler` is already ambiguous in this codebase — `graph_sampler_` (pipeline stage), path samplers on the shortest-path DAG (I9-Clarification), and `sample_choice_node` inside `ShortestPathTask` are all distinct things. Keep the full `IntRangeSampler` name at every use site.

**Range semantics — inclusive on both ends, always.** The current codebase mixes two range conventions and does ad-hoc offset arithmetic per call site, which is the root cause of the `[size()-1]` bug in `make_khops` and the silent uniform-vs-weighted-branch drift in `ShortestPathTask` / `BFSTask` / `random_tree_generator`. The wrapper class picks **one** convention and enforces it:

> **`IntRangeSampler` samples an integer in `[lo, hi]` inclusive-inclusive. Always. No exceptions.**

Rationale:
- Matches `std::uniform_int_distribution<int>(lo, hi)` semantics — the closest C++ analogue.
- Matches how humans describe ranges in the config ("path lengths 3 to 7 inclusive").
- Turns the current implicit `+ min` offset into an explicit `lo` field at construction time. The offset is set *once*, at the config site, not re-derived from a length subtraction at every sample.

**Consequence for weight vectors.** When a weighted branch is used, the wrapper *requires* `weights.size() == hi - lo + 1`. This is checked at construction time with a clear error message. No silent adjustment. The reason: today the weighted branch and the uniform-fallback branch of the *same* config field can end up with different ranges (if the caller's `weights` vector length doesn't match `max - min + 1`), and no one notices until a downstream tensor comes out the wrong shape. Failing loud at config load is strictly better than failing silently at every batch.

**Accessor naming, chosen to make "inclusive" impossible to forget:**
- `lo()` — smallest possible value (inclusive).
- `hi()` — largest possible value (inclusive).
- `span()` — `hi() - lo() + 1`. The number of possible values. Always positive.
- No `size()`, no `max_index()`, no `end()`. Those invite the "is it inclusive?" question all over again.

**Factories (all preserve the inclusive-inclusive invariant):**

```cpp
static IntRangeSampler single(int v);
    // Degenerate: always returns v. Fast path for min == max.

static IntRangeSampler uniform(int lo, int hi);
    // Uniform over [lo, hi]. Requires lo <= hi; asserts.

static IntRangeSampler weighted(int lo, std::vector<float> weights);
    // hi = lo + weights.size() - 1. Weights re-normalised internally by
    // discrete_distribution (no sum-to-1 requirement — see semantic note below).
    // Requires weights non-empty and all non-negative; asserts.

static IntRangeSampler from_py(int lo, int hi, std::optional<std::vector<float>> weights);
    // The pybind entry point.
    // If weights is nullopt or empty:  uniform(lo, hi).
    // If weights is provided and non-empty:  asserts weights.size() == hi - lo + 1,
    //                                        then weighted(lo, weights).
    // Never silently adjusts hi or ignores the weights vector's length.
```

**Semantic note: weights don't need to sum to 1.** `std::discrete_distribution` auto-normalises, so `[1, 2, 1]` and `[0.25, 0.5, 0.25]` behave identically. This is the current behaviour of the code and we keep it. What we require is `all(w >= 0)` and `sum(w) > 0` — asserted at construction. (V1's dead code in `old_code/generator.cpp` L1165 used to throw on `sum != 1`. We are deliberately not restoring that: it forced users to normalise on the Python side for no engine-side benefit.)

**Every current call site translates cleanly:**

| Today's code | Under `IntRangeSampler` |
|---|---|
| `ShortestPathTask` uniform: `vector<float>(max_path_length - min_path_length + 1, 1.0)` → `discrete_distribution` → `d(gen) + min_path_length` | `cfg.path_length_sampler(rng)`. Config field constructed as `IntRangeSampler::from_py(min_path_length, max_path_length, cfg.path_length_weights)`. |
| `ShortestPathTask` weighted: `discrete_distribution(weights)` → `d(gen) + min_path_length` (no size check) | Same. Size check is at config-load time, not at every sample. |
| `BFSTask`: identical copy-paste of ShortestPathTask | Same single line, same shared config field. Copy-paste eliminated. |
| `make_khops` uniform: `uniform_int_distribution<int>(min_khops, max_khops)` | `cfg.khops_sampler(rng)`. |
| `make_khops` weighted: `discrete_distribution(weights)(gen) + min_khops`, plus buggy `khops_max_k = weights[size()-1] + min_khops` | `cfg.khops_sampler(rng)` for the sample; `cfg.khops_sampler.hi()` for `khops_max_k`. **Bug fixed by construction** — the wrapper knows its own upper bound. |
| `sample_num_nodes` uniform: `uniform_int_distribution<int>(min, max)` with a special case for `min == max` | `IntRangeSampler::from_py(min, max, nullopt)(rng)`, or `IntRangeSampler::single(min)` if `min == max`. Special case moves to the factory. |
| `random_tree_generator` weighted: `discrete_distribution(probs)(gen) + 1` (implicit "at least 1 child" offset) | `IntRangeSampler::weighted(1, probs)(rng)`. The `+1` becomes `lo=1`, made explicit at the config site. No hidden offset. |
| `random_tree_generator` unweighted: `binomial_distribution(d, p)` (range `[0, d]`) | **Stays separate.** Binomial isn't a discrete uniform categorical; keep it as its own config field (e.g., `optional<BinomialParams>`). See "What it does NOT replace" below. |

Note the last row: the weighted branch of `random_tree_generator` today samples in `[1, len(probs)]` (min 1 child) while the unweighted branch can sample 0 (fixed later by a safety check). Under `IntRangeSampler` we make this **explicit**: the weighted case is `IntRangeSampler::weighted(1, probs)`, so the offset is stated at construction. If the user wants a weighted distribution over `[0, d]` instead, they pass `IntRangeSampler::weighted(0, probs)`. No more silent baked-in `+1`.

**Where it replaces existing code.**

- **`GeneratorConfig` (post-R1).** Replace `optional<vector<float>> task_sample_dist` with `IntRangeSampler path_length_sampler`, constructed once at config load via `IntRangeSampler::from_py(min_path_length, max_path_length, task_sample_dist_opt)`. The `from_py` factory absorbs the "if provided use weights, else uniform over `[min, max]`" logic — moves the fallback decision to *construction time*, not per-sample time, and enforces the size-match assertion once.
- **`GeneratorConfig` (post-R1).** Replace `optional<vector<float>> probs` with `IntRangeSampler branching_sampler` for the random-tree case. Note: the current fallback for `probs` is not uniform — it's *binomial*. That's a different distribution family and should stay a separate config field (see below).
- **`ShortestPathTask` and `BFSTask` ctors.** The distribution-building blocks in `tasks.h` become a single line: `int sampled_len = cfg.path_length_sampler(rng);`. Deletes ~10 lines of duplicated code across the two tasks.
- **`make_khops` (`graph_wrapper.h` L161–L167).** Becomes `khops_k = cfg.khops_sampler(rng); khops_max_k = cfg.khops_sampler.hi();` — the buggy `[size()-1]` indexing goes away.
- **`sample_num_nodes` (`graph_wrapper.h` L98–L106).** Becomes `num_nodes = cfg.num_nodes_sampler(rng);`. The `min == max` special case moves into the `IntRangeSampler::single` factory (used at config load when `min == max`), not repeated at every call site.
- **`args.h` pybind parsing.** Add a helper `IntRangeSampler parse_int_range_sampler(const py::kwargs&, string name, int lo, int hi)` that centralises the `contains → is_none → cast<py::list>().empty() → cast<vector<float>>` dance and returns a fully-formed `IntRangeSampler`. Every distribution field uses this one helper. The size-match check happens here, so config errors surface at Generator construction, not on the first `Worker::process_item` call.


**What it does NOT replace.**

- **`random_tree_generator`'s binomial fallback.** When `probs` is absent, the current behaviour is to fall back to a `binomial_distribution(d, bernoulli_p)`, not a uniform over `[1, d]`. That's a categorical-vs-binomial choice, not a "weighted-vs-uniform" one. Keep the binomial branch explicit: store a `variant<IntRangeSampler, BinomialSampler>` in the config, or (simpler) keep the `bernoulli_p` field alongside the `IntRangeSampler` and let the caller decide. Whatever the shape, don't paper over the semantic difference by pretending binomial is a special case of `IntRangeSampler`.
- **`std::uniform_real_distribution<float>` call sites** (`undirected_graphs.h` L280, `tasks.h` L169). Those sample floats, not ints in a range. Out of scope — this improvement targets the discrete-int-from-Python pattern specifically. A separate `RealSampler` could follow later if the same duplication shows up for floats, but grep shows it doesn't right now.
- **The 15+ inline `uniform_int_distribution<int>(...)(rng)` calls in `directed_graphs.h` and `undirected_graphs.h`.** Those are ephemeral, sample-once distributions with locally-computed bounds (`0, c1.size()-1`, `0, num_nodes-1`, etc.). They don't come from Python and don't repeat across items. Wrapping them buys nothing.

**Where it should live.** New header `int_range_sampler.h` in the `graphgen` namespace. Implementation inline (small class, ~50 lines). No dependencies beyond `<random>`, `<vector>`, `<optional>`.

**Cost.**
- ~60 lines for the class + tests (round-trip: weights → sampling frequencies).
- ~4 call sites updated (2 tasks, 1 `make_khops`, 1 `random_tree_generator`).
- 1 helper in `args.h` (the parse helper), replacing ~15 lines of hand-rolled kwarg parsing across two fields.
- Net line count: probably neutral or slightly negative. The main win is *conceptual*: "a Python-supplied categorical distribution over an int range" becomes a first-class thing with one name, one API, one place to fix bugs.

**Why now.** R1 rewrites the argument-parsing surface anyway (kills `TaskArgs` in favour of `GeneratorConfig`). Slotting `IntRangeSampler` in as a first-class field type at that moment is free; retrofitting it after is a rewrite of every task ctor. Also directly fixes one latent bug (`make_khops` `[size()-1]` indexing) that the refactor's testing (R7) would likely catch, but better to fix at the type-system level than by test coverage.

### I6. Deterministic per-item seeding: reproducible output regardless of worker count

**The problem.** Today the RNG state is threaded through generation — the RNG advance order depends on which item is generated when. Post-R10 this gets worse, not better: each `Worker` owns its own RNG (good for lock-free parallelism), but that means `num_workers=1` and `num_workers=4` will produce *different* batches even from the same base seed, because item `i` in the batch is generated by different workers with different RNG histories in the two cases. Reproducibility becomes tied to hardware/scheduling.

This is a real problem, not a theoretical one. Every training run needs to be reproducible for:
- Debugging: "item 173 in batch 42 produced garbage — reproduce it."
- Paper experiments: results should not depend on how many CPU cores the reviewer's machine has.
- Bisecting regressions: swapping algorithm A for algorithm B and checking that outputs match requires the *same input distribution*, which means the same item-by-item RNG stream.

**The fix.** Seed each item's RNG from a hash of `(base_seed, batch_index, item_index)` at the start of that item's precompute. Item-level determinism, worker-count-independent.

```cpp
// In Worker::process_item(int batch_index, int item_index):
uint64_t item_seed = splitmix64(base_seed_
                              ^ (uint64_t(batch_index) * 0x9E3779B97F4A7C15ULL)
                              ^ uint64_t(item_index));
rng_.seed(item_seed);
// ... proceed with graph sampling, task computation, tokenization ...
```

Any good bit-mixing hash works (`splitmix64`, `xxhash`, PCG's `.set_stream()` if we switch RNGs). What matters is:
1. **Different `(batch_index, item_index)` pairs produce statistically independent seeds** — hash mixing takes care of this.
2. **Same `(base_seed, batch_index, item_index)` always produces the same seed** — pure function, no hidden state.
3. **The mapping is worker-independent** — Worker 0 processing item 173 gets the same seed as Worker 3 would processing item 173.

**Cost.** Essentially free. One 64-bit hash call per item (~5–20 ns) versus the ~10–100 μs of work in an item. `std::mt19937_64::seed()` itself is ~10 μs; if that matters we can switch to a lighter RNG (PCG64 seeds in ~50 ns) but for now `mt19937_64` is fine.

**Reproducibility gain.**
- Any failing item can be reproduced by saving `(base_seed, batch_index, item_index)` — three integers.
- `num_workers=N` gives identical output for all `N ≥ 1`. Users can scale worker count freely.
- Composes with I7 (batch stats): stats can include the per-item seed, so "which items produced these outliers" becomes trivially answerable and reproducible.
- Composes with I5 (`IntRangeSampler`): the sampler's `operator()(rng)` is a pure function of the RNG, so seeding the RNG at item start bounds all randomness.

**Trade-off.** Re-seeding per item means the RNG's warmup cost pays per item, not per batch. For `mt19937_64` this is measurable (the state is 2.5 KB — cache-warm-friendly, but still a fixed cost). Two mitigations:
1. Only re-seed at item boundaries, so within an item the RNG runs from a single seed — natural place, no perf regression on the hot path.
2. If profiling shows the seed cost significant, switch to a splittable RNG (PCG64, Xoshiro256++, or `std::seed_seq` with a stream index). All ~10× faster to seed than `mt19937_64`.

**Not free — one real caveat.** Determinism holds *only* if every non-RNG source of nondeterminism is also controlled. Specifically:
- No use of `std::unordered_map` / `std::unordered_set` iteration order that depends on hash randomization. Grep the codebase; if any exist on the hot path, either replace with `std::map` or fix the iteration order explicitly.
- No floating-point summation whose order changes with parallelism. If we ever add a parallel reduction on floats in the compute phase, that breaks bit-identical determinism — call it out at that point.
- The system RNG (`std::random_device`) is never called during generation. Grep confirms this today; enforce it with an assertion.

**Where it lives.** One helper `graphgen::seed_for_item(uint64_t base_seed, int batch_index, int item_index)` in a small `seed.h`. Called from `Worker::process_item` (post-R10). ~10 lines total.

**Why now.** R10 is *the* refactor that introduces per-`Worker` RNG in the first place. Adding item-level seeding at the same time is one line; adding it later means auditing every `Worker::process_item` call to prove it doesn't touch RNG in the wrong order. The design decision needs to land with R10 or be deferred forever.

### I7. Return batch statistics from C++, not reconstructed on the Python side

**The problem.** Python-side stats collection (measuring things like avg num_nodes, task-kind distribution, rejection rate) is currently done by decoding the tokenized output back into primitives — parsing the token stream, mapping token IDs back to graph structure via the dictionary, counting things. This has three problems:
1. **Fragile.** Any change to tokenization breaks stats collection silently. The stats become quietly wrong until someone notices.
2. **Slow.** Decoding 512 items × N tokens × dictionary lookup is easily 10s of milliseconds per batch on the Python side. That's a real hit in a training loop.
3. **Incomplete.** Some stats (rejection count, per-phase timing) *cannot* be reconstructed from tokens because that information was thrown away during generation.

The C++ side already has all of this information — every `SampledGraph` has `num_nodes` and `num_edges` in it, every `Task` has `kind` and task-specific params, the precompute phase measures its own wall time, rejections happen inside `GraphSampler` where they can be counted. **We're computing this data, throwing it away, then reconstructing a lossy approximation from tokens.** The fix is to stop throwing it away.

**The proposal.** Add `return_stats: bool = False` to the generator entry point. When true, return a Python dict alongside the tokenized batch:

```python
batch, stats = generator.generate(config, batch_size=512, return_stats=True)

# stats is:
{
    "per_item": {
        "num_nodes":       np.ndarray[int32, (512,)],
        "num_edges":       np.ndarray[int32, (512,)],
        "diameter":        np.ndarray[int32, (512,)],   # if computed; else -1
        "task_kind":       np.ndarray[int32, (512,)],   # enum value
        "path_length":     np.ndarray[int32, (512,)],   # -1 for tasks without a path
        "k":               np.ndarray[int32, (512,)],   # for khops; -1 otherwise
        "rng_seed":        np.ndarray[uint64, (512,)],  # composes with I6
    },
    "batch": {
        "precompute_time_ms": float,
        "compute_time_ms":    float,
        "task_kind_counts":   dict[str, int],
        "total_rejections":   int,
    },
}
```

The exact schema stays flexible until we build it, but the shape is: **fixed-shape typed arrays per item, scalar aggregates per batch**. Numpy arrays (not Python lists) so downstream stats consumers can vectorize.

**Cost.** Almost nothing.
- Precompute phase already produces `PrecomputedItem { SampledGraph, Task, Scratchpad, Layout }` (per Tier 3E Approach D). Extracting `num_nodes`, `num_edges`, task-kind, etc. is one field read each.
- Rejection count is already tracked by `GraphSampler` — right now it's used for retry loops and then discarded; instead, expose it.
- Precompute and compute wall times are already measured by `Worker` (per R10's timing infrastructure).
- Total added work per batch: ~O(batch_size) integer field copies. Negligible.

The cost of *not* collecting stats when `return_stats=False` is exactly zero — the collection is guarded by a compile-time-ish `if (return_stats_)` check inside `Worker::process_item`.

**Wins across the three criteria:**
- **Debuggable:** monitor training data distribution over time. Catch drift. Answer "why did loss spike at step 4000?" with "the average num_nodes doubled that batch." Per-item `rng_seed` (composes with I6) turns "this specific item was weird" into "here's a one-line reproducer."
- **Flexible:** stats can drive adaptive curriculum ("if the model has converged on 10-node trees, sample larger"), stratified sampling, drift detection, W&B logging without writing custom decode code.
- **Fast:** replacing Python-side tokenized-batch decoding (currently 10s of ms per batch) with a direct C++ dict is a real perf win in the training loop even before counting the increased sanity.

**Migration.** The user's existing Python-side stats collection code becomes deletable once `return_stats=True` provides equivalent data. Do this as a follow-up: add the flag in C++, verify parity against the Python decoder on a sample of batches, then delete the Python decoder.

**Why now.** R10 / Tier 3E Approach D already funnels every batch item through the precompute phase with all these values in a `PrecomputedItem` struct — the stats are *right there*, we're just discarding them. Adding the collection at the same time as those pipeline structs land is trivial; retrofitting after means rewriting the precompute-to-compute handoff. Also: I6 (per-item seed) needs the stats channel to be useful — knowing an item's seed only matters if you can also see what made that item unusual.

**Not included.** No streaming stats ("give me stats as items complete"). No histogram support in the C++ side (Python can histogram a numpy array in one line). No per-item timing (would require per-item RDTSC calls, small but nonzero cost — add later only if wanted, guarded by an extra flag).

### I8. Graph-kind catalog: sparse-graph families to consider adding

The current suite covers uniform random (Erdős–Rényi), trees (via `random_tree_generator` — d-ary / binomial / weighted branching, plus path-star), Euclidean radius graph, balanced DAGs, and the k-hops construction. There are several major *topological families* not represented at all today. This catalog lists them so we can decide which to add. Each new kind maps cleanly to a `GraphKind` enum entry + a `GraphSampler::sample_<kind>` private method in the post-R15 design (one class per pipeline stage, dispatched by enum).

**How to read this catalog.** Entries are grouped by tier:
- **Tier A (recommended)**: covers a topological family not currently in the suite. Small implementation. High research payoff.
- **Tier B (nice-to-have)**: complements existing coverage without opening a new family. Small to medium cost.
- **Tier C (tree variants)**: very small — a new arg or a 10-line variation of `random_tree_generator`. Cheap to add; probably low individual value but cumulative diversity is useful.
- **Tier D (speculative)**: interesting but less standard, higher implementation cost, or unclear research payoff.

Costs listed as "lines" are approximate skeleton size (no tests, no bindings, no docstrings). Multiply by ~2 for full production-quality with R7 reference-vs-fast tests.

---

#### Tier A — the big missing families (recommended)

**A1. Stochastic Block Model (SBM).**
- **What**: N nodes partitioned into `k` blocks; each pair of nodes has edge probability `p_intra` if same block, `p_inter` if different. Set `p_intra ≫ p_inter` for the classical community-structure regime.
- **How generated**: Two nested loops over block pairs; Bernoulli draw per pair. Sparse when `p_intra ≈ c₁/n_block` and `p_inter ≈ c₂/n` with `c₁, c₂` constants.
- **Sparsity**: Yes, at the natural parameterisation. Expected edges = `O(n)`.
- **Benefits**:
  - Tests whether the LM discovers *latent* community labels — the block assignment isn't visible in the token stream, but shortest paths crossing blocks behave differently from within-block paths.
  - Standard in graph-benchmark literature; makes results comparable to graph-neural-network papers.
  - Trivially variable-community-count (`k` is a config knob) — one generator produces a family of difficulties.
- **Downsides**:
  - The obvious LM heuristic ("route through cluster centres") may be too easy to learn; distinguishing "LM discovered community structure" from "LM memorised a fast heuristic" needs careful task design.
  - Not connected by default at low densities — need to either reject disconnected samples (see `graph_datasets.py` L122 warning) or run the existing connectivity fixup.
- **Cost**: ~30 lines.
- **Extension: hierarchical SBM.** Blocks contain sub-blocks recursively. Distances reflect the tree-distance in the hierarchy. Probes reasoning at multiple scales simultaneously. ~15 additional lines wrapping A1.

**A2. Barabási–Albert (BA) preferential attachment.**
- **What**: Start with a small seed graph (usually `m` nodes forming a complete graph or a path). Add nodes one at a time; each new node connects to `m` existing nodes chosen with probability proportional to current degree.
- **How generated**: Maintain a "target list" — a vector of node indices where each node appears once per outgoing edge. Sample from it uniformly to pick a target (this is equivalent to sampling by degree). Update the target list as edges are added.
- **Sparsity**: Yes. Exactly `m·(n − m)` edges.
- **Benefits**:
  - Produces scale-free (heavy-tailed) degree distribution — the family of graphs missing from every generator we currently have.
  - Small diameter (`~log log n`) means shortest paths are usually short but not trivial.
  - Introduces a *natural heuristic* ("route through the hub") that we can specifically probe for — did the LM learn it, and can it recover when the hub is not on the geodesic?
  - Sampling procedure reuses I5's `IntRangeSampler` machinery directly (weight-by-degree = categorical over current degrees).
- **Downsides**:
  - The "route through hub" heuristic may make shortest-path tasks *too easy*; consider designing tasks that specifically avoid hub paths (e.g., queries on low-degree pairs).
  - Not variable-degree at the config level (`m` is fixed per call), so degree-distribution shape is largely determined by `n` and `m`.
- **Cost**: ~25 lines.

**A3. Watts–Strogatz small-world.**
- **What**: Start with a ring lattice where each of `N` nodes is connected to its `k` nearest neighbours on the ring (so every node has degree `2k`). Then, for each edge, with probability `p`, rewire one endpoint to a uniformly random other node.
- **How generated**: Build the ring lattice by index arithmetic (edges `(i, (i+j) mod N)` for `j = 1..k`), then walk the edge list and rewire with probability `p`.
- **Sparsity**: Yes, always. Exactly `N·k` edges regardless of `p`.
- **Benefits**:
  - Distinct middle regime (`p ≈ 0.05–0.2`) where the graph is *still* highly clustered but diameter has collapsed to `~log n`. This regime specifically probes "long-range shortcut" reasoning.
  - Two boundary cases (`p = 0` = pure ring, `p = 1` = random) are also useful as reference-vs-fast anchors: known analytic diameter, known analytic clustering coefficient.
  - Cheap tunable-difficulty knob (`p`).
- **Downsides**:
  - Not connected at very low `p` if `k` is small — usually not a problem for `k ≥ 2`, but worth an assertion at construction.
  - Doesn't produce heavy-tailed degree; every node has degree exactly `2k` (up to the rewiring perturbation). So it's not a replacement for A2.
- **Cost**: ~25 lines.

**A4. Grid / lattice graphs (2D grid, torus, hypercube).**
- **What**: Nodes at integer coordinates in `d` dimensions; edges between coordinate-adjacent nodes.
  - **2D grid**: nodes `(i, j)` for `i ∈ [0, W)`, `j ∈ [0, H)`. Boundary nodes have fewer neighbours.
  - **Torus**: same but with wrap-around edges (`(i, 0)` connects to `(i, H-1)`). Every node has exactly `2d` neighbours.
  - **Hypercube**: `2^d` nodes labeled by d-bit strings; edges between strings differing in exactly one bit.
- **How generated**: Direct index arithmetic. No RNG needed for the graph structure itself (only for choosing source/target for tasks).
- **Sparsity**: Yes. Grid/torus: `d·N` edges. Hypercube: `d · 2^(d-1)` edges (very sparse for large d).
- **Benefits**:
  - **Analytic ground truth.** Grid shortest path = Manhattan distance (`|Δx| + |Δy|`). Torus = ditto with wrap. Hypercube = Hamming distance. These are the best possible reference implementations for R7's reference-vs-fast testing — you can catch BFS/Dijkstra bugs by direct comparison, no other graph algorithm involved.
  - **Determinism**: same input `(W, H)` always produces the same graph. Cheap way to write regression tests that pin behaviour exactly.
  - **Probes coordinate reasoning.** If the LM's tokenisation includes any spatial hint, does it discover to add coordinates? If not, does it still learn to route?
  - Hypercube specifically has clean combinatorial structure — dimension `d` doubles the graph size and adds one to the diameter. Nice for scaling experiments.
- **Downsides**:
  - **Not variable across items in an interesting way**: two grids of the same size are the *same graph* (up to labelling). Diversity has to come from varying `W, H` and choice of source/target. Consider whether that's enough variety for training data or if grids are better as a *test* set only.
  - Hypercube's `2^d` node count means the size knob is coarse — 4, 8, 16, 32, 64 only. Between-size interpolation isn't possible.
- **Cost**: ~15 lines each; three variants share ~30 lines total.

**A5. Random Spanning Tree + Extras (RSTE) — the ER analogue of Delaunay.**
- **What**: A connected sparse random graph, built by combining a uniform random spanning tree (which is connected by construction) with independent Bernoulli-sampled extra edges from the remaining pool.
- **How generated**: (i) Sample a uniform spanning tree over `n` labeled nodes via Wilson's algorithm (`O(n log n)` expected on the complete graph). That's `n − 1` edges, guaranteed to form a single connected component. (ii) Walk the remaining `C(n, 2) − (n − 1)` non-tree edges and include each with probability `p'`.
- **Sparsity**: Yes, controllable. Total expected edges = `(n − 1) + p' · (C(n, 2) − (n − 1))`. For `p' = O(1/n)` you get an `O(n)`-edge connected sparse graph.
- **Benefits**:
  - **Connected by construction.** Same guarantee Delaunay gives for the Euclidean case, but for the random-edge family. No rejection loop, no post-hoc fixup.
  - **Bounded, deterministic runtime.** Wilson's algorithm has expected `O(n log n)` runtime with a small variance; unlike rejection sampling, worst-case is bounded. Compatible with the two-phase precompute architecture (R10 / Tier 3E) which needs predictable per-item cost.
  - **Sparse in the strong sense** — you can get `O(n)`-edge connected random graphs, which post-hoc-fixup ER cannot cleanly deliver without operating below the connectivity threshold and paying rejection cost.
  - **Uniform spanning tree marginal** — the spanning-tree part is exactly a uniform random labeled tree (Cayley's formula: `n^(n−2)` such trees). Well-studied distribution; if a paper needs a distributional statement, this is a clean one.
  - **Parameter `p'` has the same interpretation as ER's `p`**, just applied to the non-tree edge set — easy to explain in a paper.
- **Downsides**:
  - **Not literally ER conditioned on connectedness.** The distribution is *biased toward tree-like graphs* — every sample contains the `n−1` tree edges plus independent extras. Very close to `G(n, p') | connected` at high `p'` but distinct at low `p'`. If a paper claim depends on the exact ER distribution, this substitution is not defensible.
  - **Wilson's algorithm is ~30–40 lines correctly.** Not hard, but requires care with the random walk termination condition (each new node walks until it hits the current tree). A reference implementation via Aldous–Broder (simpler but less efficient) is another ~15 lines and is a good R7 reference target.
  - **Degree distribution differs from ER** — RSTE's degree distribution is a convolution of the UST-degree distribution (which has heavier tails than ER at similar density) with the Bernoulli extras. Not necessarily bad, but worth knowing if degree distribution is a claim in the paper.
- **Cost**: ~40 lines (Wilson) + ~10 lines (extra-edge Bernoulli pass) + ~15 lines (Aldous–Broder as R7 reference).

##### Connectivity strategies for random-edge families (ER, RSTE, and relatives)

Since the codebase already has an ER connectivity fixup at [undirected_graphs.h#L243](undirected_graphs.h#L243) (referenced by Tier 2C), it's worth documenting the four strategies for producing a connected sparse random graph so the choice per graph kind is deliberate:

| Strategy | How it works | Distribution | Runtime | Sparse to `O(n)`? | Where used |
|---|---|---|---|---|---|
| **Post-hoc bridge edges** | Generate ER at any `p`. Find components. Add bridge edges between components. | ER + non-uniform bridges (biased by component-size choices) | Bounded, deterministic | Yes | Current codebase (`GraphKind::ErdosRenyi`). Keep as-is. |
| **Rejection sampling** | Generate ER. BFS-check connectivity. Retry if disconnected. | Exactly `G(n, p) | connected` | Unbounded near `p = log n / n`; can retry 100–1000× at sparse `p` | Yes (in principle; slow) | Only usable when `p > log n / n` gives high acceptance. Not recommended below threshold. |
| **Dense above threshold** | Choose `p > log n / n` so the ER graph is almost surely connected without any fixup. | ER distribution | Bounded, deterministic | No — edges scale as `n log n`, not `n` | Fine for `n`-log-`n`-edge regime. Loses the very-sparse regime. |
| **Spanning-tree backbone (RSTE)** | Wilson UST + Bernoulli extras. | UST + Bernoulli overlay. *Not* ER, but a clean nearby distribution. | Bounded, expected `O(n log n)` | Yes | New kind: `GraphKind::RSTE`. This entry. |

**Recommendation.** Keep the current `GraphKind::ErdosRenyi` with its post-hoc fixup (it's what the paper baseline currently uses; no reason to disturb it). Add `GraphKind::RSTE` as a *new* kind for use cases where "connected by construction, `O(n)` edges, distributional cleanliness" is preferred over "true ER distribution." The two coexist; the user picks per experiment.

Rejection sampling and "dense above threshold" don't warrant their own `GraphKind` entries — the first is unusable in the sparse regime, the second is what the current ER does when the user just picks a high `p`.

---

#### Tier B — nice-to-have, complements existing coverage

**B1. Delaunay triangulation.**
- **What**: Sample `n` points uniformly in the unit square. Connect them via the Delaunay triangulation — each triangle's circumcircle contains no other point.
- **How generated**: Incremental Bowyer–Watson algorithm (add points one at a time, retriangulate affected region). Or use a pre-built library.
- **Sparsity**: Yes. Planar graph, so at most `3n − 6` edges.
- **Benefits**:
  - Always connected, always planar. Fixes the Euclidean radius graph's tendency to produce isolated components.
  - Graph distance closely tracks Euclidean distance — good for testing whether the LM discovers geometric shortcuts.
  - Standard geometric graph. Well-studied.
- **Downsides**:
  - Bowyer–Watson is finicky to implement correctly. Roughly 5× the code volume of A1–A3. Consider whether we actually need it or whether A4 (grid) + Euclidean radius already cover the geometric-graph slot.
  - Numerical robustness: near-degenerate point configurations (four cocircular points) require careful predicate implementations. Use exact-arithmetic predicates or accept occasional topology glitches.
- **Cost**: ~50 lines for correct Bowyer–Watson, or ~20 lines wrapping an existing lib (adds a dependency).

**B2. k-nearest-neighbours (k-NN) graph.**
- **What**: Sample `n` points in `R^d`. For each point, connect it to its `k` nearest neighbours. Undirected variant: take the symmetric union or intersection.
- **How generated**: For each point, compute all pairwise distances (`O(n²)` — fine for `n ≤ few thousand`) and pick the `k` smallest. Or use a kd-tree if `n` grows.
- **Sparsity**: Yes. Exactly `k·n` directed edges (fewer in the symmetric-union undirected variant, more in intersection).
- **Benefits**:
  - Different edge structure from the existing Euclidean radius graph: k-NN is *k-regular by construction*, radius graph has variable degree.
  - Always connected for reasonable `k` (say `k ≥ 5`).
  - Directly comparable to graph-representation-learning benchmarks (many GNN papers use k-NN graphs).
- **Downsides**:
  - Duplicates capabilities of Euclidean radius graph unless we specifically want the k-regular / always-connected property.
  - Directed vs undirected choice adds a config knob.
- **Cost**: ~30 lines (naïve O(n²) version).

**B3. Random regular graph.**
- **What**: Every node has exactly the same degree `d`. Otherwise uniformly random over all such graphs.
- **How generated**: Pairing model (also called the configuration model with all half-edges numbered equal): create `d` copies of each node index (called half-edges), pair them uniformly at random, retry on self-loops or parallel edges.
- **Sparsity**: Yes if `d = O(1)`. Exactly `d·n / 2` edges.
- **Benefits**:
  - **Probes reasoning with no degree cues at all.** Every node looks locally identical. If the LM was cheating by using degree as a route heuristic, this graph reveals that.
  - Well-connected (expander-like properties for random `d`-regular graphs), small diameter.
- **Downsides**:
  - Rejection loop can be slow when `d` is close to `n` (many rejections). For sparse cases (`d ≪ n`), acceptance rate is high; ignore this concern for our use case.
  - Requires `d · n` to be even (parity constraint from the pairing model).
- **Cost**: ~40 lines including rejection retry.

**B4. Cactus graph.**
- **What**: A connected graph in which every edge belongs to at most one cycle. Equivalently, a tree of cycles.
- **How generated**: Start with a tree. Randomly select subtrees and replace them with cycles (by adding a single back edge to close the subtree). Or grow by attaching new cycles / new trees to existing nodes.
- **Sparsity**: Yes. At most `2(n − 1)` edges (roughly twice a tree).
- **Benefits**:
  - **Mixes two reasoning types.** Tree portions have unique paths between nodes; cycle portions have exactly two paths. Tests whether the LM can distinguish "one path" from "two paths" locally, and handle branching correctly.
  - Well-studied class of graphs with clean algorithms (many hard graph problems become polynomial on cactuses).
- **Downsides**:
  - Slightly awkward generation — no single canonical algorithm; the "grow by attaching cycles" approach requires care to avoid overlapping cycles that would violate the cactus property.
- **Cost**: ~40 lines.

---

#### Tier C — tree variants (cheap additions given existing tree infrastructure)

Each below is a small variation of `random_tree_generator` — either a new parameter, a new attachment rule, or a small post-processing step on an existing tree. Individually each adds a small amount of diversity; collectively they'd roughly double the "tree shape" repertoire at very low cost. Table format because there's not much to say per entry.

| Kind | Description | Attachment rule | Sparsity | Benefit | Cost |
|---|---|---|---|---|---|
| **C1. Caterpillar** | A path with leaves attached | Build a path; each internal node gets 0–k leaves | Yes, `≤ 2n` edges | Long spine + leaf noise. Distinguishes "on-spine" from "off-spine" nodes | ~10 lines |
| **C2. Broom** | A path with a star at one end | Path of length `p`, then star of `n−p` leaves at endpoint | Yes | Asymmetric — one end looks like a tree, the other like a spider | ~10 lines |
| **C3. Random recursive tree** | Each new node attaches to a uniformly random existing node | Uniform pick from `[0, cur)` | Yes | Balanced-ish depth ~`log n`, no degree preference | ~5 lines |
| **C4. Preferential-attachment tree** | Each new node attaches to an existing node with probability ∝ degree | Weight-by-degree pick | Yes | Highly skewed degree — a few high-degree hubs. Like BA with `m=1` | ~10 lines |
| **C5. Deterministic complete d-ary tree** | Every internal node has exactly `d` children | Deterministic BFS fill | Yes | Perfect regularity, analytic diameter (`log_d n`). Reference-vs-fast anchor | ~10 lines |

Common downside for all of Tier C: individually they're small variations, and if we add all five plus the existing `random_tree_generator` we get 6 tree-shaped kinds which may over-represent trees in the training mix. Consider adding them as *options within* the existing tree generator (a `TreeShape` enum: `Random | Caterpillar | Broom | RecursiveUniform | RecursivePreferential | CompleteDAry`) rather than as separate `GraphKind`s. This keeps the config surface small while adding real diversity.

---

#### Tier D — speculative

**D1. Kronecker graph.**
- **What**: Recursively defined via the tensor (Kronecker) product of a small "seed" adjacency matrix with itself `k` times. Adjacency matrix has entries in `[0, 1]` interpreted as edge probabilities.
- **How generated**: Choose a `2×2` (or `3×3`) seed matrix. Kronecker-product it with itself `k` times; sample edges from the resulting probability matrix.
- **Sparsity**: Yes with careful seed choice. Scales as `n · log n` typically.
- **Benefits**:
  - Popular in ML benchmarks (Graph500). Produces heavy-tailed degree + self-similar structure simultaneously.
  - Tunable via the 4 (or 9) entries of the seed matrix.
- **Downsides**:
  - Node count fixed to `n = 2^k` (or `3^k`). Coarse size knob.
  - Fitting the seed matrix to produce specific properties is nontrivial. If we just use a canonical seed (like `[[0.9, 0.5], [0.5, 0.1]]`) we get a canonical Kronecker graph, which limits diversity.
  - Marginal over A2 (BA) — both produce heavy-tailed degree. Question is whether we want *two* heavy-tailed graph families or *one*.
- **Cost**: ~30 lines.

**D2. Random hyperbolic graph.**
- **What**: Sample `n` points in the hyperbolic disk. Connect points within a hyperbolic distance threshold.
- **How generated**: Sample `(r, θ)` polar coordinates: `θ` uniform on `[0, 2π)`, `r` sampled from a specific distribution to give the desired degree exponent. Connect points via hyperbolic distance formula.
- **Sparsity**: Yes, controllable.
- **Benefits**:
  - Produces both heavy-tailed degree *and* high clustering simultaneously (real-world graphs often show both; A2 gives one, A3 the other, this gives both).
  - Recent addition to network-science literature; distinguishes us from the standard-benchmark crowd if that matters for the paper narrative.
- **Downsides**:
  - Nontrivial to implement. Hyperbolic distance formula is straightforward but numerical precision at large `r` is finicky.
  - Less well-known — reviewer familiarity may vary. Explaining the model in the paper adds friction.
- **Cost**: ~50 lines with the distance predicate.

**D3. Series-parallel graph.**
- **What**: Recursively defined. A single edge is series-parallel. If `G1` and `G2` are series-parallel, then:
  - *Series composition*: identify a terminal of `G1` with a terminal of `G2`.
  - *Parallel composition*: identify both terminals of `G1` with both terminals of `G2`.
- **How generated**: Recursive construction — at each level, sample "series" or "parallel" from a Bernoulli, recurse on two smaller series-parallel graphs.
- **Sparsity**: Yes.
- **Benefits**:
  - Strong decomposition structure — many hard problems become polynomial. Probes whether the LM discovers hierarchical decomposition.
  - Two terminals per subgraph gives a natural "source, target" pair for shortest-path tasks — no separate query sampling needed.
- **Downsides**:
  - Somewhat esoteric. Reviewer familiarity likely low.
  - Diameter depends heavily on the series/parallel mix — hard to hit a specific target diameter.
- **Cost**: ~40 lines.

---

#### Cross-cutting concerns for all new kinds

Independent of which specific kinds get added, three questions apply uniformly:

1. **Connectivity guarantee.** Some kinds (A1, A4-hypercube, B1, B2, B3, C1-C5) are always connected by construction; others (A1 low-density, existing ER) can produce disconnected graphs. For task types that require a path between source and target (shortest_path, BFS), disconnected components silently break sampling. Two options:
   - **Reject non-connected samples** at generation time (`GraphSampler` retries). Simple; potentially slow at high disconnection rates.
   - **Sample source/target from the same component** at task time (`TaskComputer`'s responsibility). More work but no rejection cost.
   Decide once, apply uniformly. This is a shared design decision, not a per-kind decision.

2. **Directedness.** Per I1 (all kinds directed-or-undirected), each new kind should have both a directed and undirected constructor from the start. For most of these the natural definition is undirected; the directed variant is either "orient each edge with a coin flip" or "orient by node index" (DAG). Pick one convention.

3. **R7 reference tests.** For each new kind, the R7 reference implementation needs a "slow but obviously correct" version to compare against. For A4 (grids) and C5 (complete d-ary) the reference is *analytic* (Manhattan distance, tree distance) — the highest-quality reference possible. For A2, A3, B1–B4 the reference is a plain BFS on the naïvely generated graph. Grids and complete d-ary trees are therefore doubly-valuable — they anchor the reference-implementation infrastructure.

---

#### Recommendation

If I had to pick a minimal set that meaningfully expands the topological coverage:

1. **A1 (SBM)** — for community structure.
2. **A2 (BA)** — for scale-free degree.
3. **A3 (Watts–Strogatz)** — for small-world.
4. **A4 (grid + hypercube)** — for analytic ground truth, R7 anchoring, and dimension-scaling experiments.
5. **A5 (RSTE)** — for connected-by-construction sparse random graphs. The direct ER analogue of Delaunay for the Euclidean case; complements the existing `ErdosRenyi` (which stays with its post-hoc fixup) rather than replacing it.

Total cost: ~170 lines of graph code + ~150 lines of R7 reference impls + config plumbing.

If we want to double the tree repertoire cheaply, fold Tier C into `random_tree_generator` as a `TreeShape` enum (~40 lines total for all 5 variants combined).

Everything in Tier B and D can wait until we see how the model actually performs on Tier A and can identify specific missing capabilities to probe.

**Why now.** New graph kinds map trivially into the R15 design: one enum entry + one private `GraphSampler::sample_<kind>` method + one entry in the `Task::from` dispatch. Adding a graph kind post-R15 is *cheaper* than adding one today (which requires threading through `undirected_graphs.h` / `directed_graphs.h`, `graph_wrapper.h`, `args.h`, `instance.h`, and Python bindings). Adding these during or shortly after the refactor is the natural time to expand the catalog while the machinery is fresh.

---

### I9. Sample *genuinely k-hop-hard* paths, not just paths of length k

**Problem.** The current path tasks (`ShortestPathTask`, `BFSTask`) sample pairs `(u, v)` with `d(u, v) + 1 = k` and hope this gives a k-hop reasoning problem. It doesn't — two failure modes collapse the true difficulty below `k`:

1. **Deterministic prefix/suffix (forced hops).** At interior node `v_i` on the sampled path, if the only neighbor lying on *any* shortest `u → v` path is the next node `v_{i+1}`, the model has no decision to make at position `i`. It can follow the forced edge without reasoning about the endpoint. In the extreme, a random tree has *zero* decisions after the first step: whichever child you pick at `u`, the remainder to any descendant is fully forced.

2. **Trivially-refutable alternatives.** At `v_i` with multiple neighbors, if the non-path neighbors `w` have `d(w, v) > d(v_i, v)` (they lead *away* from `v`), the model looks one step ahead and rules them out for free. The choice among neighbors collapses to "pick the one closer to `v`" — a one-step, non-k-hop decision. Only when the alternative *also* lies on a shortest path (or on an equally-competitive continuation) does the step demand real look-ahead.

**These are the same thing.** A step at `v_i` is a *genuine* k-hop decision iff `v_i` has out-degree ≥ 2 in the **shortest-path DAG** `D_{u,v}` — the subgraph of edges `(x, y)` satisfying `d(u, x) + 1 + d(y, v) = d(u, v)`. Every shortest `u → v` path is a source-to-sink path in `D_{u,v}`. Steps with DAG-out-degree 1 are forced / trivially refutable regardless of full-graph degree.

Note that failure mode (2) in the *strict* sense ("a shorter alternative exists") is already prevented by the current `d(u, v) + 1 = k` filter — no `u → v` route can be shorter than `k` by construction. What (2) actually means in practice is "the alternatives at each step are cheaply refutable," which is exactly the DAG-out-degree-1 condition. So (1) and (2) are one issue: **not enough branching on the shortest-path DAG along the sampled path.**

**Current ad-hoc fix.** `ShortestPathTask::sample_choice_node = 0.9` at [tasks.h#L120](tasks.h#L120): with 90 % probability, prefer `(u, v)` pairs whose *start* has degree ≥ 2 *in the full graph*. This has three defects:
- Only enforces branching at position 0. Steps 1 … k−1 can all be forced.
- Uses full-graph degree, not DAG-out-degree. A start node of full-graph degree 5 whose neighbors mostly lead *away* from `v` still has DAG-out-degree 1 — trivially refutable.
- Priority-queue tie-break by `degree_choice_nodes` at [tasks.h#L92-L107](tasks.h#L92-L107) is a proxy on the same signal and inherits the same defect.

`random_tree_generator` at [undirected_graphs.h#L434](undirected_graphs.h#L434) shows the same fix applied at graph-generation time: `num_children = 2` if the root would otherwise have one child. Again, only fixes the *first* step.

**Proposed formulation.** Define, for a sampled path `p = (u = v_0, v_1, …, v_k = v)`:

- `hardness(p) = |{ i ∈ [0, k) : deg_out^{D_{u,v}}(v_i) ≥ 2 }|`

`hardness(p) ∈ [0, k]`. `hardness = k` means every hop is a genuine choice. `hardness = 0` means the entire path is forced (tree case).

Three sampling strategies, cheap → principled:

**Strategy A — DAG-weighted pair sampling (cheap, general).**
1. For each candidate `(u, v)` with `d(u, v) = k`, compute the number of shortest `u → v` paths, `N_{u,v}`, and the per-layer widths `L_i(u, v)` of `D_{u,v}`. Both fall out of one forward-BFS from `u` restricted to nodes with `d(·, v) = k − d(u, ·)` — `O(V + E)` per pair, and the "restricted" filter uses the already-precomputed APSP.
2. Score the pair by `log N_{u,v}` (or `min_i L_i(u, v)`, or a product).
3. Sample `(u, v)` weighted by score.
4. Sample the actual path by walking `D_{u,v}` layer by layer, uniform over out-neighbors at each step. Equivalently: sample uniformly among all shortest `u → v` paths.

This maximises expected `hardness(p)` and gives a well-defined distribution.

**Strategy B — hard filter (`min_choice_ratio`).**
Same as A, but reject any candidate with `hardness(p) < ⌈α · k⌉` for user-specified `α ∈ [0, 1]` (e.g., `α = 0.75`: at least 75 % of hops must be genuine choices). Fall back to Strategy A after `N` rejections to avoid infinite loops in sparse graphs.

**Strategy C — distractor-aware relaxation (for sparse graphs, especially trees).**
In sparse structures (trees, low-radius Euclidean graphs) most nodes have DAG-out-degree 1 and A/B would reject almost every candidate. Relax "genuine choice at `v_i`" to also count *non-trivial distractors*: an off-DAG neighbor `w` of `v_i` counts if the subgraph rooted at `w` (excluding backtracking) has depth ≥ `d_min`. Rationale: even if there's only one DAG-successor, if the misleading branch is deep, the model can't refute it in one step and must still reason multiple hops.

Requires per-node "distraction depth" — one BFS per off-DAG entry point, reusing APSP. `O(V + E)` per pair on top of Strategy A.

**Where it fits in the R15 design.**
- DAG construction + per-pair hardness scoring lives in `GraphSampler` as per-graph auxiliaries owned by `SharedContext` (const, shared across workers). Computed once per sampled graph, before task sampling.
- `TaskComputer::compute_shortest_path` consumes the DAG to sample paths.
- `ShortestPathTaskArgs` gains a `hardness_mode : {none, score_weight, min_choice_ratio, distractor_depth}` enum + associated params (`alpha`, `d_min`).
- The current `sample_choice_node`, the `PathPriority`/`ComparePathPriority` priority-queue logic, and the `sample_start_end` retry loop all go away.

**Interaction with `label_smoothed_path`.** Uniform sampling on the DAG produces the sampled path as *one* of many valid shortest paths. `label_smoothed_path` should enumerate the DAG (or a sample of paths through it) so the loss is fair when the model produces a different but equally-shortest path.

**Applies to.**
- `ShortestPathTask` (primary — the `sample_choice_node` mechanism lives here).
- `BFSTask` at [tasks.h#L460-L465](tasks.h#L460-L465) — same start/end sampling pattern, same defect.
- `random_tree_generator`'s "ensure start is choice node" hack at [undirected_graphs.h#L434](undirected_graphs.h#L434) — graph-time analogue of the same problem. Trees have DAG-out-degree 1 at *every* step (there is only one `u → v` path in a tree, so the DAG is a line), so `hardness(p) = 0` structurally for any path in any tree. The `num_children = 2` root-widen hack does not raise DAG-out-degree — it only adds a distractor sibling; the on-path child is still unique. See I9-Clarification for the full analysis and per-family knobs.
- Any future k-hop-flavoured task (label-propagation, cycle-finding, etc.).

**Cost.**
- DAG construction + `N_{u,v}` + layer widths per pair: ~40 lines (one restricted BFS).
- Layer-uniform DAG walk to sample a path: ~20 lines.
- Distractor-depth precompute (Strategy C only): ~30 lines.
- Removes: `sample_choice_node`, `PathPriority`, `ComparePathPriority`, the ad-hoc retry loop in `sample_start_end`, and the `num_children ≥ 2` hack in `random_tree_generator`.
- Net: ~+90 lines new, ~-60 lines removed, and a well-defined hardness metric that can be *reported per-batch* (see I7) so training curves can be conditioned on true difficulty, not sampled path length.

**Priority.** Medium-high. This is a correctness issue for the k-hop claim, not just a nicety — reporting "trained on 20-hop paths" when average `hardness` is 3 misrepresents the experiment. Worth doing during R15 so the DAG-scoring infrastructure lands with the new stage classes rather than being retrofitted.

#### I9-Clarification: efficient sampling of *close-to-truly-k-hop* paths within a fixed graph

**Design invariant.** Once a graph is drawn from its family, it is immutable — no edges may be added or removed to control task difficulty. Paths are sampled *from* the fixed graph. **Rejecting an entire graph and drawing a fresh one is still permitted** (the accepted graph remains an unmodified family sample); *modifying* an accepted graph is not.

**Framing.** There is no way to guarantee `hardness(p) = k` on an arbitrary graph — hardness is topology-bounded. Trees have `H(G, k) := max_{u,v,p} hardness(p) = 0` for any `k ≥ 1` (unique u→v path ⇒ DAG-out-degree 1 everywhere). Very sparse ER has `H(G, k) ≪ k` for typical pairs. So the honest question is: **how efficiently can we sample paths that get *close to* `H(G, k)`?** And when they don't, how efficiently can we know we tried?

##### Building block: one BFS per target gives everything

For any node `v`, one BFS from `v` (cost `O(V + E)`) produces, in one pass:

- `d(x, v)` for every `x`.
- `dag_out_deg(x, v) := |{y ∈ N(x) : d(y, v) = d(x, v) − 1}|` — DAG-out-degree of `x` in the shortest-path DAG *toward* `v`. Computed in one linear pass over edges after the BFS levels are set.
- `N_v(x)` := number of shortest `x → v` paths, via the layer-count recurrence `N_v(x) = Σ_{y ∈ shortest-successors} N_v(y)` with `N_v(v) = 1`. Also one linear pass.

Given these three arrays, sampling a shortest `u → v` path from any `u` at distance `k` is trivial: at each step from `x`, pick a shortest-successor `y` uniformly from `{y ∈ N(x) : d(y, v) = d(x, v) − 1}`. Cost `O(k)`. Realised `hardness(p)` is observed during the walk: count positions where `dag_out_deg(v_i, v) ≥ 2`. No extra cost.

This is the whole machinery. Every scheme below is a different amortisation strategy over the same primitive.

##### Efficient schemes

| Scheme | Precompute per graph | Per-item cost | Hardness guarantee |
|---|---|---|---|
| **S1** single-BFS-per-item | 0 | `O(V + E)` | Best-effort. Reports realised hardness. |
| **S2** batched-BFS (`M` targets) | `O(M(V + E))` | `O(k)` amortised for large batch | Best-effort with varied `v`. |
| **S3** full precompute | `O(V(V + E))` = `O(V·E)` for sparse graphs | `O(k)` | Best-effort optimal — every pair scored. |
| **S4** item rejection (threshold `H₀`) | 0 or as above | `O((V+E) / α)` where `α = Pr[hardness ≥ H₀]` | ≥ `H₀`, if `H(G, k) ≥ H₀`. Bounded by `max_item_retries`. |
| **S5** graph rejection (threshold `H₀`) | `O(V(V+E))` per graph attempt | 0 additional | ≥ `H₀`, always. Unbounded expected time on families with `Pr[H(G,k) ≥ H₀] = 0` (trees). |

**S1** (single-BFS-per-item):

```
for each item:
    Sample v uniformly.
    BFS from v; compute d, dag_out_deg, N_v.
    candidates_u = {x : d(x, v) = k}. If empty, resample v.
    Sample u from candidates_u weighted by N_v(u).  // more shortest paths ≈ more branching
    Walk shortest path u → v (uniform on DAG-successors).
    Record realised hardness.
```
Simple, no cache. Fine when items per graph is 1 or 2.

**S3** (full precompute) is the R10-friendly version: run BFS from every node during graph precompute, store `dag_out_deg` and `N_v` matrices. Then item sampling is `O(k)`. For `V = 1000`, `E ≈ 10⁴`, precompute is `~10⁷` ops (~10 ms); for `V = 10⁴`, `E ≈ 10⁵`, ~10⁹ ops (~1 s). Amortises well over hundreds of items per graph.

**S2** is the practical middle ground for typical batch sizes: pick `M = O(√B)` random targets per graph, do `M` BFSes, then sample items across them. Balances precompute vs per-item cost.

##### Ranking / weighting candidate pairs

Given the primitives, we have several proxies for "which `(u, v)` will yield high hardness":

- **`log N_v(u)`** — number of shortest paths from `u` to `v`. More paths ⇒ wider DAG ⇒ more branching per position. Best cheap proxy, one number per pair.
- **Layer-width vector** `(L_0, L_1, …, L_k)` where `L_i = |{x : d(u, x) = i ∧ d(x, v) = k − i}|` — width of DAG at each position along the path. `min_i L_i` is a hard lower bound on per-step branching. Falls out of the same BFS but requires materialising per-position counts.
- **Per-path realised hardness** — walk the path first, then decide whether to keep. Used by S4.

For efficiency: `log N_v(u)` is a single scalar per pair, computed in the BFS pass; use as the sampling weight. Layer-width vector is used only by S4's post-hoc filter.

##### What breaks and where

**Trees** (`H(G, k) = 0`). Every scheme above still terminates: S1/S2/S3 return `realised hardness = 0` and honestly report it. S4 with `H₀ ≥ 1` and `max_item_retries` finite gives up gracefully; with `max_item_retries = ∞` it loops. S5 with `H₀ ≥ 1` never accepts. Configuration must guard against this: refuse `H₀ ≥ 1` on tree kinds at config-parse time.

**Very sparse ER** (`p ≈ 1/n`, near connectivity threshold). `H(G, k)` is low for typical pairs. S1/S2/S3 give what's available. S4 acceptance rate is low; use `max_item_retries` sensibly. S5 has low graph-acceptance rate ⇒ blows up in expected time.

**Radius / Delaunay** at moderate radius/density. `H(G, k)` is modest — a few genuine choices per path. S1/S2/S3 work. S4 acceptance depends on threshold. S5 costly.

**Grids, dense ER, SBM, random regular, hypercube**. `H(G, k)` is high. All schemes work; S4 with a threshold is efficient because acceptance is high; S5 is bounded and fine.

##### Efficiency verdict

- **Sampling the hardest available path (best-effort):** efficient. Cost `O(V + E)` per item (S1) or `O(V(V + E))` per graph plus `O(k)` per item (S3). Well-defined, deterministic modulo the RNG, no rejection loops, honestly reports realised hardness. Recommended default.
- **Rejecting individual items below a hardness threshold (S4):** efficient *if* the graph natively supports the threshold. Bounded by `max_item_retries` regardless — never loops.
- **Rejecting entire graphs to hit a hardness threshold (S5):** *not* efficient in general. Expected cost is `1 / graph-acceptance-rate`, which can be zero on families that structurally cannot meet the threshold. Only viable when the family's `H(G, k)` distribution is known to typically clear the threshold. Reserve for narrow experiments where distributional bias is acceptable and the family is known-competent.
- **Guaranteeing target hardness on any graph:** *impossible* under the invariant. This is a topological ceiling, not an implementation gap.

##### Concrete design under the invariant

- **`ShortestPathTaskArgs`** gains:
  - `path_hardness_scheme : {none, dag_weighted, layer_min, per_item_reject, graph_reject}` — dispatch enum for S1–S5 above (default `dag_weighted` = S1/S3).
  - `path_hardness_target : optional<int>` — advisory only. If unreachable, sampler picks closest achievable; realised value reported in I7 stats.
  - `min_hardness_reject_threshold : optional<int>` + `max_item_retries : int = 10` — enables S4.
  - `min_hardness_graph_reject : optional<int>` + `max_graph_retries : int = 0` — enables S5 (default off).
- **`SharedContext`** (via R10) caches the S3 tables (`d`, `dag_out_deg`, `N_v` per source) alongside APSP. `GraphSampler` computes them once when the graph is precomputed.
- **`TaskComputer::compute_shortest_path`** consumes the cached tables to walk paths in `O(k)` per item.
- **I7 batch stats** report per-item realised hardness, per-batch mean/median/std, per-batch item-reject count, per-batch graph-reject count. Downstream analysis conditions on realised hardness, not nominal `k`.
- **Removed:** `ShortestPathTask::sample_choice_node`, `PathPriority`, `ComparePathPriority`, the ad-hoc retry loop in `sample_start_end` at [tasks.h#L120](tasks.h#L120), and the `num_children = 2` root-widen hack in `random_tree_generator` at [undirected_graphs.h#L434](undirected_graphs.h#L434) (ineffective — one wide layer cannot raise tree hardness above 0).
- **Config validation:** reject `path_hardness_target ≥ 1` (or `min_hardness_*_threshold ≥ 1`) on tree kinds at parse time. Trees structurally cannot support hardness ≥ 1; failing loudly at config beats an infinite retry loop at runtime.

##### Cost summary

- Per-graph precompute (S3): `O(V·E)` — one BFS per node, plus one linear pass for `dag_out_deg` and `N_v` per source. Fits in R10's precompute phase.
- Per-item sampling: `O(k)` — one walk on the cached DAG.
- Memory: `O(V²)` for the per-source arrays (already present in APSP); one extra `int` per pair for `N_v` (or `float` for `log N_v` to avoid overflow at large `V`).
- Item-level rejection (S4): expected cost per accepted item is `O(k / α)` where `α = Pr[realised hardness ≥ H₀]`, bounded by `max_item_retries`.
- Graph-level rejection (S5): expected cost per accepted graph is `O(V·E / β)` where `β = Pr_family[H(G, k) ≥ H₀]`. Unbounded if `β = 0` (trees, at any `H₀ ≥ 1`).

##### What we cannot do (and shouldn't pretend to)

- Force hardness above `H(G, k)` on a fixed graph. Topologically impossible.
- Get high hardness on trees. Structurally impossible for k ≥ 1.
- Get *uniform* hardness across families without a distributional footnote. S5 gives uniform hardness above threshold but the accepted graph subpopulation is no longer an unbiased family sample. Cross-family comparison should happen at analysis time (filter/reweight to a common realised-hardness stratum), not at sampling time.

**Bottom line.** The efficient sampling problem is *solved* if we accept "close-to-truly-k-hop" as "best-available within the fixed graph, with realised hardness reported." The `O(V·E)` per-graph precompute plus `O(k)` per-item cost is the fundamental price and it is small. What is *not* efficiently solvable — and not solvable at all under the invariant — is forcing target hardness on families that don't support it. That's a graph-choice problem, not a path-sampling problem.

**What this replaces.** The earlier "topology-independent path sampling / path-first construction" material is withdrawn. It violated the invariant by generating a spine DAG and decorating it with family-flavoured distractors — the graph was constructed to match the target hardness, not sampled independently.

### I10. Task owns the path; Scratchpad consumes it (kill the reconstruct-and-back-write anti-pattern)

**The problem.** In V1, the pipeline's nominal dataflow is `Task → Scratchpad` (Task defines the query and answer; Scratchpad produces the traversal trace that leads to that answer). The actual dataflow for BFS/DFS tasks is the opposite: the Task samples only `(start, end)`, the Scratchpad runs the traversal *and reconstructs a path*, then the Task copies that path back out.

**Concrete evidence in V1.**
- `BFSScratchPad::BFSScratchPad(..., start, end)` at [scratch_pads.h#L56-L143](scratch_pads.h#L56-L143) runs BFS from `start` to `end`, then walks backwards through `levels` (L130-L142) to reconstruct `path` — a field that the scratchpad's own `tokenize()` never reads.
- `DFSScratchPad(start, end, ...)` at [scratch_pads.h#L665-L700](scratch_pads.h#L665-L700) does the same: DFS-until-found, then reconstructs `path` from `dfs_steps` (L689-L697).
- `BFSTask::BFSTask(...)` at [tasks.h#L446-L485](tasks.h#L446-L485) samples `(start, end)` via `ShortestPathTask::sample_start_end`, constructs the scratchpad (L475), then copies `scratchpad->path` back into `this->path` (L476).
- `ShortestPathTask::set_path` at [tasks.h#L285-L295](tasks.h#L285-L295) exists purely to accept a path *from* the scratchpad. Its comment — *"some scratchpads may generate their own paths, and we need to respect that"* — is a written admission of the anti-pattern.

**Why this is bad.**
- **Inverts pipeline order.** `TaskComputer` runs before `ScratchpadBuilder` in R15's Approach D as originally written (which is why I10 was needed at all); Task's `path` field isn't final until after the Scratchpad has been built. Under **I10-Followup** the two stages are merged into one call, so this ordering problem becomes even more impossible — both Task and Scratchpad are produced together and neither can hold a value that depends on the other's not-yet-existent output.
- **Splits the "ground truth" definition.** For `BFSTask`, the answer path is whatever the BFS traversal-and-reconstruct happens to pick. It is not the same code path as `ShortestPathTask::path`, so two tasks that should share a definition of "the shortest path for this `(u, v)`" don't. Label smoothing (I9's `label_smoothed_path`) has to reach through the scratchpad to work.
- **Recomputes work already available.** APSP is cached in `SharedContext` under R10. Reconstructing a shortest path from cached predecessors is `O(k)`. Reconstructing it inside the scratchpad by walking backwards through `levels` is also `O(k)` but requires *first* running the full BFS `O(V+E)`, and the reconstruction path is a distinct code path from every other place that walks APSP.
- **Couples correctness to traversal.** If someone swaps in a variant BFS ordering for the scratchpad (say, to test a different tokenization), they may silently change which shortest path the Task reports as ground truth.

**The fix under R15 + R10 + I9-Clarification.**

The rule is one sentence: **`Task` owns `(start, end, path, label_smoothed_path)`. `Scratchpad` never writes to any of them.**

Under R15's Approach D:

1. `TaskComputer::run(TaskKind::ShortestPath | BFS | DFS, graph, rng, cfg)` produces a `Task` with:
   - `start`, `end` — sampled by whatever scheme the task kind uses (uniform, `IntRangeSampler`-weighted by `d(u,v)`, hardness-aware per I9-Clarification schemes S1–S5).
   - `path` — reconstructed in `O(k)` by walking APSP predecessors cached in `SharedContext` (R10). One codepath, shared by all shortest-path-flavoured tasks.
   - `label_smoothed_path` — computed from the cached shortest-path DAG (same one I9-Clarification uses for hardness scoring). Also one codepath.
2. The scratchpad-construction branch inside `TaskComputer::run` (per I10-Followup; formerly `ScratchpadBuilder::run(ScratchpadKind::BFS | DFS, graph, task, rng, cfg)` before that merge) produces a `Scratchpad` that:
   - Reads `task.start` and `task.end` for the traversal endpoints.
   - Runs BFS or DFS purely for the *trace* (BFS `levels`, DFS `steps`).
   - Does **not** have a `path` field. Anything that used to read `scratchpad.path` now reads `task.path`.
3. `Task::set_path` (V1's back-write) is deleted. The comment "some scratchpads may generate their own paths, and we need to respect that" goes away — no scratchpad generates a path anymore.
4. DFS's actual traversal *walk* (the order the DFS visited nodes) is semantically different from the task's shortest path — it may be longer, may visit non-path nodes, etc. That trace stays on `Scratchpad` as `dfs_visit_order` (or similar, name TBD) — a field that is *not* trying to be a shortest path. This preserves the semantic distinction without pretending it's the same thing as `task.path`.

**What this replaces.**
- `BFSScratchPad::path` field and the L130-L142 reconstruction block: deleted.
- `DFSScratchPad::path` field and the L689-L697 reconstruction block: deleted (the DFS's traversal order, if needed for tokenization, becomes its own field with a distinct name).
- `BFSTask` L475-L476 back-write: deleted.
- `ShortestPathTask::set_path` at [tasks.h#L285](tasks.h#L285): deleted.
- The `sample_path = false` flag on `ShortestPathTask` (a workaround for the case where the scratchpad supplies the path): deleted.

**Cost.**
- **Removes** one duplicate `O(k)` path-reconstruction codepath per BFS/DFS task and the ad-hoc back-write plumbing (~40 lines across `tasks.h` and `scratch_pads.h`).
- **Adds** one `reconstruct_shortest_path(u, v)` helper on `SharedContext` that walks cached APSP predecessors. `O(k)`. ~15 lines.
- **Net:** less code, single source of truth for `path`, pipeline order is real.

**Dependencies.**
- **R10** (SharedContext caches APSP) — required. Without cached predecessors, the Task would have to run its own BFS/Dijkstra, defeating the win. If R10 is deferred, I10 can still land by having `TaskComputer` compute predecessors on-demand and pass them via `SharedContext`'s scratch area — same interface, cost bumped to `O(V+E)` per task instead of `O(k)`.
- **I9-Clarification** (path-hardness schemes consume cached DAG) — reinforces I10. Both improvements assume the same "compute per-source BFS-DAG once at graph-precompute time, consume it O(k) per task" building block. Landing them together shares implementation.

**Order of operations.** Land I10 with (or immediately after) R15's task/scratchpad split. Doing it any later means writing the scratchpad's reconstruction code twice (once for the naive port, once for the fix). Doing it earlier is impossible because V1's `Task` / `Scratchpad` are so tightly coupled through inheritance that the back-write pattern is load-bearing.

#### I10-Followup: merge `ScratchpadBuilder` into `TaskComputer` (task-scoped scratchpad production)

**The question.** Since scratchpads are task-dependent, should the Scratchpad just be owned by / part of the Task?

**The two possible merges.** "Owned by the Task" is ambiguous — there are two independent design axes and they should be answered separately:

- **Merge the data classes** — Scratchpad becomes a field on `Task` (`task.scratchpad`).
- **Merge the stage classes** — `Scratchpad` remains its own data class, but its construction moves into `TaskComputer::run` and the standalone `ScratchpadBuilder` stage class is deleted.

##### Data merge (C): rejected

Making `Scratchpad` a field on `Task` is *worse* than the R15-as-planned split:

- **Asymmetric fit.** V1's `BFSTask` always has exactly one scratchpad kind (`BFSScratchPad`), but `ShortestPathTask` can pair with `{None, BFS, DFS}` scratchpads chosen by config. Merging forces every `Task` to carry a `std::optional<Scratchpad>` or a `std::variant`, which is dead weight for the task/no-scratchpad combinations and forces `Task` to know about every scratchpad kind.
- **Category confusion.** `label_smoothed_path` (I9) is a task-level concept — it describes the answer, not the model's shown work. Making it live alongside `task.scratchpad.levels` on the same object encourages the same "scratchpad writes back into task" leak that I10 was fixing.
- **No downstream saving.** `Tokenizer::tokenize_into_row(graph, task, scratchpad, layout, cfg, out_row)` still needs both pieces as distinct inputs (they're tokenized into different regions of the output row). Merging them into one struct just means the tokenizer immediately pulls them apart again.
- **Rejection ordering regresses.** Under Approach D, rejections happen during precompute. If `Task` construction produces the (potentially large) scratchpad in one step, a task-only rejection check can't fire until after the scratchpad is built. Keeping the two productions separable inside `TaskComputer` (see below) preserves the option to reject cheaply.

##### Stage-class merge (B): accepted — supersedes the ScratchpadBuilder in R15's stage list

Delete `ScratchpadBuilder` as a stage class. `TaskComputer::run(task_kind, scratchpad_kind, graph, rng, cfg)` produces both a `Task` and a `Scratchpad` in one call, since both are task-scoped by nature and both require the same inputs (graph, task_kind, rng, cfg — the only extra dimension is `scratchpad_kind`, which the same class can dispatch on).

**Revised R15 stage-class list (three stages, not four):**

```
├── GraphSampler                 .run(GraphKind, rng, cfg) → SampledGraph
├── TaskComputer                 .run(TaskKind, ScratchpadKind, graph, rng, cfg) → (Task, Scratchpad)
│       (Layout is computed via Layout::from(graph, task, sp, cfg) — no stage class)
└── Tokenizer                    .tokenize_into_row(graph, task, sp, layout, cfg, out_row)
```

**Revised `Worker::generate_batch` precompute loop:**

```cpp
while (static_cast<int>(precomputed.size()) < cfg.batch_size
       && attempts < cfg.max_attempts) {
    SampledGraph g = graph_sampler_.run(cfg.graph_kind, gen_, cfg);
    if (!g.passes_attempt_check(cfg)) { ++attempts; continue; }
    auto [t, sp]   = task_computer_.run(cfg.task_kind, cfg.scratchpad_kind, g.graph(), gen_, cfg);
    Layout ly      = Layout::from(g, t, sp, cfg);
    precomputed.push_back({std::move(g), std::move(t), std::move(sp), ly});
    ++attempts;
}
```

`Task` and `Scratchpad` are returned as a `std::pair` (or a small `TaskComputeResult { Task task; Scratchpad scratchpad; }`), remaining independently addressable data classes so `PrecomputedItem` / `Tokenizer` see the same peer-fields structure as before.

##### Why this dominates the four-stage R15 plan

- **Kills I10's anti-pattern by construction.** With Task and Scratchpad produced inside the same call, there's no cross-stage back-write to inhibit. `task.path` is computed from cached APSP predecessors *first*, then the scratchpad's traversal reads `task.start` / `task.end` — no scratchpad ever needs to reconstruct a path. I10's *rule* ("Task owns `(start, end, path, label_smoothed_path)`; Scratchpad never writes to any of them") still applies; the *cost analysis* simplifies because the "back-write plumbing" line item drops to zero.
- **One fewer stage class, one fewer file.** `scratchpad_builder.h` and its class disappear; `task_computer.h` grows two extra private methods (`build_bfs_scratchpad`, `build_dfs_scratchpad`) and one extra switch on `ScratchpadKind`.
- **Rejection stays fast.** `TaskComputer::run` can (and should) construct the `Task` first, run the attempt-check, and only proceed to scratchpad construction if the task is accepted. This is a cheaper reject than the current R15 plan (which builds the Task via `TaskComputer`, returns to `Worker`, checks, and only then calls `ScratchpadBuilder`).
- **Task-side and scratchpad-side logic for the same task kind live together.** Adding a new task kind is one enum entry + one method on `TaskComputer` (which internally decides how to build any compatible scratchpad kinds), rather than one entry + one method on `TaskComputer` + one entry + one method on `ScratchpadBuilder` with the two staying in sync.

##### Trade-offs to acknowledge

- **`TaskComputer` grows.** It now dispatches on both `TaskKind` and `ScratchpadKind`. Since the valid `(task_kind, scratchpad_kind)` combinations are already asymmetric in V1, this is honest — the class becomes the single point that knows which combinations are legal, rather than distributing that knowledge across two classes and a config validator.
- **The name `TaskComputer` becomes slightly wider than the pure computation of Task**. If this is jarring, rename to something like `TaskAndScratchpadComputer`, `TaskStage`, or `TaskProducer`. The class's job is still "everything task-scoped in the precompute phase," which is a coherent responsibility.

##### Where this leaves I10 proper

I10's technical content (Task is the single source of truth for `path` / `label_smoothed_path`; Scratchpad has no `path` field; DFS's visit-order lives on the Scratchpad under a distinct name; `set_path` deleted) stays exactly as written — it describes an invariant on the *data classes*, which is independent of how many stage classes produce them. I10-Followup only changes the *stage-class count* (4 → 3) and the *entry-point signature* (two `run(...)` calls → one `run(...)` returning a pair). Land them together.

---

## Monitored execution plan — step-by-step refactor sequence

**Purpose.** The "Suggested refactor order" section above groups the work into four coarse phases. This section is the finer-grained sequence to actually execute against — designed so that after every numbered step the tree builds, the (updated) test suite passes, and the change can be reviewed / rolled back in isolation. The user requested this ordering explicitly: **config first, then the C++ generator entry point, then step into the pipeline stages one at a time.**

**Contract for every step.**

- **Ends green.** Full build + `pytest` (or doctest) suite passes at the end of the step. If it doesn't, the step is too big — split it.
- **Cutover, not coexistence.** V1 is *not* kept in the tree for side-by-side comparison. It is deleted wholesale in Step 2, and every later step works only against the new pipeline. If you want to spot-check numeric output against the V1 submission, use `git worktree add ../graphgen-v1 v1.0-submission` in a sibling directory and run V1 there — no `V2` / `_new` / `_legacy` suffixes ever appear in the tree.
- **One reviewable change.** Each step is a candidate commit (or a small stack of commits). If a step's "Touches" list spans more than ~3 files and 200 lines, split it. (Step 2 is the intentional exception: it's a single mechanical deletion pass + empty-skeleton install, which reads more cleanly as one commit than as a chain.)
- **Verification is a real command, not a vibe.** Every step names the exact test/script that proves it works. If no test exists yet, writing it is the *first* sub-task of the step.

**Prerequisites** (Phase 0 of the "Suggested refactor order"): **R6** `GG_CHECK`/`GG_ASSERT`, **R7** test-harness scaffolding (doctest + pytest wiring; there is no in-tree V1 to diff against, so R7 in this plan means "real tests, invariant checks, and analytic ground truth" rather than "reference-vs-fast pair"), **R13** tooling — CMake, doctest, clang-format, sanitizers. Land them before Step 1.

**On numeric parity with V1.** Where possible, verification uses *invariants* (e.g., every emitted path is a valid path in its graph; trees have `hardness == 0`; grid shortest-path lengths equal Manhattan distance) and *analytic ground truth*, not seeded byte-for-byte agreement with V1. Seeded byte-for-byte agreement is likely impossible anyway — R15's Migration-cost section already documents that RNG advance order will differ once precompute and compute phases are separated. When you want a numeric sanity check against V1, do it manually from a `v1.0-submission` git worktree; do not build formal golden-fixture infrastructure unless a specific test needs it.

### Step 1 — Build-system migration + `GeneratorConfig`

Two logically separate concerns bundled into one step under the "one commit per step" cadence — the user may split them into commits **1a** (build system) and **1b** (config type) for cleaner review. Both are strictly additive; V1 keeps working end-to-end after Step 1.

#### Step 1a — CMake + scikit-build-core

**Goal.** Replace the ad-hoc `install.sh` + `setup.py` build chain with **CMake** (as the C++ builder) + **scikit-build-core** (as the Python build backend that invokes CMake for `pip install .`). Zero C++ source changes — the same V1 sources still build to the same `generator${EXT_SUFFIX}` module and pass the same `import generator` smoke test.

**Locked-in choices** (see `/memories/repo/graphgen-refactor.md`):

- C++ standard: `-std=c++20` + vendored **Kokkos `<mdspan>` reference header** (single-header, C++17+) rather than `-std=c++23`'s native `std::mdspan`. Bumps to C++23 later if the reference impl causes friction.
- `v1.0-submission` git tag exists — no need to create it.

**Touches.**

- New: `CMakeLists.txt` at repo root (finds Python + pybind11 via CMake's `Python_add_library` / `pybind11_add_module`, sets `-std=c++20 -O3 -DNDEBUG -fPIC`, `-undefined dynamic_lookup` on macOS, still builds `generator` from the current V1 sources with Boost as an `IMPORTED` target).
- New: `third_party/mdspan/` — vendored Kokkos mdspan reference header (single file; git-submodule or plain copy — plain copy is simpler for now).
- Rewritten: `pyproject.toml` — declares `[build-system] requires = ["scikit-build-core", "pybind11"]` and `build-backend = "scikit_build_core.build"`; adds `[tool.scikit-build]` block pointing at the root `CMakeLists.txt`.
- Rewritten: `install.sh` — becomes a ~10-line shim: pick Python interpreter (same logic as V1), run `"${PYTHON}" -m pip install -e .`, done. scikit-build-core drives CMake internally.
- Modified: `get_generator_module.py::build_module` (~L806) — replace the hardcoded `g++ ... V1 header list ... -o generator${EXT_SUFFIX}` invocation with `subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)`. Update the freshness check to compare the `.so`'s mtime against everything under `include/graphgen/`, `src/`, and `CMakeLists.txt` instead of the hardcoded V1 file list. **This is the load-bearing edit** — the user explicitly requires the Python auto-compile path to keep working.
- Deleted: `setup.py` (replaced by scikit-build-core reading `pyproject.toml`).

**Verify.**

- Fresh clone works: `rm -rf .venv build generator*.so && uv venv && bash install.sh && python -c "import generator; print(generator)"`.
- Auto-compile path works: `rm generator*.so && python -c "from get_generator_module import get_generator_module; g = get_generator_module(); print(g)"` — should invoke `pip install -e .` and land with a fresh `.so`.
- Every V1 pytest and Python-script call site (`main.py`, `main_test_pybind_compile.py`, `python_sample_generator.py`) still runs to completion with the same output.
- Record clean-build time to `docs/benchmarks.md` — this is the "V1-on-CMake" baseline. Should be within noise of V1-on-install.sh; if CMake configure adds >1s of overhead per rebuild, note it but don't chase.

**Do NOT touch yet.** Any C++ source. Any pybind binding. Any V1 header. This step is 100% build-system rewiring; the module's behaviour is byte-for-byte unchanged.

#### Step 1b — Add `graphgen::GeneratorConfig`

**Goal.** Add the `GeneratorConfig` type and its pybind binding as a new, standalone artifact. Does not modify any V1 code path — V1 keeps its positional `*_n` signatures untouched until Step 2 deletes them wholesale. Step 1b is deliberately additive so that the Step 2 cutover can `#include` a stable header.

**Touches.** New: `include/graphgen/generator_config.h`, `src/generator_config.cpp`, plus a pybind block appended to `generator.cpp` that exposes `graphgen::GeneratorConfig` as `generator.GeneratorConfig`. Modified: `CMakeLists.txt` to include the new `.cpp` in the module target. No V1 files edited.

**Design choices to lock in here.** Option 1 (`std::optional<T>` per task-specific field) from R1 — simplest and matches V1's already-flat `Args`. Namespace: `graphgen::GeneratorConfig` (introduces the `graphgen::` namespace, R12's namespace bullet). `[[nodiscard]]` on `GeneratorConfig::validate()`. Use C++20 designated initializers in the test file to prove the aggregate shape.

**Verify.**

- New unit test: `tests/test_generator_config.cpp` (doctest) — round-trips `py::kwargs → GeneratorConfig → GeneratorConfig::to_dict()` and checks every field; `validate()` accepts every legal `(task_kind, populated fields)` combination and rejects every illegal one.
- Existing V1 `pytest` suite still passes byte-for-byte — nothing V1 was touched.
- `python -c "from generator import GeneratorConfig; help(GeneratorConfig)"` shows the fields (validates pybind exposure).

**Do NOT touch yet.** Stage classes, `Worker`, `SharedContext`, algorithms, task/scratchpad types, any V1 code.

### Step 2 — Cutover: delete V1, stand up the empty new pipeline

**Goal.** Single mechanical commit that (a) deletes every V1 code path and (b) replaces it with the *shape* of the post-refactor top-level entry point — `graphgen::Worker::generate_batch(cfg) → py::dict` — whose body is a stub that raises `GG_CHECK(false, "graph kind X not yet ported (Step N)")` for every kind. Nothing behaviourally works after Step 2 except `import generator` and the `GeneratorConfig` round-trip from Step 1. That's the intended state; Steps 3–12 fill it in.

**Why one commit.** Trying to sequence "delete V1" and "install skeleton" separately means either V1 sits in the tree next to a half-built successor (the coexistence pattern the user explicitly rejected) or the module doesn't build for a step. Doing both together is a clean mechanical diff: minus V1 files, plus new skeleton files, plus a rewritten `generator.cpp` pybind block. The commit is large but the review is easy — nothing subtle happens, it's mostly `git rm` + boilerplate.

**Delete.** `main.cpp`, `generator.cpp` V1 pybind block, `args.h`, `instance.h`, `tasks.h`, `scratch_pads.h`, `undirected_graphs.h`, `directed_graphs.h`, `graph_wrapper.h`, `graph_tokenizer.h`, V1's `dictionaries.h`/`matrix.h`/`utils.h` if their contents don't survive the refactor (audit and keep what does), every `#include <boost/graph/...>` site, `install_mac_boost.sh`. The `refactor` branch's `git log` still contains all of it; `git checkout v1.0-submission` is the recovery path if anything V1-only turns out to be worth resurrecting.

**Install.** New: `include/graphgen/worker.h`, `include/graphgen/batch_output_arrays.h`, `include/graphgen/shared_context.h`, `src/worker.cpp`. Rewritten: `generator.cpp` — now a thin ~50-line file that binds only `graphgen::GeneratorConfig`, `graphgen::SharedContext`, `graphgen::Worker`.

**Skeleton shape.**

```cpp
// worker.h
namespace graphgen {
class Worker {
public:
    Worker(std::shared_ptr<const SharedContext> ctx, uint64_t seed);
    py::dict generate_batch(const GeneratorConfig& cfg);  // stub: GG_CHECK(false, ...)
private:
    std::shared_ptr<const SharedContext> ctx_;
    std::mt19937_64 gen_;
    // Stage-class members added in Steps 4, 6, 8:
    // GraphSampler  graph_sampler_;
    // TaskComputer  task_computer_;
    // Tokenizer     tokenizer_;
};
} // namespace graphgen
```

**Verify.**

- Build succeeds; `import generator` works; `generator.GeneratorConfig` round-trips (Step 1's test still green).
- `generator.Worker(ctx, 0).generate_batch(cfg)` raises the `GG_CHECK` message for every legal `graph_kind`, with the kind name in the message (regression fence against silent no-ops).
- `grep -r 'boost/graph' .` returns nothing under source directories.
- Build time from clean drops toward the R5-Level-2 target (~5s vs the ~45s baseline documented in Section 3/R5). Record the number; it's the new baseline.

**Do NOT touch yet.** GIL release (Step 11), threading (Step 12), any real algorithm.

### Step 3 — `SharedContext` populated

**Goal.** Move module-level globals (dictionaries, config caches previously in `dictionaries.h`) into `SharedContext`. `Worker` now holds a `shared_ptr<const SharedContext>` populated with real content and constructed with a seed. `generate_batch` still raises for every kind — no algorithm yet — but per-worker RNG + shared const data are now real.

**Touches.** `include/graphgen/shared_context.h` (populated with what survived the Step 2 audit), `src/shared_context.cpp`, `src/worker.cpp`, `generator.cpp` (`py::class_<SharedContext>` + factory function `graphgen::make_shared_context()`).

**Verify.**

- `test_shared_context_lifetime.py` — construct one `SharedContext`, hand it to N `Worker` instances, destroy the context reference on the Python side, confirm workers still work (shared_ptr keeps it alive).
- Sanitizer run (`clang++ -fsanitize=address,undefined`) on the doctest suite passes.

### Step 4 — `GraphSampler` skeleton (all kinds throw `unimplemented`)

**Goal.** Add the `GraphSampler` class with its `.run(GraphKind, rng, cfg) → SampledGraph` dispatcher. Every private `sample_<kind>` method is a stub that calls `GG_CHECK(false, "sample_<kind> not implemented (Step 5+)")`. `Worker::generate_batch` gains one call to `graph_sampler_.run(...)`; its own top-level `GG_CHECK` from Step 2 is moved down past the sampler call so the sampler's more specific message wins.

`SampledGraph` and `CsrGraph` are defined as complete types here, since the sampler's return type needs them. `CsrGraph::from_directed_edges` / `CsrGraph::from_undirected_edges` implemented and unit-tested independently. No BFS / Dijkstra / APSP yet — those come with the first real graph kind.

**Touches.** New: `include/graphgen/csr_graph.h`, `include/graphgen/sampled_graph.h`, `include/graphgen/graph_sampler.h`, `src/graph_sampler.cpp`, `src/csr_graph.cpp`. Modified: `src/worker.cpp`.

**Verify.**

- `test_csr_graph.cpp` (doctest) — round-trip edge lists through both constructors, check degrees, neighbour iteration order, undirected symmetry.
- `test_graph_sampler_skeleton.cpp` — `sampler.run(GraphKind::ErdosRenyi, rng, cfg)` triggers a `GG_CHECK` failure with the kind name in the message.

### Step 5 — Port `GraphSampler::sample_erdos_renyi` + `CsrGraph` algorithms it needs

**Goal.** First real graph kind. Erdős–Rényi because it's the simplest. This is also where `CsrGraph::bfs` / `distance_bounded_bfs` / `connected_components` land, since ER is the natural first algorithm consumer.

**Touches.** `src/graph_sampler.cpp` (fills `sample_erdos_renyi`), `include/graphgen/csr_graph.h` + `src/csr_graph.cpp` (adds `bfs`, `distance_bounded_bfs`, `connected_components`), corresponding doctest files.

**Verify.**

- `test_csr_bfs.cpp` — BFS on hand-constructed CSR graphs matches expected distances; distance-bounded BFS terminates at the bound; unreachable pairs return the sentinel.
- `test_er_sampler.py` — for a matrix of `(n, edge_prob, seed)`, the returned graph has the right vertex count, edge count is within tolerance of the ER expectation `n(n−1)p/2`, and `connected_components` returns a plausible component count. **Invariant-based, not V1-seeded.** If you want to eyeball against V1, do it manually from a `v1.0-submission` worktree.

### Step 6 — `TaskComputer` skeleton + `compute_shortest_path` for ER (with `Scratchpad = None`)

**Goal.** Adopt the I10-Followup shape — `TaskComputer::run(TaskKind, ScratchpadKind, graph, rng, cfg) → (Task, Scratchpad)` — for exactly one combination: `(ShortestPath, None)` on an ER graph. Every other combination hits an `unimplemented` `GG_CHECK`.

`Task` and `Scratchpad` land as flat data classes per R15. `Task` owns `(start, end, path, label_smoothed_path)` per I10; `Scratchpad` in the `None` case is an empty variant alternative.

**Touches.** New: `include/graphgen/task.h`, `include/graphgen/scratchpad.h`, `include/graphgen/task_computer.h`, `src/task_computer.cpp`. Modified: `src/worker.cpp` (chains `graph_sampler_.run(...)` into `task_computer_.run(...)`). `include/graphgen/csr_graph.h` (adds `all_pairs_distances` / `distance_bounded_all_pairs_distances` — needed for path construction).

**Verify.**

- `test_task_computer_shortest_path.cpp` — on hand-constructed CSR graphs, `compute_shortest_path` returns a `Task` whose `path` is a valid shortest path from `start` to `end` (every consecutive pair is an edge; length matches `all_pairs_distances(start, end)`) and whose `label_smoothed_path` matches I9's formula. Trees have `hardness == 0` (regression fence for the I9-Clarification fix).
- `test_er_shortest_path.py` — for random ER seeds, every produced `Task` satisfies the path-validity invariant, and rejection-rate stays within a documented bound.

### Step 7 — `Layout::from(...)` + `BatchOutputArrays` sized from real layout

**Goal.** Replace Step 2's stub sizing with the real Tier 3E Approach D layout: precompute the per-item shape from the produced `SampledGraph` / `Task` / `Scratchpad`, allocate `BatchOutputArrays` from the batch's max, and return a dict whose shapes are correct even though the tensors are still zero-filled (tokenizer isn't wired yet).

**Touches.** New: `include/graphgen/layout.h`, `src/layout.cpp`. Modified: `include/graphgen/batch_output_arrays.h` (constructor takes `Layout`), `src/worker.cpp` (does the precompute loop with rejection handling, then allocates).

**Verify.**

- `test_batch_layout_shapes.py` — for a matrix of `(graph_kind=ER, batch_size, min/max_num_nodes)` configs, `Worker(...).generate_batch(cfg)`'s output dict has the expected key set and per-key shape (batch axis = `batch_size`, sequence axis = the batch's actual `max_seq_len_this_batch` ≤ `cfg.max_seq_len`).

### Step 8 — `Tokenizer::tokenize_into_row` for `(ER, ShortestPath, None)`

**Goal.** End-to-end pipeline lights up for the one supported combination. `Worker::generate_batch` now: precompute → allocate → tokenize → return. Output tensors contain real token values for `(ER, ShortestPath, None)`.

**Touches.** New: `include/graphgen/tokenizer.h`, `src/tokenizer.cpp`. Modified: `src/worker.cpp` (final loop calls `tokenizer_.tokenize_into_row(...)` per row). `RowView` type finalised (`std::mdspan`-based per R12's std::mdspan bullet and the `BatchOutputArrays` section).

**Verify.**

- `test_er_shortest_path_end_to_end.py` — for a matrix of seeds and configs, output tensors satisfy the token-level invariants: `tokenized_targets` decodes back to the same `Task.path` the precompute produced; every token in `tokenized_inputs` is a valid vocabulary index; padding regions past the actual sequence length hold the pad token, nothing else.
- Bench: single-thread `generate_batch` at V1's default batch size beats or matches V1's throughput (run V1 once in a `v1.0-submission` worktree to capture the baseline; commit the number to `docs/benchmarks.md`). Regression fence for later steps.
- **Optional manual spot-check.** If a real numeric-parity check is wanted, run the same seed through V1 in a git worktree and diff the two output dicts by hand. Not a required CI test; do it once and move on.

### Step 9 — Fill out remaining `(task, scratchpad)` combinations for ER

**Goal.** Extend `TaskComputer` and `Tokenizer` to cover every `(TaskKind, ScratchpadKind)` combination worth carrying forward from V1 on ER. Each combination is one sub-step, each with its own invariant tests. Order by rising complexity: `(BFS, None)` → `(BFS, BFSScratchpad)` → `(ShortestPath, BFSScratchpad)` → `(ShortestPath, DFSScratchpad)` → `(CenterCentroid, ...)`.

Each sub-step is a mini-Step-8: add one private method to `TaskComputer` (and possibly `Tokenizer`), add its invariant tests, ship. No touching other graph kinds yet.

**Verify.** Per-combination invariant tests all green. Bench per combination.

### Step 10 — Additional graph kinds, one at a time

**Goal.** Port the remaining graph kinds in order: path-star → balanced → Euclidean → random-tree. Each kind is a Step-5-through-Step-9 replay scoped to that kind. Because `CsrGraph` and all its algorithms are already written and shared, each kind is *only* a new `sample_<kind>` method + whatever new `(task, scratchpad)` combinations that kind unlocks.

`khops` is separately structured per R8 (different data shape, own pipeline surface `Worker::generate_khops_batch`) — treat it as its own subsequence at the end.

**Verify.** Full invariant-test matrix for each kind's supported `(task, scratchpad)` combinations. Analytic-ground-truth tests where available (Euclidean grids have known Manhattan-distance ground truth per Section 2C — use them). Bench per kind against the Step-8 baseline.

### Step 11 — GIL release around the compute loop (Tier 4A)

**Goal.** Wrap `Worker::generate_batch`'s post-precompute compute loop in `py::gil_scoped_release`. Precompute stays under the GIL. Tokenize releases.

**Touches.** `src/worker.cpp` only. ~5 lines.

**Verify.**

- Every test from Steps 5–10 still green (behaviour unchanged).
- Bench: throughput under concurrent Python-side work (e.g., a DataLoader with `num_workers > 1` calling `generate_batch`) is measurably higher than pre-Step-11.
- ThreadSanitizer run on the doctest suite is clean.

### Step 12 — Multi-worker parallelism (Tier 4B)

**Goal.** Enable `PyTorch DataLoader(num_workers > 0)` correctness: `Worker` construction happens inside `worker_init_fn`, per-worker RNGs are derived via `derive_worker_seed(master_seed, worker_id)`, `SharedContext` is `const`-shared across workers via `shared_ptr<const>`. Follows R10's Migration-cost gotcha (do not construct `Worker` in `Dataset.__init__`).

**Touches.** `graph_datasets.py` update (worker_init_fn pattern documented in R10). `src/worker.cpp` (nothing new — the design was ready since Step 3).

**Verify.**

- `test_dataloader_multiprocess.py` — `DataLoader(num_workers=4)` produces deterministic, seed-reproducible, non-duplicated batches across workers.
- Throughput bench scales sub-linearly but meaningfully with `num_workers` (record numbers as the new bench baseline in `docs/benchmarks.md`).

### Post-Step-12: Phase 4 opportunistic tuning

After Step 12 the codebase is in its post-refactor shape. Tuning items from Phase 4 of "Suggested refactor order" (Tier 3A flat matrices, Tier 3C `flat_hash_map`, Tier 2C ER-via-union-find, Tier 2D squared-Euclidean, R4 PCH, R9 documented invariants) are safe to pursue **only when the profiler says they matter**, guarded by the pytest + doctest suite plus the benchmark baselines recorded in Steps 8 and 12. Order by profiler evidence, one item per commit, benchmark before/after.

### Hop-distance matrix (landed mid-refactor)

Documenting the design as-implemented so future changes preserve the invariants.

**Type + naming.** `graphgen::HopDistView` is a `md::mdspan<int32_t, dextents<int, 2>, layout_stride>` (`md` routes to `std` under C++23 when `__cpp_lib_mdspan >= 202207L`, else to the vendored Kokkos reference impl — see `include/graphgen/mdspan_shim.h`). The type name is `Hop*` rather than a generic `Distance*` because a `WeightedDistanceMatrix` will slot in alongside it later; the two matrices are semantically independent (a weighted graph can still be asked for hop counts). Sentinels: `HOP_UNREACHABLE = -1` inside the valid `(n_i, n_i)` region, `HOP_PAD = INT32_MIN` outside it.

**Accessor on `SampledGraph`.** `sg.hop_distances()` is lazy: first call runs `n` single-source BFS and caches, subsequent calls hit the cache. Storage is either `hop_owned_` (contiguous `n*n int32_t`) or an externally attached view into a `(B, max_n, max_n)` batch tensor. `sg.attach_hop_distances_view(v)` is call-once-before-first-use with runtime rejection of double-attach and too-small views. `sg.release_hop_distances()` drops the owned buffer but is a no-op on attached views (the caller owns that buffer's lifetime). Consumers always get a `(n_i, n_i)` `HopDistView` — the batch-tensor padding is invisible.

**Batch return.** `cfg.return_hop_distances` (default `false`) toggles the `"hop_distances"` key in `generate_batch`'s dict. When set, the worker allocates one `py::array_t<int32_t>` of shape `(batch_size, cfg.max_num_nodes, cfg.max_num_nodes)` pre-filled with `HOP_PAD`, attaches per-row views to each item's `SampledGraph` before the task pass, and after tasks calls `hop_distances()` on any item whose task did not touch it (e.g. `shortest_path` uses its cheap 2-BFS pattern; the post-task fill guarantees the returned tensor is populated). When the flag is off, tasks that need distances (Center/Centroid) still fill an owned buffer that gets released between the task and tokenize stages. Two storage modes, one consumer-facing view type; the flag is the only visible knob.

**Clipping without submdspan.** `std::submdspan` is P2630 / C++26 and libc++ 18 doesn't ship it. `sampled_graph.cpp` clips manually by constructing a fresh `layout_stride::mapping` with smaller extents but the same strides and data pointer. Zero portability surface added.

### Future work: weighted-distance matrix

Deferred until weighted graphs land. When they do, follow the hop-matrix template exactly:

- New header `weighted_distance_matrix.h` with `WeightedDistView = md::mdspan<double, dextents<int, 2>, layout_stride>`, sentinels `WEIGHT_UNREACHABLE = std::numeric_limits<double>::infinity()`, `WEIGHT_PAD = NaN` (or another out-of-band value).
- New accessors on `SampledGraph`: `weighted_distances()`, `attach_weighted_distances_view(v)`, `release_weighted_distances()`. Independent from the hop accessors — a weighted graph may still expose both matrices simultaneously.
- Fill via Dijkstra on `CsrGraph` (needs an edge-weight vector added to CSR).
- New config flag `return_weighted_distances`. Once weighted graphs work, lift the current `return_hop_distances && weighted` blanket-reject in `generator_config.cpp`.
- Batch return + view-attach + release pipeline mirrors `return_hop_distances`.

Keep both matrices' lifecycles orthogonal: the hop matrix's compute + release is unaffected by the weighted matrix's, and vice versa. Both flags default `false`; both allocations only happen on-demand.

### If a step fails a test

The correct response is *not* to weaken the test. In order:

1. **Confirm the divergence is real.** Re-run; check for uninitialised memory (sanitizer); check for `unordered_map` iteration order dependence (frequent divergence source when writing new algorithms).
2. **Check against invariants first, V1 second.** Ask "is the emitted output valid on its own terms?" (path is a valid path, distances satisfy the triangle inequality, etc.) before asking "does it match V1?". Most bugs surface as invariant violations, and invariant tests catch them without needing V1 in scope at all.
3. **If numeric parity with V1 is what's disagreeing** and everything above passes — accept it. R15's Migration-cost section already predicts seeded byte-for-byte parity will not hold once precompute and compute are separated. Downgrade any test that assumed parity to an invariant / distributional test, and document the divergence at the test's top comment.
4. **If the divergence is algorithmic** (BFS returns different distances on a hand-constructed graph, task path is not a valid path, etc.) — that's a real bug; do not proceed to the next step until it's fixed.

The whole point of the step-by-step ordering is that a failing test at step N implicates only step N's changes. Preserve that property by keeping each step small.
