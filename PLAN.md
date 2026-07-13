# GraphGen C++ Speedup Plan

Prioritized speedup plan based on inspection of `generator.cpp`, `graph_wrapper.h`, `undirected_graphs.h`, `instance.h`, and `matrix.h`. Ordered by expected impact per effort. No code changes made yet.

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

## 3. Data structures & allocation churn

### 3A. Drop `vector<vector<int>>` everywhere it's used as a matrix

`distances_ptr`, `graph_ground_truths_ptr`, `node_ranks_ptr` in `graph_wrapper.h` are all `unique_ptr<vector<vector<int>>>`. Every row is a separate heap allocation and iteration touches non-contiguous memory.

You already have `Matrix<int>` in the codebase — use it (or a plain `vector<int>` of size `N*M` accessed as `data[i*M + j]`). Impact: fewer allocations, better cache behavior, roughly a 1.5–3× speedup on the shortest-paths/ground-truth phases in my experience.

### 3B. Use `boost::vecS` (or `boost::hash_setS`) instead of `boost::setS` for edges

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

**Approach A — Two-pass with size prescan (best memory usage, more refactor):**
- **Pass 1 (sizing):** For each of the `batch_size` accepted instances, run only the parts of generation that determine layout (sample `num_nodes`, generate the graph, decide task, compute query / graph / task / scratch-pad / thinking-token lengths). Keep a lightweight per-instance record of these sizes. Handle rejections here (regenerate until accepted; only accepted instances contribute to the max).
- Compute `max_seq_len`, `max_struct_dim`, `max_labels`, etc. across the accepted set.
- **Allocate the final numpy arrays once** at `[B, max_seq_len, max_struct_dim]` etc. via `py::array_t<int>({B, max_seq_len, max_struct_dim})` and get raw mutable pointers.
- **Pass 2 (fill):** For each instance, run the expensive tokenization phase writing straight into its slice of the numpy buffer.

**Approach B — Preallocate to a hard upper bound (simplest, wastes memory):**
- Use existing arguments (`max_num_nodes`, `max_edges`, `num_thinking_tokens`, max query length, etc.) to compute a *theoretical* upper bound on each output dim.
- Allocate `[B, max_theoretical_seq, max_struct]` once.
- Every instance writes into its slot; unused tail is left as pad tokens.
- Simplest to implement; only viable if the upper bound isn't wildly larger than typical (otherwise you'll allocate 10× more memory than you use and hurt cache behavior).

**Approach C — Hybrid, size prescan without regenerating (recommended):**
- Do a cheap sizing pass that generates only the graph and picks task/scratchpad, computing lengths but *not* tokenizing.
- Allocate final tensors from the true max.
- Then tokenize each instance directly into its slice.
- Similar wins to A but avoids the "regenerate on rejection during sizing" bookkeeping — rejections are cheap because sizing didn't do the expensive tokenization.

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

Recommendation: **do Tier 3E (approach C) instead of 3D**. It subsumes 3D's win (no per-instance allocations) and adds the "one copy, not two" win, and it's a prerequisite for the biggest gains in Tier 4. Skip 3D unless 3E is out of scope for the sprint.

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
- `boost::sorted_erdos_renyi_iterator` with `setS`: consider generating edges into a `vector<pair<int,int>>` and constructing the graph in one shot with the iterator pair (already what you do), but on `vecS` edges — much faster.
- `pybind11` array construction in `is_in_validation` / `is_in_test`: use `py::array_t<bool>({n})` and write via `mutable_unchecked<1>()` — you're already close, but the `arr[py::make_tuple(py::ellipsis())] = false` init in `is_invalid_example` is a Python-side op; use `std::fill_n` on the raw buffer.

## Suggested order

1. Turn on profiling (30 minutes).
2. Tier 1 flags (5 minutes; ~10–20%).
3. Tier 2A (Johnson → bounded BFS behind a dispatch layer) (~half a day; often 3–10× on the shortest-paths phase).
4. Tier 3E (batch-first preallocated tensors, approach C) — big architectural change but subsumes 3D and unlocks the parallelism wins in 4B. If done first, 4A becomes cheap.
5. Tier 3B (`hash_setS` → `vecS` after audit) — do alongside 3A (flat matrices) if picking up 3A separately.
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

**Same logic extends to `Task`/`ScratchPad` themselves** — those hierarchies exist to model "different tasks do different work". R15 replaces them with `compute_task(TaskKind, ...)` and `build_scratchpad(ScratchpadKind, ...)` free functions. The config struct kills the **argument** hierarchies; R15 kills the **runtime object** hierarchies. Same principle, applied at two layers.

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

**Why superseded:** R15 collapses the whole `Args`/`Instance`/`Task`/`ScratchPad` inheritance chain into plain-data structs + free functions dispatched by enum (R11). No factory registry, no virtual dispatch, no strategy interface — a fresh new sampler is one free function + one enum entry. Strictly less machinery for the same win.

## R3. ~~Phase-separate `Instance`~~ **superseded by R15**

*Original suggestion below; kept for historical context.*

> `Instance<D>`'s constructor mixes graph construction, distance computation, task selection, scratchpad selection, tokenization, and packaging into one call. That makes each phase hard to test in isolation and hard to debug when the output is wrong.
>
> - Split into explicit phases: `build_graph`, `compute_distances_or_ranks`, `select_task`, `select_scratchpad`, `tokenize`, `pack_into_batch_slot`. Each takes the previous phase's output and returns its own.
> - Each phase gets a unit test (see R7). Debugging becomes: "which phase's output first went wrong?" — often findable in minutes instead of hours.
> - Bonus: the phase boundaries are exactly where the Tier 3E prescan pass would split (sizing phases vs fill phases), so this refactor and 3E are complementary, not competing.

**Why superseded:** R15 achieves phase separation as a *natural consequence* of deleting `Instance` entirely. Each phase becomes a free function returning a plain-data struct; the `Worker::generate_batch` method is those functions called in sequence. No class to phase-separate because there's no class.

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
3. Update pybind bindings: `py::class_<SharedContext>`, `py::class_<Worker>`, keep the top-level module functions as thin wrappers around a "default worker" for backward compatibility during migration. Delete the wrappers once Python callers are migrated.
4. Change your Python `DataLoader` `worker_init_fn` to construct one `Worker` per thread.

Wins across the three criteria:

- **Flexibility:** multiple contexts coexist (different dictionaries, different seeds). Perfect for A/B experiments in one process. Multiple workers per context is the *native* mode, not an afterthought.
- **Debuggability:** every test creates a fresh `Worker` (and typically a fresh `SharedContext`) — no shared state, no test-ordering coupling. Bug reports become "here's the `(master_seed, worker_id)` that repro's it" — one line.
- **Speed:** parallel batch fill (Tier 4B) becomes trivial and correct. `const` sharing needs no locks; per-worker RNG needs no locks. The GIL release from Tier 4A gets its full parallelism benefit.

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
- **`std::mdspan` (C++23) or a small custom equivalent** for multi-dim views on flat buffers. Cleaner than `data[i * cols + j]` and just as fast.
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

## R15. Flatten the class hierarchy — collapse `Args`/`Instance`/`Task`/`ScratchPad` into pipeline of free functions over plain data

**This is the biggest architectural improvement in the plan.** It supersedes R2 and R3.

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

Everything reduces to **plain-data structs** connected by **free functions**. No inheritance anywhere in the pipeline. The whole thing on top of R10's `Worker` becomes:

```cpp
py::dict Worker::generate_batch(GraphKind kind, const GeneratorConfig& cfg) {
    BatchOutputArrays out = allocate_output_arrays(cfg);   // Tier 3E preallocation
    for (int i = 0; i < cfg.batch_size; ++i) {
        SampledGraph graph      = sample_graph(kind, gen_, cfg, *ctx_);
        TaskAnswers  answers    = compute_task(cfg.task_kind, graph, gen_, cfg);
        Scratchpad   scratchpad = build_scratchpad(cfg.scratchpad_kind, graph, answers, cfg);
        tokenize_into_row(graph, answers, scratchpad, cfg, *ctx_, out.row(i));
    }
    return out.to_pydict();
}
```

**Depth from `generate_batch` to the actual work: one function call.** That's the entire target.

What gets deleted:

| Deleted | Replaced by | Class count change |
| --- | --- | --- |
| `Args`, `TaskArgs`, `ShortestPathTaskArgs`, `BFSTaskArgs`, `CenterCentroidTaskArgs`, `KhopsArgs`, `ScratchpadArgs`, `BFSScratchpadArgs`, `TokenizationArgs`, `PosArgs` (10 classes, several with inheritance) | One flat `GeneratorConfig` struct (R1) with `std::optional`/`std::variant` per-task fields; `TaskKind`/`GraphKind`/`ScratchpadKind` enums (R11) | 10 → 1 |
| `GraphWrapper<D>` (owns Boost graph + 6 vectors + methods) | `SampledGraph` (plain struct: CSR arrays from R5 Level 2, node_list, edge_list, shuffle_map, optional positions) | 1 → 1 (but flat) |
| `Task`, `ShortestPathTask`, `BFSTask`, `CenterTask`, `KHopsTask`, `KHopsGenTask` (6 classes) | `TaskAnswers` plain struct + `compute_task()` free function dispatched by `TaskKind` (R11) | 6 → 1 struct + 1 function |
| `ScratchPad`, `BFSScratchPad`, `DFSScratchPad` (3 classes with virtual methods) | `Scratchpad` plain struct (tagged with `ScratchpadKind`) + `build_scratchpad()` free function | 3 → 1 struct + 1 function |
| `GraphTokenizer` (owns tokenization state per instance) | `tokenize_into_row()` free function that writes directly into pre-allocated batch tensors | 1 → 1 function |
| `Instance<D>` (owns everything above) | *nothing — deleted entirely* | 1 → 0 |
| `BatchedInstances<D>` (owns `vector<Instance>` + packaging logic) | `BatchOutputArrays` (thin wrapper over the preallocated `py::array_t`s from Tier 3E) + `Worker::generate_batch()` | 1 → 1 (thin) |

**Net: ~22 classes → ~5 plain-data structs + a handful of free functions.**

### Why free functions over methods on `Worker`

Making `sample_graph`, `compute_task`, etc. free functions (rather than methods on `Worker`) has three payoffs:

1. **Testability.** A free function has no hidden state — its signature *is* its full contract. Every function gets a unit test in ~5 lines: `auto g = sample_erdos_renyi(rng, cfg); check_invariants(g);`. Compare to testing an `Instance` method today, which requires constructing the full `Args` object, dictionaries, etc.
2. **Reusability.** `compute_task` can be called from CLI tools, from tests, from Python via a separate binding — without instantiating a `Worker`. The current `Task*` objects can't be used outside of an `Instance`.
3. **Clear dependencies.** `sample_graph(kind, rng, cfg, ctx)` visibly depends on exactly `(kind, rng, cfg, ctx)`. `Instance` method dependencies are hidden inside `this->args`, `this->graph`, `this->args.tok`, etc. When something goes wrong, tracing the actual inputs takes seconds instead of minutes.

The `Worker` class is still useful — it holds the per-thread RNG and the `shared_ptr<const SharedContext>`. But it's a *thin container*, not a place where logic lives. Ideally `Worker`'s only methods are `generate_batch(GraphKind, config)` and simple accessors.

### What about polymorphism? Isn't inheritance sometimes right?

Inheritance is right when you have an open set of types that will be extended by *code you don't control*. That's not the case here — every task, scratchpad, and graph kind lives in this codebase and is added by you. For a closed set of variants:

- **Enum + switch** is faster than virtual dispatch (branch prediction beats indirect call), compiler-checked for exhaustiveness, and generates zero heap allocations.
- **`std::variant` + `std::visit`** (from R11) is right when different variants carry different data payloads. Compiler still checks exhaustiveness.

Use a class only when it *owns* something (like `Worker` owns an RNG, `SharedContext` owns dictionaries). Never use it to model "a kind of task" — that's what enums are for.

### Wins across the three criteria

- **Flexibility.** Adding a new task is: (1) add a `TaskKind::MyNewTask` enum entry, (2) add a case in `compute_task()`'s switch, (3) done. No new class, no new file mandatory, no registration boilerplate. Compiler complains at every switch you didn't update — no silent runtime fallthroughs like today's `if (task_type == "my_new_task")` chain.
- **Debuggability.** Every function is a pure function of its inputs. A failing test produces a stack trace with exactly the right nesting depth — one frame per pipeline stage, not five. Ownership is trivial: `SampledGraph` owns its arrays and dies at end-of-scope; no `unique_ptr` chains, no polymorphic destructors to trace through.
- **Speed.** No per-item heap allocation for pipeline objects (only the graph *data* allocates, and that's ideally per-batch reused per Tier 3E approach C). No virtual calls in the hot loop. The whole batch iteration is inlinable by the compiler because there are no polymorphic barriers. Free functions on plain data are what the CPU cache and branch predictor were designed for.

### Migration cost

High — this is the biggest single refactor in the plan. But it's mostly **mechanical** once the `SharedContext`/`Worker` split (R10) and the `GeneratorConfig` struct (R1) are in place:

1. Introduce `SampledGraph`, `TaskAnswers`, `Scratchpad`, `BatchOutputArrays` as plain structs.
2. Port one graph kind end-to-end (Erdős–Rényi is simplest): write `sample_erdos_renyi`, `compute_bfs`, `build_bfs_scratchpad`, `tokenize_into_row`. Add one `Worker::generate_batch(GraphKind::ErdosRenyi, cfg)` method that calls them.
3. Verify against the old code path on fixed seeds using R7's reference-vs-fast test infrastructure.
4. Port the rest one at a time, deleting the corresponding old `*_n` function each time.
5. Once all callers have migrated, delete `Args`, `Instance`, `BatchedInstances`, `Task`, `ScratchPad`, `GraphWrapper` (or reduce `GraphWrapper` to the CSR-only `SampledGraph` if R5 Level 2 is done).

R6 (assertions) and R7 (reference/fast test pairs) are prerequisites; without them this refactor can't be verified. That's why they come first in the suggested order.

## Suggested refactor order (interleaved with the perf work)

1. **R6 + R7 first**, before any perf changes. Assertions and reference-vs-fast tests are what let you refactor everything else without regressing correctness. Especially critical for R15. ~1 day.
2. **R13 (tooling) alongside R6/R7** — clang-format, clang-tidy config, doctest wired in, sanitizers on the debug build. Once, then compounding benefit forever. ~half a day.
3. **R1 (config struct + pybind class + auto-`.pyi`)** — replaces the whole `Args`/`TaskArgs`/... inheritance chain with one flat struct. Prerequisite for R15.
4. **R10 (`SharedContext` + per-thread `Worker`)** paired with R1. Both touch the pybind boundary; do them together so callers migrate once.
5. **R11 (enum classes + `std::variant`)** immediately after R10. Prerequisite for R15's enum-dispatched free functions.
6. **R12 (modern C++ hygiene)** as an opportunistic cleanup while doing R1/R10/R11 — you're touching those files anyway. Drop `using namespace std;`, wrap in `namespace graphgen`, switch sentinels to `std::optional`.
7. **R15 (flatten the pipeline)** — the main event. With R1, R10, R11 in place, this is mechanical porting: one graph kind at a time, deleting the corresponding `*_n` function as you go. ~R7 verifies each step. This subsumes what R2 and R3 were describing.
8. **Tier 3E (batch-first preallocation)** falls out of R15's `tokenize_into_row(..., out.row(i))` design — R15 already wrote the pipeline in the shape that 3E needs.
9. **R8 (khops vs regular graph split)** during R15 — set up two separate `Worker::generate_khops_batch(...)` entry points rather than trying to unify them under one `GraphKind` enum, per the design note.
10. **R4 (header split) + PCH** whenever compile times start hurting, or when moving to CMake in R13.
11. **R5 (Boost isolation, then possibly replacement)** — Level 1 alongside Tier 2A; Level 2 for the big compile-time / debuggability win (also simplifies R15's `SampledGraph` to plain arrays).
12. **R9** as an opportunistic documentation pass at the end of each refactor step.
13. **~~R2, R3~~** — superseded by R15; skip.
14. **~~R14~~** — considered and rejected; keep the pipeline in C++. See R14 for the reasoning.
