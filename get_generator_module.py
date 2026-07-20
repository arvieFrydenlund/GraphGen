import os
import sys
import pydoc
import numpy as np

try:
    import networkx as nx
except ImportError as e:
    print(f"NetworkX is not installed or broken. {e}")
    nx = None

"""
Python-side interface to the C++ `generator` module.

Contents:
  * :func:`get_generator_module` -- lazy import + rebuild of the
    editable-installed C++ extension.
  * :class:`GraphPlotter` -- render a single graph + task instance
    for debugging.
  * :func:`pprint_batch` + :func:`pprint_distance_matrix`
    + :func:`pprint_distance_ranks` -- column-aligned batch
    renderer, semantic-agnostic distance matrix printer, and
    per-source distance-rank table printer for
    ``batch['distance_rank_targets']``.
  * :data:`SECTION_KEYS` + ``section_*`` helpers -- slice per-row
    or padded batched tensors out of a ``Worker.generate_batch()``
    dict.
"""


#############################
# Extra Generator Functions #
#############################

class GraphPlotter:
    """Render a single graph + task instance for inspection / debugging.

    Given an edge list (with vertex ids ALREADY demangled from token ids
    to their symbolic form) plus optional query / target sets, computes
    a layout, colours vertices by role, and draws with networkx +
    matplotlib.

    Vertex colouring uses TWO independent channels:
      * fill    (``node_color``)   -- default / query / target
      * outline (``edgecolors``)   -- default / query / target /
                                       alt (red, for shortest-path
                                       label-smoothing alternatives)

    A target vertex on an alternative shortest-path hop is green-fill
    + red-outline; a center vertex that also sits in the query is
    green-fill + purple-outline. This preserves the label-smoothing
    signal that a single-channel colour scheme would flatten.

    Extraction is out of scope: give this class already-symbolic
    edges + role sets. A typical debug loop pulls per-row data with
    ``section_tokens(batch, section, row)`` and inverts
    ``ctx.token_dict`` before handing the results here.
    """

    # ---- Palette --------------------------------------------------------
    DEFAULT_COLOUR = '#1f78b4'   # matplotlib tab:blue
    QUERY_COLOUR   = 'purple'
    TARGET_COLOUR  = 'green'
    ALT_COLOUR     = 'red'

    # Graph kinds whose default layout is a rooted tree (Graphviz
    # `twopi` radial or `dot` LR hierarchical). Everything else falls
    # through to `neato` force-directed.
    _TREE_KINDS = frozenset({
        'path_star', 'balanced', 'random_tree',
    })

    def __init__(self, edges, *,
                 graph_kind,
                 directed,
                 task_kind=None,
                 query=None,
                 task_targets=None,
                 pos=None,
                 root=None):
        """
        Args:
          edges: iterable of ``(u, v)`` pairs of symbolic vertex ids.
                 Already demangled from token ids -- this class is
                 display-only, not extraction.
          graph_kind: string from the generator's graph kinds. Drives
                 the default layout choice (see :data:`_TREE_KINDS`);
                 does NOT determine directedness -- pass that
                 explicitly via `directed`.
          directed: whether to build a ``nx.DiGraph`` (True) or
                 ``nx.Graph`` (False). Mirrors ``cfg.directed`` --
                 the plotter has no independent opinion about which
                 kinds are directed; each sampler enforces its own
                 constraint (e.g. path_star requires directed=True,
                 euclidean requires False), and the caller passes
                 through whatever the batch was generated with.
          task_kind: string from the generator's task kinds, or
                 ``None`` / ``'none'`` to skip role colouring
                 entirely.
          query: iterable of vertex ids appearing in the query, or
                 ``None``.
          task_targets: ``list[list[symbol]]``. Outer index is the
                 generation step; inner list is the alternatives at
                 that step with position 0 = CHOSEN, positions > 0 =
                 label-smoothing alternatives. Matches the shape you
                 get by gathering ``batch['targets']`` along the
                 label-smoothing axis and stripping pad. For
                 center / centroid pass a single-step list, e.g.
                 ``[[c0, c1, c2]]``; the plotter flattens across
                 steps for those task kinds.
          pos: optional ``{vertex_id: (x, y)}`` layout override.
                 When provided, no layout engine is run.
          root: for tree layouts, the root vertex passed to
                 Graphviz. Defaults to ``task_targets[0][0]`` on
                 shortest_path tasks (path start), else ``None``.
        """
        self.graph_kind   = graph_kind
        self.directed     = bool(directed)
        self.task_kind    = task_kind
        self.query        = set(query) if query is not None else set()
        self.task_targets = task_targets
        self.pos          = dict(pos) if pos is not None else None

        # Build graph up front so subsequent passes can iterate G.
        self.G = nx.DiGraph() if self.directed else nx.Graph()
        self.G.add_edges_from(edges)

        # Role sets + rank map (for shortest_path alt-outlining).
        (self._target_set,
         self._target_rank) = self._extract_targets(task_targets)

        # Root for tree layouts. Old behaviour: path start on
        # shortest_path, else nothing.
        if root is not None:
            self.root = root
        elif self.task_kind in ('shortest_path', 'path') \
                and task_targets and task_targets[0]:
            self.root = task_targets[0][0]
        else:
            self.root = None

        # Colour vectors, one entry per node in G.nodes() iteration
        # order. Populated by _compute_colours(); left as scalars if
        # role colouring is disabled.
        self.node_color      = self.DEFAULT_COLOUR
        self.node_edge_color = self.DEFAULT_COLOUR
        self._compute_colours()

    # ---- Public API -----------------------------------------------------

    def is_directed(self):
        return self.directed

    def plot(self, *, ax=None, node_size=200, with_labels=True,
             trees_to_left=False, spring_k=1.5, spring_scale=1.5,
             verbose=False, show=False, **draw_kwargs):
        """Draw the graph.

        Layout is computed on first call (Graphviz preferred, spring
        fallback). The matplotlib backend is NOT set here -- pick
        one at the top of your script if you need something specific
        (e.g. ``matplotlib.use('Qt5Agg')``).

        Args:
          ax: existing matplotlib axes to draw on; a fresh figure is
              created when ``None``.
          node_size: passed to ``nx.draw``.
          with_labels: passed to ``nx.draw``.
          trees_to_left: for tree kinds, use ``dot`` with
              ``rankdir=LR`` instead of the default radial
              (``twopi``).
          spring_k, spring_scale: spring-layout parameters used only
              when the Graphviz fallback fires.
          verbose: print a note when the Graphviz layout fails.
          show: call ``plt.show()`` after drawing. Off by default
              because most callers want to compose several axes
              before showing.
          **draw_kwargs: forwarded to ``nx.draw``.

        Returns the axes that were drawn on.
        """
        assert nx is not None, (
            "GraphPlotter.plot requires networkx: pip install networkx")
        from matplotlib import pyplot as plt

        self._ensure_layout(trees_to_left=trees_to_left,
                            spring_k=spring_k,
                            spring_scale=spring_scale,
                            verbose=verbose)
        if ax is None:
            _fig, ax = plt.subplots()

        # `edgecolors` (node outline) is a DIFFERENT kwarg from
        # `edge_color` (line colour); keep both spellings visible in
        # the call so future edits don't collapse them.
        nx.draw(self.G, ax=ax, pos=self.pos,
                with_labels=with_labels,
                node_size=node_size,
                linewidths=2,
                node_color=self.node_color,
                edgecolors=self.node_edge_color,
                **draw_kwargs)
        if show:
            plt.show()
        return ax

    def save(self, path, name='graph.png'):
        """Write the current matplotlib figure to ``path/name``.

        Creates ``path`` if missing. Must be called after :meth:`plot`
        (or with an active figure on the stack).
        """
        from matplotlib import pyplot as plt
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, name))

    # ---- Colouring ------------------------------------------------------

    def _extract_targets(self, task_targets):
        """Return ``(target_set, rank_map)``.

        Rank map is only meaningful for shortest_path (position 0 =
        chosen, positions > 0 = label-smoothing alts). For other task
        kinds it stays empty and the outline logic short-circuits.
        """
        if task_targets is None:
            return set(), {}

        target_set = set()
        rank = {}
        for step in task_targets:
            for i, node in enumerate(step):
                target_set.add(node)
                # First occurrence wins (a node that appears both as
                # chosen at one step and as an alt at another is
                # treated as chosen for outline purposes).
                rank.setdefault(node, i)
        return target_set, rank

    def _compute_colours(self):
        # No task -> uniform structural colour.
        if self.task_kind in (None, 'none', 'None') \
                or self.task_targets is None:
            return

        if self.task_kind not in ('shortest_path', 'path',
                                  'center', 'centroid'):
            raise ValueError(
                f"GraphPlotter: unsupported task_kind {self.task_kind!r}")

        node_color      = []
        node_edge_color = []
        is_shortest_path = self.task_kind in ('shortest_path', 'path')

        for v in self.G:
            in_query  = v in self.query
            in_target = v in self._target_set

            # Fill: target > query > default. A target vertex that
            # also appears in the query stays green -- target
            # dominates. Matches the old ReconstructedGraph
            # behaviour and reads as "this is what the model must
            # produce".
            if in_target:
                node_color.append(self.TARGET_COLOUR)
            elif in_query:
                node_color.append(self.QUERY_COLOUR)
            else:
                node_color.append(self.DEFAULT_COLOUR)

            # Outline: secondary channel encoding the finer role.
            if is_shortest_path and in_target:
                # Rank 0 = chosen hop; rank > 0 = valid alternative
                # (from label smoothing). Alt hops get a red ring
                # so the chosen path pops visually.
                node_edge_color.append(
                    self.TARGET_COLOUR
                    if self._target_rank.get(v, 0) == 0
                    else self.ALT_COLOUR)
            elif in_target and in_query:
                # Center / centroid: target vertex that's also in
                # the query gets a purple ring so both roles read
                # at a glance.
                node_edge_color.append(self.QUERY_COLOUR)
            elif in_target:
                node_edge_color.append(self.TARGET_COLOUR)
            elif in_query:
                node_edge_color.append(self.QUERY_COLOUR)
            else:
                node_edge_color.append(self.DEFAULT_COLOUR)

        self.node_color      = node_color
        self.node_edge_color = node_edge_color

    # ---- Layout ---------------------------------------------------------

    def _ensure_layout(self, *, trees_to_left, spring_k, spring_scale,
                       verbose):
        if self.pos is not None:
            return
        try:
            self.pos = self._auto_layout(trees_to_left)
        except Exception as e:  # pygraphviz missing, dot not on PATH, ...
            if verbose:
                print(f"GraphPlotter: layout failed ({e}); "
                      f"falling back to spring layout")
            n = max(len(self.G.nodes), 1)
            k = (1.0 / np.sqrt(n)) * (spring_k if spring_k else 1.0)
            self.pos = nx.spring_layout(self.G, k=k, scale=spring_scale)

    def _auto_layout(self, trees_to_left):
        # path_star: bespoke radial layout. Divide the plane into
        # `num_arms` equal wedges, place the root at the origin, and
        # let each arm radiate STRAIGHT outward along its wedge angle
        # -- vertices at integer depths along a ray. Reads cleaner
        # than Graphviz `twopi` (which arcs the arms around) and
        # works without pygraphviz installed. Only applies when
        # graph_kind explicitly names the topology; anything else
        # goes through the generic layout paths below.
        if self.graph_kind == 'path_star':
            return self._path_star_layout()

        # Rooted-tree kinds: Graphviz twopi (radial) or dot (LR
        # hierarchical). Everything else: neato force-directed.
        #
        # Fix vs old code: the LR branch used to be a plain string
        # (`'-Grankdir=LR -Groot={self.root}'`) not an f-string, so
        # `-Groot` never actually resolved -- Graphviz picked its own
        # root. It's an f-string here.
        if self.graph_kind in self._TREE_KINDS:
            root_arg = f'-Groot={self.root}' if self.root is not None else ''
            if trees_to_left:
                args = ('-Grankdir=LR ' + root_arg).strip()
                return nx.nx_agraph.graphviz_layout(
                    self.G, prog='dot', args=args)
            return nx.nx_agraph.graphviz_layout(
                self.G, prog='twopi', args=root_arg)
        return nx.nx_agraph.graphviz_layout(self.G, prog='neato')

    def _path_star_layout(self):
        """Radial layout for a path_star (rooted directed tree of chains).

        Root at the origin; each arm i occupies angle
        ``2*pi * i / num_arms`` and its vertices sit at unit-integer
        depths along that ray. Handles the case where the plotter
        received edges from a shuffled batch (which is always) by
        rediscovering root + arms from ``self.G``'s in/out-degree
        structure -- caller doesn't have to pass anything extra.
        """
        # Root: unique vertex with in-degree 0. For a well-formed
        # path_star this is exactly vertex 0's SYMBOL id (whatever
        # random vocab id it drew). If we can't find one -- shouldn't
        # happen -- surface the assumption loudly rather than picking
        # arbitrarily.
        if not isinstance(self.G, nx.DiGraph):
            raise ValueError(
                "GraphPlotter._path_star_layout requires a directed "
                "graph (was `directed=True` passed?)")
        roots = [v for v in self.G if self.G.in_degree(v) == 0]
        if len(roots) != 1:
            raise ValueError(
                f"_path_star_layout: expected exactly one root "
                f"(in-degree 0), found {len(roots)}")
        root = roots[0]

        # Each arm: chain starting from a direct successor of root,
        # following the single out-edge until we hit a leaf. If a
        # vertex ever has multiple successors, the topology isn't a
        # pure path_star; walk stops at that vertex.
        arms = []
        for child in self.G.successors(root):
            arm = [child]
            cur = child
            while self.G.out_degree(cur) == 1:
                nxt = next(iter(self.G.successors(cur)))
                arm.append(nxt)
                cur = nxt
            arms.append(arm)

        # Radial placement. Arms in G.successors iteration order --
        # which is edge-insertion order, i.e. the shuffled order the
        # sampler produced. That's fine: the visual is still a
        # pretty star; the arm labelling is what carries "which arm
        # is which", and the plotter doesn't care.
        pos = {root: (0.0, 0.0)}
        num_arms = max(len(arms), 1)
        for i, arm in enumerate(arms):
            theta   = 2.0 * np.pi * i / num_arms
            cos_t   = float(np.cos(theta))
            sin_t   = float(np.sin(theta))
            for depth, v in enumerate(arm, start=1):
                pos[v] = (depth * cos_t, depth * sin_t)
        return pos


def pprint_distance_matrix(matrix, *, name='distances', unreachable=-1,
                           cell_width=3, formatter=None):
    """Print a single ``(n, n)`` distance matrix as a labelled grid.

    Semantic-agnostic: works for hop counts, weighted / Dijkstra
    distances, or any pairwise scalar you want to eyeball. Rows and
    columns are labelled ``v0 .. v(n-1)`` in internal-vertex-id
    order; cells equal to ``unreachable`` render as the sentinel
    string so unreachable pairs pop visually.

    Args:
      matrix: any 2D array-like of shape ``(n, n)``. Slice the batch
              tensor yourself before calling
              (e.g. ``batch['hop_distances'][b, :nn[b], :nn[b]]``).
      name: heading printed once above the grid.
      unreachable: sentinel value; matching cells render as
              ``' -1'`` (right-justified to ``cell_width``). Set to
              ``None`` to disable the sentinel branch entirely.
      cell_width: column width per numeric cell. Default 3 fits
              hop-count matrices; bump for wider floats.
      formatter: optional ``callable(value) -> str`` producing the
              cell text. Default int-formats every value with
              ``>{cell_width}d`` alignment (appropriate for hop
              counts). For weighted / float matrices pass e.g.
              ``formatter=lambda v: f'{v:>{cell_width}.2f}'``.

    Called by :func:`pprint_batch` under ``show_hop_distances=True``;
    also callable standalone from a debug session.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"pprint_distance_matrix: expected square 2D matrix, got "
            f"shape {matrix.shape}")
    n = matrix.shape[0]

    if formatter is None:
        def formatter(v):
            return f'{int(v):>{cell_width}d}'

    unreachable_cell = f'{-1:>{cell_width}d}'   # '-1' right-justified

    # Row label is `' v{i:<label_pad} '`. label_pad scales with n so
    # multi-digit ids don't collide with the header spacing.
    label_pad = max(1, len(str(max(n - 1, 0))))
    row_label_width = label_pad + 3   # ' v' + digits + ' '

    print(f'{name}  (n={n}, [0,n) x [0,n))')
    header = ' ' * row_label_width + ' '.join(
        f'v{i}'.rjust(cell_width) for i in range(n))
    print(header)
    for i in range(n):
        cells = []
        for j in range(n):
            v = matrix[i, j]
            if unreachable is not None and v == unreachable:
                cells.append(unreachable_cell)
            else:
                cells.append(formatter(v))
        print(f' v{i:<{label_pad}} ' + ' '.join(cells))
    print()


def pprint_distance_ranks(batch, ctx, indices=None, *,
                          unreachable=-1, max_width=140, title=''):
    """Print the per-source distance-rank table for each item in a batch.

    Requires ``batch['distance_rank_targets']`` (from
    ``cfg.return_distance_rank_targets=True``). For each source
    vertex ``u`` in ``[0, num_nodes[b])`` it prints one block::

        Source v{u} ({source_symbol}):
          d=0: {source_symbol}
          d=1: {tied vertex symbols at hop distance 1, in vocab-id order}
          d=2: ...
          ...

    Values are demangled through ``ctx.token_dict`` for readability.
    The source symbol at each block header is read directly from the
    tensor's own ``(u, d=0, k=0)`` slot (guaranteed to hold the
    source vertex itself), so this works whether or not
    ``cfg.return_positions`` is on -- no separate
    ``internal_to_vocab`` lookup needed.

    Long tie rows are soft-wrapped at ``max_width`` characters,
    continuation lines indented under the ``d=`` prefix.

    Args:
      batch: dict from ``Worker.generate_batch``. Must contain
             ``'distance_rank_targets'`` and ``'num_nodes'``.
      ctx: ``WorkerSharedContext`` used to generate the batch;
             ``token_dict`` is inverted here to demangle vocab ids.
      indices: which batch rows to render. ``None`` or ``-1`` = all
             rows; a positive int = first N; a list = specific rows.
      unreachable: sentinel marking pad slots (default ``-1``).
      max_width: soft wrap width for the tie-row content.
      title: optional heading printed once above the whole rendering.
    """
    if title:
        print(title)

    assert getattr(ctx, 'token_dict', None), \
        "pprint_distance_ranks: ctx.token_dict is empty; call " \
        "ctx.set_default_dictionary()"
    assert 'distance_rank_targets' in batch, (
        "pprint_distance_ranks: batch missing 'distance_rank_targets' "
        "-- did you set cfg.return_distance_rank_targets=True?")
    assert 'num_nodes' in batch, \
        "pprint_distance_ranks: batch missing 'num_nodes'"

    T  = batch['distance_rank_targets']
    nn = batch['num_nodes']
    B, _, max_dist, _ = T.shape

    if indices is None or (isinstance(indices, int) and indices < 0):
        idxs = list(range(B))
    elif isinstance(indices, int):
        idxs = list(range(min(indices, B)))
    else:
        idxs = [b for b in indices if 0 <= b < B]

    # Invert token_dict for id -> symbol; unknown ids fall back to
    # their numeric string form so nothing crashes on stray padding.
    id_to_symbol = {v: k for k, v in ctx.token_dict.items()}
    def tok_str(tok_id):
        return id_to_symbol.get(int(tok_id), str(int(tok_id)))

    sep_width = min(78, max_width)
    for b in idxs:
        n = int(nn[b])
        print('=' * sep_width)
        print(f'BATCH INDEX {b}  n={n}  max_distance={max_dist}  '
              f'(distance_rank_targets)')
        print('-' * sep_width)
        for u in range(n):
            # Source symbol lives at (u, d=0, k=0) by construction.
            src_sym = tok_str(T[b, u, 0, 0])
            print(f'Source v{u} ({src_sym}):')
            for d in range(max_dist):
                ties = T[b, u, d, :]
                real = [int(t) for t in ties if int(t) != unreachable]
                if not real:
                    continue
                sym_list = [tok_str(t) for t in real]
                # Soft-wrap the tie row so long tie groups
                # (dense-graph hubs, etc.) don't blow past max_width.
                prefix = f'  d={d}: '
                indent = ' ' * len(prefix)
                line = prefix
                for i, s in enumerate(sym_list):
                    piece = s if i == 0 else ' ' + s
                    if len(line) + len(piece) > max_width and line != prefix:
                        print(line)
                        line = indent + s
                    else:
                        line += piece
                print(line)
            print()


# --------------------------------------------------------------------------
#  Section slicing helpers
# --------------------------------------------------------------------------
#
# Every section in a Worker.generate_batch() row is a CONTIGUOUS span
# of columns. The batch dict already reports the span shape via
# `<section>_start_indices` and `<section>_lengths` (both (B,) int32),
# so the old V1-style gather-index arrays -- (B, max_len) int32 tensors
# per section, allocated per batch -- were pure overhead: they encoded
# `[s, s+1, ..., s+len-1]` for every row when a single `(s, s+len)`
# slice per row would suffice.
#
# These helpers give you both:
#   * `section_span(batch, section, row)` -- the (start, end) column
#     pair, cheap enough to compute inline every call.
#   * `section_tokens(batch, section, row, key='src_tokens')` -- a
#     numpy view into the batch tensor for that row + section, no copy.
#
# `section_tokens_padded` gives you a padded (B, max_section_len)
# tensor ready to hand to `torch.from_numpy(...).to(device)` -- this
# is the shape training / inference loops actually want, since GPUs
# consume whole batches at a time. `section_tokens` (row-wise view)
# is the debug / inspection tool: cheap, zero-copy, no padding.
#
# Sections supported (name -> (start_key, len_key) in the batch dict):
SECTION_KEYS = {
    'edges':      ('graph_edge_start_indices',  'graph_edge_lengths'),
    'query':      ('query_start_indices',       'query_lengths'),
    'thinking':   ('thinking_start_indices',    'thinking_lengths'),
    'scratchpad': ('scratchpad_start_indices',  'scratchpad_lengths'),
    'target':     ('task_start_indices',        'task_lengths'),
}


def section_span(batch, section, row):
    """Return ``(start, end)`` column indices for ``section`` in
    ``row`` of ``batch``.

    ``end`` is exclusive. Returns ``(start, start)`` (an empty span)
    when the section is absent for that row -- lets callers do
    ``if start == end: skip`` without a special-case for missing
    sections (e.g. no scratchpad when scratchpad_kind='none').
    """
    if section not in SECTION_KEYS:
        raise ValueError(
            f"section_span: unknown section {section!r}; "
            f"valid sections are {sorted(SECTION_KEYS)}")
    s_key, l_key = SECTION_KEYS[section]
    s = int(batch[s_key][row])
    l = int(batch[l_key][row])
    return s, s + l


def section_tokens(batch, section, row, key='src_tokens'):
    """Slice out the given section from ``batch[key]`` for one row.

    Returns a NumPy view (no copy). Shape:
      * SEAN token tensor: ``(length,)``
      * STAN token tensor: ``(length, struct_dim)``
      * any other 2D/3D per-row batch tensor with row index at axis 0

    An empty slice is returned when the section is absent for that row.

    Typical uses:
      edges = generator.section_tokens(batch, 'edges', b)
      # -> (u, v, EDGE, u, v, EDGE, ...) under SEAN concat mode
      target = generator.section_tokens(batch, 'target', b)
      # -> the vocab tokens the model must produce
    """
    s, e = section_span(batch, section, row)
    return batch[key][row, s:e]


def section_tokens_padded(batch, section, key='src_tokens', pad_value=None):
    """Return a padded ``(B, max_section_len [, D])`` tensor of the
    section content across ALL rows in the batch.

    This is the primary shape for training / inference: hand the
    result to ``torch.from_numpy(...).to(device)`` (zero-copy on the
    numpy side, one host->device copy) and you have a ready-to-use
    GPU batch tensor for the section. Row-wise :func:`section_tokens`
    is the debug / inspection counterpart -- use it when you want to
    look at one row without materialising padding.

    Args:
      batch: the generate_batch dict.
      section: name from :data:`SECTION_KEYS`.
      key: source tensor to slice (default ``'src_tokens'``).
      pad_value: fill for the padded tail per row. Defaults to
                 TOK_PAD (1) for token tensors, 0 otherwise.

    Returns a numpy array. Shape:
      * SEAN: ``(B, max_section_len)`` int32
      * STAN: ``(B, max_section_len, struct_dim)`` int32
    """
    tensor = batch[key]
    if pad_value is None:
        pad_value = 1 if key in ('src_tokens', 'targets', 'positions') else 0

    # Reuse the vectorised gather-index / mask builder. `pad_value=0`
    # inside the gather keeps every tail-column index in-bounds
    # (column 0 is always valid); we overwrite those positions with
    # the caller's `pad_value` via the mask after the gather. The
    # gather itself is one C-level call -- no Python row loop.
    gather_ids, mask = section_gather_ids(batch, section, pad_value=0)
    B, max_len = gather_ids.shape

    if tensor.ndim == 2:
        out = np.take_along_axis(tensor, gather_ids, axis=1)      # (B, max_len)
        out[~mask] = pad_value
    elif tensor.ndim == 3:
        # take_along_axis needs indices with the same ndim as the input;
        # broadcast the (B, max_len) column indices across the trailing
        # struct-dim so every channel is gathered from the same column.
        D = tensor.shape[2]
        gids3 = np.broadcast_to(gather_ids[:, :, None], (B, max_len, D))
        out = np.take_along_axis(tensor, gids3, axis=1)           # (B, max_len, D)
        out[~mask, :] = pad_value
    else:
        raise ValueError(
            f"section_tokens_padded: expected 2D or 3D tensor, got shape "
            f"{tensor.shape}")
    return out


def section_span_batched(batch, section):
    """Return ``(starts, ends)`` int32 arrays of shape ``(B,)`` --
    end is exclusive. When a section is empty for a row, ``end == start``.

    Useful for vectorised downstream logic that wants to build masks
    or slice with fancy indexing without a Python loop.
    """
    s_key, l_key = SECTION_KEYS[section]
    starts  = np.asarray(batch[s_key], dtype=np.int32)
    lengths = np.asarray(batch[l_key], dtype=np.int32)
    return starts, starts + lengths


def section_gather_ids(batch, section, pad_value=0):
    """Return ``(gather_indices, mask)`` for a section across the batch.

    Shape (both):
      * ``gather_indices``: ``(B, max_section_len)`` int32 --
        column index of each in-section position, or ``pad_value``
        where the section is shorter than ``max_section_len``.
      * ``mask``: ``(B, max_section_len)`` bool -- True inside the
        section, False in the padded tail.

    Feed ``gather_indices`` to ``np.take_along_axis(tensor, gather_indices, axis=1)``
    or ``torch.gather(tensor, 1, gather_indices)`` (after converting
    to a torch tensor) to pull the section into a batched
    ``(B, max_section_len, ...)`` tensor in one op -- the standard
    "fancy indexing" pattern.

    Use :func:`section_gather_ids` when you need shape-preserving
    fancy indexing that PyTorch autograd can differentiate through,
    or when the same gather is applied to several batch tensors
    (compute once, use N times). If you just need the section values
    themselves as a batched padded tensor, :func:`section_tokens_padded`
    is more direct.

    This one function replaces the V1-era per-section wrappers
    (``task_gather_ids``, ``edge_gather_ids``, ``scratchpad_gather_ids``,
    ``node_gather_ids``, ``query_gather_ids``) with a single dispatch
    on the section name.
    """
    starts, ends = section_span_batched(batch, section)  # (B,), (B,)
    B       = starts.shape[0]
    lengths = ends - starts
    max_len = int(lengths.max()) if B > 0 else 0

    # gather_indices[b, i] = starts[b] + i, clipped to pad_value once past
    # each row's actual length. Broadcasting is cheaper than a Python loop.
    positions      = np.arange(max_len, dtype=np.int32)[None, :]     # (1, max_len)
    gather_indices = starts[:, None] + positions                     # (B, max_len)
    mask           = gather_indices < ends[:, None]                  # (B, max_len)
    gather_indices = np.where(mask, gather_indices, pad_value).astype(np.int32)
    return gather_indices, mask


def pprint_batch(batch, ctx, indices=None, *,
                 show_positions=True, show_hop_distances=False,
                 max_width=140, title=''):
    """Human-readable, column-aligned rendering of a
    ``Worker.generate_batch()`` dict.

    Every annotation row (Src / Pos / Edg / Qry / Thk / Scr / Tgt /
    Lbl_k / Idx) uses the SAME fixed column width so a section's
    tokens line up VERTICALLY under the src tokens they annotate.
    Positions outside a given section stay blank on that row --
    visually revealing exactly where each section sits in the
    sequence, with what content, and where label smoothing kicks in.

    Rows are wrapped in horizontal chunks so wide sequences fit
    inside ``max_width`` characters.

    Args:
      batch: dict returned by ``Worker.generate_batch``.
      ctx: the ``WorkerSharedContext`` whose ``token_dict`` was used
           to fill this batch (i.e. the ctx the Worker was constructed
           with). Required so pprint renders with the same symbol
           table the tokenizer used -- no chance of drift.
      indices: which batch rows to render. ``None`` or ``-1`` = all
           rows; a positive int = first N; a list = specific rows.
      show_positions: print the positional-id row (default True).
      show_hop_distances: also print the (n, n) hop-distance matrix
           per item, clipped to the item's actual num_nodes (default
           False; requires ``cfg.return_hop_distances=True``).
      max_width: soft target for wrapped-line width in characters.
      title: optional heading printed once above the whole rendering.
    """
    # --- Resolve inputs and symbol table -----------------------------------
    if title:
        print(title)

    # ctx.token_dict is the sole source of truth. If the caller
    # generated the batch with this ctx (the only sensible flow),
    # every id in src / targets is guaranteed to have an entry.
    assert getattr(ctx, 'token_dict', None), \
        "pprint_batch: ctx.token_dict is empty; call ctx.set_default_dictionary()"

    # Invert token_dict (symbol -> id) into id -> symbol.
    id_to_symbol = {v: k for k, v in ctx.token_dict.items()}

    def tok_str(tok_id):
        tid = int(tok_id)
        # Every id in a batch produced by this ctx's Worker MUST be
        # in id_to_symbol. Missing entries almost always indicate the
        # caller passed a different ctx than the one that generated
        # the batch -- surface that loudly rather than silently
        # rendering a fallback symbol.
        assert tid in id_to_symbol, (
            f"pprint_batch: token id {tid} not in ctx.token_dict; the "
            f"ctx handed to pprint_batch differs from the one that "
            f"generated this batch")
        return id_to_symbol[tid]

    # --- Resolve which rows to print --------------------------------------
    src = batch['src_tokens']
    B   = src.shape[0]
    if indices is None or (isinstance(indices, int) and indices < 0):
        idxs = list(range(B))
    elif isinstance(indices, int):
        idxs = list(range(min(indices, B)))
    else:
        idxs = [b for b in indices if 0 <= b < B]

    targets       = batch.get('targets')
    positions     = batch.get('positions')
    hop_distances = batch.get('hop_distances')
    src_lengths           = batch['src_lengths']
    num_nodes             = batch['num_nodes']
    num_edges             = batch['num_edges']
    left_pad_lengths      = batch['left_pad_lengths']
    graph_edge_start      = batch['graph_edge_start_indices']
    graph_edge_lengths    = batch['graph_edge_lengths']
    query_start           = batch['query_start_indices']
    query_lengths         = batch['query_lengths']
    thinking_start        = batch['thinking_start_indices']
    thinking_lengths      = batch['thinking_lengths']
    scratchpad_start      = batch['scratchpad_start_indices']
    scratchpad_lengths    = batch['scratchpad_lengths']
    task_start            = batch['task_start_indices']
    task_lengths          = batch['task_lengths']

    pad_id = 1  # TOK_PAD

    # Fixed label column ("Src:  ", "Tgt:  ", "Lbl0: ", "Idx:  ", ...)
    # All chunks share a row-label margin of this width.
    row_label_width = 6

    for b in idxs:
        L        = int(src_lengths[b])
        n        = int(num_nodes[b])
        m        = int(num_edges[b])
        left_pad = int(left_pad_lengths[b])
        content_start = left_pad
        content_end   = left_pad + L    # exclusive

        # Row tokens (SEAN col-0 / STAN col-0). Vocab-token rendering
        # only uses col-0; STAN's cols 1..D-1 carry structural padding
        # or the second vertex of an edge triple that already appears
        # elsewhere in the sequence.
        row_tokens = src[b, content_start:content_end, 0] if src.ndim == 3 \
                     else src[b, content_start:content_end]
        symbols = [tok_str(t) for t in row_tokens]

        # Column width: max symbol width across THIS row, wide enough
        # to also hold column indices.
        col_w = max((len(s) for s in symbols), default=1)
        col_w = max(col_w, len(str(content_end - 1)))
        col_w = max(col_w, 2)

        # ------------- Header -----------------------------------------
        print('=' * min(78, max_width))
        print(f'BATCH INDEX {b}   n={n}  m={m}  src_len={L}'
              f'  left_pad={left_pad}')
        # Section boundaries at a glance.
        def section_str(label, start_arr, len_arr):
            s = int(start_arr[b]); ln = int(len_arr[b])
            return None if ln == 0 else f'{label}[{s}:{s + ln})'
        sections = [f'BOS@{content_start}']
        for lbl, s_arr, l_arr in [
            ('Edges',      graph_edge_start,   graph_edge_lengths),
            ('Query',      query_start,        query_lengths),
            ('Thinking',   thinking_start,     thinking_lengths),
            ('Scratchpad', scratchpad_start,   scratchpad_lengths),
            ('Target',     task_start,         task_lengths),
        ]:
            item = section_str(lbl, s_arr, l_arr)
            if item is not None:
                sections.append(item)
        sections.append(f'EOS@{content_end - 1}')
        print('Sections: ' + '  '.join(sections))
        print('-' * min(78, max_width))

        # ------------- Section masks: which columns each row covers ---
        # Each entry: (row_label, start_col, len, per_col_symbol_fn)
        #   per_col_symbol_fn(c) returns the symbol to render at
        #   ROW column c, or None if that column isn't in this section.
        section_rows = []

        def make_src_sub(s_col, s_len):
            # For every column in [s_col, s_col + s_len), render the
            # actual src token at that column; return None outside.
            def fn(c):
                if s_col <= c < s_col + s_len:
                    idx_in_row = c - content_start
                    if 0 <= idx_in_row < len(symbols):
                        return symbols[idx_in_row]
                return None
            return fn

        for lbl, s_arr, l_arr in [
            ('Edg',  graph_edge_start,   graph_edge_lengths),
            ('Qry',  query_start,        query_lengths),
            ('Thk',  thinking_start,     thinking_lengths),
            ('Scr',  scratchpad_start,   scratchpad_lengths),
            ('Tgt',  task_start,         task_lengths),
        ]:
            s = int(s_arr[b]); ln = int(l_arr[b])
            if ln > 0:
                section_rows.append((lbl, make_src_sub(s, ln)))

        # ------------- Targets tensor -> per-k label rows -------------
        # Generation region starts at the earliest section marker
        # (thinking/scratchpad/task, whichever exists), which sits ONE
        # column before its content start.
        def marker_col_of(sec_start, sec_len):
            s = int(sec_start[b]); ln = int(sec_len[b])
            return s - 1 if ln > 0 else None
        gen_cands = [
            marker_col_of(thinking_start,   thinking_lengths),
            marker_col_of(scratchpad_start, scratchpad_lengths),
            marker_col_of(task_start,       task_lengths),
        ]
        gen_cands = [c for c in gen_cands if c is not None]
        gen_col = min(gen_cands) if gen_cands else content_start

        label_rows = []  # list of (row_label, per_col_fn)
        if targets is not None:
            gen_len_row = targets.shape[1]   # global max_gen_len
            max_labels  = targets.shape[2]
            for k in range(max_labels):
                def make_label_fn(kk):
                    def fn(c):
                        gp = c - gen_col
                        if 0 <= gp < gen_len_row:
                            tok = int(targets[b, gp, kk])
                            if tok != pad_id:
                                return tok_str(tok)
                        return None
                    return fn
                label_rows.append((f'Lbl{k}' if k > 0 else 'L0*', make_label_fn(k)))
            # `L0*` on the first label row emphasises "chosen" (must
            # match src at that column); Lbl1..Lbl(k-1) are alternatives.

        # ------------- Chunk-wrapped rendering ------------------------
        cols_per_line = max(1,
            (max_width - row_label_width - 2) // (col_w + 1))
        total_cols = content_end - content_start

        def render_row(prefix, cells):
            # cells is a list of strings or None (blank).
            padded = [(c if c is not None else '').rjust(col_w) for c in cells]
            print(prefix.ljust(row_label_width) + ' '.join(padded))

        for start in range(0, total_cols, cols_per_line):
            end = min(start + cols_per_line, total_cols)
            cols = list(range(content_start + start, content_start + end))

            # Idx: column indices.
            render_row('Idx:', [str(c) for c in cols])
            # Src: full row's tokens.
            render_row('Src:', [symbols[c - content_start] for c in cols])
            # Pos: positional ids (optional).
            if show_positions and positions is not None:
                pos_row = positions[b, content_start:content_end, 0] \
                          if positions.ndim == 3 \
                          else positions[b, content_start:content_end]
                render_row('Pos:',
                           [str(int(pos_row[c - content_start])) for c in cols])
            # Per-section overlays.
            for lbl, fn in section_rows:
                cells = [fn(c) for c in cols]
                # Skip a section row if it has no content in THIS chunk.
                if not any(x is not None for x in cells):
                    continue
                render_row(lbl + ':', cells)
            # Targets alternatives, one row per k. Skip k>0 rows that
            # are empty in this chunk (common: chosen-only positions
            # like markers show only on L0*).
            for lbl, fn in label_rows:
                cells = [fn(c) for c in cols]
                if not any(x is not None for x in cells):
                    continue
                render_row(lbl + ':', cells)
            print()

        # ------------- Optional hop-distance matrix -------------------
        # Delegates to pprint_distance_matrix so the same renderer
        # backs both `pprint_batch(..., show_hop_distances=True)` and
        # standalone ad-hoc `pprint_distance_matrix(hd)` calls. Any
        # future distance semantics (weighted, ...) get their own
        # slice + call with a different `name` / `formatter`.
        if show_hop_distances and hop_distances is not None:
            pprint_distance_matrix(
                hop_distances[b, :n, :n],
                name='Hop distances',
                unreachable=-1,
            )


def get_generator_module():
    """
    Import the C++ `generator` module, installing / rebuilding it via
    scikit-build-core (`pip install -e .`) when necessary.

    The C++ build is driven by CMake + scikit-build-core (see pyproject.toml).
    Two mechanisms keep the module up to date:

      1. If `import generator` fails, we shell out to `pip install -e .` from
         the repo root. That runs CMake and drops an editable-install shim
         into site-packages.
      2. Once installed in editable mode, scikit-build-core's
         `editable.rebuild = true` makes subsequent imports transparently
         rebuild the extension when C++ sources change -- no explicit
         freshness check needed here.
    """
    def build_module():
        """Install the extension in editable mode from the repo root."""
        import subprocess
        repo_root = os.path.dirname(os.path.abspath(__file__))
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", repo_root],
            check=True,
        )

    try:
        import generator  # noqa: F401 -- probe import
    except ModuleNotFoundError:
        print("C++ module `generator` not found. Building via scikit-build-core...")
        build_module()
        import generator  # noqa: F811

    setattr(generator, "pprint_distance_matrix", pprint_distance_matrix)
    setattr(generator, "pprint_distance_ranks",  pprint_distance_ranks)
    setattr(generator, "pprint_batch",     pprint_batch)
    setattr(generator, "GraphPlotter",     GraphPlotter)

    # Section slicing helpers (see docstrings above). ONE dispatch on
    # section name; replaces V1's per-section wrappers
    # (task_gather_ids, edge_gather_ids, scratchpad_gather_ids,
    # node_gather_ids, query_gather_ids).
    setattr(generator, "SECTION_KEYS",           SECTION_KEYS)
    setattr(generator, "section_span",           section_span)
    setattr(generator, "section_tokens",         section_tokens)
    setattr(generator, "section_tokens_padded",  section_tokens_padded)
    setattr(generator, "section_span_batched",   section_span_batched)
    setattr(generator, "section_gather_ids",     section_gather_ids)

    def help_str():  # displays docstrings from cpp files with print(generator.help_str())
        # note this only works for the cpp functions not the added python functions above
        return pydoc.render_doc(generator, "\nDocstring for %s:")

    setattr(generator, "help_str", help_str)

    return generator


if __name__ == "__main__":
    # RUN ME TO CONFIRM COMPILE
    np.set_printoptions(threshold=np.inf, edgeitems=10, linewidth=np.inf, precision=2, suppress=True, )

    print(f'Running from CWD: {os.getcwd()}')
    generator = get_generator_module()
    print("C++ module `generator` loaded.")

    print(f'Random seed is {generator.get_seed()}')
    generator.set_seed(42)
    # generator.set_seed(3172477368)
    print(f'Random seed is {generator.get_seed()} after setting to 42')

    print(generator.help_str())

