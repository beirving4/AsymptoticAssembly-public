# asymptotic.merger_trees — Merger-Tree Boundary

Canonical home of the low-level merger-tree layer, separated from the
scientific interpretation in `asymptotic/assembly/`:

| Module | Responsibility |
|---|---|
| `backends.py` | `MergerTreeBackend` contract, canonical (quantity, aperture) vocabulary, and the `open_merger_tree` factory |
| `full_tree.py` | `FullTreeStore` (GADGET-4 tree storage, single-file and sharded) and `TesseraTreeBackend` (Tessera-backed topology + column gather) |
| `topology.py` | `BranchTopology` (main-progenitor) and `ProgenitorTreeTopology` (all-progenitor) records plus branch flattening/scatter machinery |

Scientific histories, cleaning, gather orchestration, accretion, merger
definitions/rates, statistics, and visualization stay in
`asymptotic/assembly/` (see `docs/assembly_pipeline_requirements.md`).

New code should import from the concrete modules —
`asymptotic.merger_trees.backends`, `asymptotic.merger_trees.full_tree`, or
`asymptotic.merger_trees.topology`; the package `__init__` is empty and
re-exports nothing. The old
`asymptotic.assembly.{backends,full_tree,topology}` paths are temporary
compatibility re-exports of the same objects and will be removed at a
documented release boundary.

## Production traversal

- Tessera's C++ batch tracer (`tessera.io.extract_branches_from_rows`) is
  the authoritative production main-branch traversal;
  `TesseraTreeBackend.main_branches` is a thin adapter over it.
- `parallel=False` runs the same C++ tracer with exactly one thread; there
  is no per-descendant Python traversal in production.
- The vectorized NumPy walk survives only as the private test oracle
  `_main_branches_numpy_oracle`. Production code never calls it, and there
  is no silent Python fallback: constructing the Tessera backend without a
  usable tessera raises an actionable `ImportError`.
- Portable tessera wheels/distribution remain a separate release task; until
  then the backend requires the project's local tessera build. Tessera is
  deliberately **not** a declared dependency of the public installation
  until its public distribution identity is resolved, so the wheel stays
  portable and merger-tree workflows fail with an actionable `ImportError`
  when it is absent.

## Full-tree (all-progenitor) traversal

- `TesseraTreeBackend.progenitor_trees(desc_nodes, a_min, parallel,
  n_threads)` is the production all-progenitor path: one GIL-released
  `tessera.io.extract_progenitor_trees_from_rows` call returns every
  progenitor reachable from each selected row (`TreeFirstProgenitor` plus
  each child's `TreeNextProgenitor` chain), in tessera's deterministic
  depth-first preorder. No ytree, no runtime fallback, and no Python
  traversal is involved — the capability is checked per call, so
  main-branch workflows still run against an older tessera build.
- The result is a `ProgenitorTreeTopology`, deliberately separate from
  `BranchTopology`: a full tree holds many nodes per (descendant,
  snapshot), which is exactly the invariant `BranchTopology.scatter`
  relies on, so the full-tree record has no `scatter`.
- `build_saved_order_info(backend, desc_nodes, a_min, parallel, n_threads)`
  produces the canonical saved-order mapping
  `{descendant GroupNr: {"tree": …, "prog": …}}` — the all-progenitor call
  supplies `"tree"`, the accepted main-branch call `"prog"`, both with
  identical selection, `a_min`, and threading. Write it with
  `asymptotic.assembly.trees.save_merger_tree_info` and read it back with
  `load_merger_tree_info`:

  ```python
  info = build_saved_order_info(backend, desc_rows)
  save_merger_tree_info(save_base, info)      # <base>_tree_order.hdf5 + _prog_order.hdf5
  ```

- Performance model: two native batch traversals for the whole selection,
  one bulk cached `GroupNr` column read, then vectorized gathers,
  snapshot grouping, and `np.unique` deduplication. The converter loops
  only over descendants and snapshots — never over nodes — and creates no
  Python object per node.

## Optional-backend imports

- Importing `asymptotic.merger_trees` (or any of its modules) imports
  neither tessera nor ytree.
- Tessera stays lazy: it is imported only when the Tessera backend is
  constructed, with an actionable error when a usable build is absent.
- The public ytree backend was retired before the first public release
  (A3b1): Tessera is the sole backend, `open_merger_tree` accepts only
  `auto|tessera`, and arbor inputs are rejected explicitly. The historical
  `YtreeArborBackend` implementation survives only as an excluded migration
  example under `postprocessing/legacy_ytree/`, which the package never
  imports.
- ytree is not a dependency of the core installation, and since A3b2 the
  shipped package contains no ytree import statements at all: the mixed
  arbor helpers formerly in `utils.get_data`, `utils.analysis`,
  `assembly/trees.py`, and `bounded.fof` were retired to
  `postprocessing/legacy_ytree/arbor_analysis.py`. The generic saved-order
  serialization (`save_merger_tree_info`, the order loaders) and
  order-based FOF history construction (`GroupTree`,
  `get_main_progenitor_branch*`, the `*_from_order` helpers) remain and
  operate purely on precomputed order artifacts.
- `utils.analysis.get_tree_order` no longer synthesizes a missing order
  artifact from an arbor; it raises an actionable error instead, pointing
  at the current production replacement. The retired synthesis had
  all-progenitor (full-tree) semantics per descendant, which is now served
  natively: build the artifact with `TesseraTreeBackend.progenitor_trees`
  via `build_saved_order_info` (see the section above) and write it with
  `asymptotic.assembly.trees.save_merger_tree_info`, then read it back
  with `load_merger_tree_info`. Both halves come from Tessera C++ — no
  ytree runtime and no fallback of any kind.
