# Four-Country Lexicographic MILP Protocol

## Status

CONFIRMATORY extension. This protocol is locked before any of the four optimization runs.

## Question

Does the archived EU three-stage lexicographic MILP execute successfully on the four other datasets already mapped by the runner—China (`cn`), Australia (`aus`), United States (`usa`), and Brazil (`br`)—and how much wall-clock time does each dataset require?

## Frozen Model

Reuse the archived EU implementation without changing its objective or constraints:

1. maximize source nitrogen resolved;
2. lock level 1 to the best whole kg and minimize route-wise L1 structure deviation;
3. lock level 2 and maximize destination environmental score with a 300 s limit.

Solver: SCIP through PuLP `SCIP_PY`. Input files and country mappings are exactly those in `code/run_lexicographic.py`.

## Locked Measurements

For every country record:

- total wall-clock seconds measured by `time.perf_counter()` around data loading, model construction, all solver stages, and result serialization;
- SCIP solving seconds and Python wall-clock seconds for every objective stage;
- solver status, solution count, gap, and node count for every stage;
- peak resident memory;
- source/destination counts, integer movement-variable count, result shape, and output size;
- feasibility checks from `verify_lexicographic.py`.

Also report the four-country sequential wall-clock total. Timings are measured on this host and are not hardware-independent benchmarks.

## Predictions

- H4: Australia completes first because it has the smallest extension model (387,455 integer movement variables).
- H5: China, United States, and Brazil take substantially longer than EU/Australia because their integer-variable counts are 10.65M, 7.49M, and 27.23M, respectively.
- H6: The free-median structure reference will again favor sparse route compositions; this is descriptive because the objective is intentionally unchanged.

## Failure Rule

A run is not silently dropped. If model construction or a required-optimal stage fails, preserve the error and elapsed time, then continue with the remaining countries. A time-limited final environment stage is valid only when SCIP has at least one feasible incumbent, matching the EU protocol.
