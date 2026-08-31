# Protocol Amendment 10 — EU Unified Weighted 4:2:1 MILP

Locked before implementation and before any weighted-objective solve, following explicit user confirmation on 2026-08-31.

## Scope and question

Run one EU integer optimization on `欧盟更新PB后第一步.xlsx` and compare it with the existing EU lexicographic baseline. Replace the three sequential objectives and inter-stage locks with one unified weighted objective. Keep the movement variables, source/destination inventory and nitrogen constraints, destination ammonia constraint, 1 kg per-source nitrogen numerical guard, and the corrected source-region composition semantics unchanged.

The three confirmed objectives are:

1. maximize source nitrogen resolved;
2. preserve each outbound/source region's livestock composition by minimizing unequal species-specific removal rates within that region;
3. maximize destination environmental placement score.

This experiment does **not** use the obsolete source→destination route-proportionality objective.

## Locked normalized objective

Let `x[i,j,k]` be integer livestock movement, `A[i,k]` source stock, and `m[i,k] = sum_j x[i,j,k]`.

### 1. Source nitrogen

`N_hat = resolved_source_n_kg / total_source_n_kg`.

The denominator is fixed from the EU input, so this remains linear and `N_hat` is bounded in `[0, 1]` by the physical source constraints.

### 2. Source-region composition

For each source `i`, introduce a common removal fraction `q[i] in [0,1]` and deviations:

`d[i,k] >= m[i,k] / A[i,k] - q[i]`

`d[i,k] >= q[i] - m[i,k] / A[i,k]`

for each species present at that source. Define the equal-source-weight loss

`L_hat = (1 / I) * sum_i (1 / K_i) * sum_k d[i,k]`,

where `I` is the number of active source regions and `K_i` is the number of species present at source `i`. Because every absolute deviation is in `[0,1]`, `L_hat` is in `[0,1]`. Equal removal rates produce `L_hat = 0` and preserve the source livestock shares exactly whenever stock remains.

### 3. Destination environment

Retain the existing environment coefficient for destination `j`:

`e[j] = 2 * (1 - sensitivity[j]) + (1 - pm25[j])`.

Let `E_raw = sum x[i,j,k] * e[j]`. Normalize by the fixed inventory-derived upper bound

`E_max = sum_i sum_k A[i,k] * max_{j: destination j accepts k} e[j]`.

Then `E_hat = E_raw / E_max`. `E_max` ignores destination nitrogen/ammonia competition and is therefore an attainable-or-loose physical upper bound, but it is fixed before solving and guarantees `E_hat in [0,1]`. The internal environmental sensitivity:PM2.5 weighting remains `2:1` and is separate from the outer three-objective `4:2:1` weighting.

### Unified objective

Maximize

`J = 4 * N_hat - 2 * L_hat + 1 * E_hat`.

No lexicographic retention constraints are added. Trade-offs are intended: source nitrogen may decrease if the weighted structure/environment gains compensate for it.

## Solver and acceptance

- Solver: SCIP through PuLP `SCIP_PY`, using the installed project environment.
- Relative MIP-gap target: `0.001` (0.1%).
- SCIP time limit: `7200` seconds, used only as a hard safety cap.
- Success requires at least one feasible incumbent and one of:
  - SCIP `optimal`;
  - SCIP `gaplimit` with native gap `<= 0.001`;
  - SCIP `timelimit` with native gap `<= 0.001`.
- A time-limit incumbent above 0.1% is preserved and reported as uncertified, not as a successful/converged result.
- Record status, termination reason, certification, target/achieved gap, primal/dual bounds, solution count, nodes, SCIP time, `solve()` wall time, model-build time, serialization time, total time, variable/constraint counts, and peak RSS.

Per the user's decision, this EU solve may run concurrently with the already-running Australia exact-primary experiment. Timings must therefore be labeled as concurrent-load timings rather than isolated host benchmarks. Neither process may be paused or altered solely to improve timing comparability.

## Required preflight and validation

Before the EU solve:

1. run a tiny unequal-stock source example through the actual weighted solver and serialization path;
2. assert integer conservation and all physical constraints;
3. recompute `N_hat`, `L_hat`, `E_hat`, and `J` independently;
4. assert the normalized component bounds and objective identity.

For EU, independently validate serialized workbooks/routes and report:

- source N resolved kg and ratio;
- normalized source-composition loss, actual before/after source-share L1 change, and their maxima;
- raw and normalized environment scores;
- total and per-species animals moved;
- all constraint residuals and numerical-tolerance exceptions;
- unified objective and all weighted component contributions;
- solver certificate and timing;
- comparison against the existing EU lexicographic baseline using the corrected source-region composition metrics recomputed from its saved movement output.

## Artifact root

All new protocol, code, logs, raw results, validation, analysis, and human reports remain under `baselines/milp/`. Weighted-run outputs use `baselines/milp/results/weighted_4_2_1/eu/` and sibling JSON/log files so no existing baseline is overwritten.
