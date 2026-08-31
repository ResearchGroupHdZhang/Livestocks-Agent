# Protocol Amendment 11 — Four-Country Weighted 4:2:1 Without a Time Limit

Locked before modifying the runner or launching any new weighted solve, following explicit user instruction on 2026-08-31.

## Scope

Extend the confirmed EU unified weighted MILP to the four remaining mapped national datasets, in this sequential order:

1. Australia (`aus`) — `澳大利亚空间优化更新PB第一步.xlsx`;
2. China (`cn`) — `中国国家尺度更新PB第一步.xlsx`;
3. United States (`usa`) — `美国数据国家尺度第一步1223.xlsx`;
4. Brazil (`br`) — `巴西指标国家优化更新PB第一步.xlsx`.

EU remains the completed pilot and is not rerun. Preserve its existing artifacts unchanged.

## Frozen model and objective

Reuse the EU model, constraints, normalization, signs, and weights unchanged:

`maximize J = 4*N_hat - 2*L_hat + E_hat`.

- `N_hat`: source nitrogen resolved divided by that dataset's fixed total source nitrogen.
- `L_hat`: equal-source mean of equal-species mean absolute deviations from a common source removal rate; this is the corrected source-region composition proxy, not route proportionality.
- `E_hat`: destination environment score divided by that dataset's fixed inventory-derived upper bound.
- Internal destination environment weighting remains sensitivity:PM2.5 = `2:1`.
- Uniform solver objective scale remains `1,000,000`.
- Physical inventory, source/destination nitrogen, destination ammonia, destination-species admissibility, integrality, and the 1 kg per-source nitrogen guard remain unchanged.
- No movement cost, total movement cap, or minimum source-retention constraint is added. The EU finding that these omissions can produce extreme movement is intentionally preserved for cross-country comparison.

## Stopping policy

Per user instruction, there is **no SCIP time limit and no external process timeout**.

The scientific stopping target remains the previously confirmed relative MIP gap of `0.001` (0.1%):

- `optimal` succeeds;
- SCIP `gaplimit` succeeds only when native relative gap is `<= 0.001`;
- no incumbent or any other terminal status fails and is preserved as a machine-readable failure.

This is a certified weighted-objective experiment, not an exact-optimality experiment. “No time limit” means the solver may run as long as needed to prove optimality or reach the 0.1% gap certificate; it does not silently change the certificate to exact optimality.

## Compute and run order

- Run the four datasets sequentially to avoid concurrent memory pressure among the very large dense models.
- The already-running Australia exact-primary experiment is a separate preregistered service. Do not pause, terminate, or modify it. Weighted-run timings while it remains active must be labeled concurrent-load timings.
- Before each solve, write a preflight record with rows, variable counts, constraint counts, coefficient range, build time, and peak RSS.
- Persist a liveness ledger every minute with country, stage, PID, start time, elapsed wall time, RSS, CPU where available, log path, stop rule, boot ID, and active exact-primary concurrency.
- Use a persistent user systemd service with restart on failure. On restart/reboot, skip only a country with a saved certified (`optimal` or qualifying `gaplimit`) result and successful independent validation. Archive stale/interrupted attempt metadata, then restart the first incomplete country from scratch.

## Required outputs and validation

Canonical root: `baselines/milp/results/weighted_4_2_1/`, alongside the existing EU result.

For every named country preserve:

- preflight JSON;
- SCIP log, stdout, and stderr;
- saved movement, route, and source-structure workbooks;
- summary JSON with status, native gap, primal/dual bounds, solutions, nodes, solver/build/serialization/total timings, model size, peak RSS, all normalized components, weighted contributions, scalar score, total/per-species movement, and capacity diagnostics;
- independent validation JSON recomputing IDs, integrality, conservation, route aggregates, physical capacities, normalized components, weighted objective, and saved N/ammonia fields.

A country is complete only when its solver certificate and independent validation both pass. A failed country is never silently omitted; continue to the next country only when it is safe to do so.

After all four countries terminate, create a five-country table/report including EU, and explicitly audit:

- total/per-species movement;
- source regions near depletion;
- proxy loss versus actual remaining-share L1;
- environment gain;
- status/gap and timing under concurrent versus non-concurrent host load.

## Scale warning

Previous dense primary-only models contained approximately 0.39M (Australia), 10.65M (China), 7.49M (USA), and 27.23M (Brazil) integer movement variables. The weighted model adds only source×species continuous auxiliaries, but Python/PuLP construction, solver translation, memory, serialization, and proof time may still be very large. No runtime estimate is asserted before actual preflight/solve evidence.
