# Protocol Amendment 9 — Exact Primary-Stage Reruns

Locked before restarting any solver.

## Scope

Rerun **only stage 1** of the source-composition lexicographic MILP for datasets whose primary nitrogen stage did not reach SCIP `optimal` in the gap-certified five-dataset batch:

1. Australia (`aus`) — prior status `gaplimit`, not exact;
2. China (`cn`) — prior status `timelimit`, uncertified;
3. United States (`usa`) — prior status `timelimit`, uncertified;
4. Brazil (`br`) — prior status `timelimit`, uncertified.

EU is excluded because its primary stage was already SCIP `optimal` with zero gap.

## Model and stopping rule

- Reuse the current source-composition model's stage-1 integer decision variables and all physical constraints unchanged.
- Objective: maximize source nitrogen resolved only.
- **No SCIP time limit, MIP-gap limit, node limit, or external process timeout.**
- A country succeeds only when SCIP returns `optimal`; any other terminal status is recorded as non-optimal and does not count as an exact result.
- Run sequentially in the locked order above to avoid memory contention.

## Required records

For each country record input/model-build wall time, solver-reported time, `solve()` wall time, end-to-end wall time, peak RSS, status, primal/dual bounds, gap, solution count, nodes, model size, and independently recomputed feasibility/inventory diagnostics. Record the sequential batch wall time.

## Durability and outputs

- Use a persistent user systemd service with automatic restart after host reboot.
- On restart, skip only countries with a saved `status: optimal` result; restart an incomplete country from scratch and retain its prior attempt metadata/logs as interrupted evidence.
- Canonical root: `baselines/milp/results/source_composition_primary_exact/`.

This is a primary-stage scalability/exactness experiment, not a completed three-stage comparison. Stage 2 and stage 3 are deliberately deferred pending review of the exact stage-1 results.
