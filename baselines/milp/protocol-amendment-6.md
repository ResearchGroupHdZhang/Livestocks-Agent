# Protocol Amendment 6 — Fixed Per-Stage Compute Budget

Locked before restarting the five-dataset batch, following user approval.

Final termination policy for every dataset:

1. Stage 1 (`maximize_source_n_resolved`): 2 hours SCIP time.
2. Stage 2 (`minimize_route_structure_deviation`): 1 hour SCIP time.
3. Stage 3 (`maximize_destination_environment`): 1 hour SCIP time.

Rules:

- relative gap is not a stopping condition;
- when a stage reaches its time limit, accept its best feasible incumbent and record the achieved gap and primal/dual bounds;
- if a stage proves optimal before its time limit, continue immediately;
- if no feasible incumbent exists when the budget expires, record the country as failed rather than creating a zero or fabricated result;
- lock each accepted incumbent objective before continuing to the next lexicographic stage;
- run Australia, China, United States, Brazil, and EU sequentially;
- report per-stage SCIP time, Python wall time, status, gap, bounds, solution count, nodes, per-country end-to-end time, peak RSS, and total batch time;
- preserve previous exact/gap attempts as diagnostics, but keep final fixed-budget outputs separately under `results/fixed_budget_2h_1h_1h/`.

The nominal solver budget is at most four hours per dataset. End-to-end duration can be longer because PuLP model construction, conversion to SCIP, result extraction, validation, and file serialization are outside SCIP's solve-time limits.
