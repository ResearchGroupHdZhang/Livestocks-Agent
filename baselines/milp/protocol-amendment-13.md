# Protocol Amendment 13 — China Nonzero Full-Warm-Start Restart

Locked before modifying the weighted solver or launching the new China attempt, following the user's explicit request on 2026-09-08.

## Correct interpretation of the prior attempt

The prior China attempt did **not** establish that only the all-zero solution is feasible. Evidence already proves the opposite: the earlier primary-N run found a nonzero feasible incumbent (`primal_bound=4,275,114.6955` in its scaled primary objective). The weighted run retained only SCIP's automatically generated all-zero incumbent because no explicit nonzero warm start was supplied; after 5,665 minutes its huge root LP encountered unresolved numerical trouble and used a pseudo solution. The attempt was then stopped by explicit user request. It is an interrupted, uncertified solver attempt—not an infeasibility or zero-optimality result.

## Restart objective and certificate

Restart China on the same input and keep unchanged:

- all integer movement variables and physical constraints;
- source-region composition proxy semantics;
- normalized objective `maximize 4*N_hat - 2*L_hat + E_hat`;
- uniform objective scale `1,000,000`;
- target native relative MIP gap `0.001` (0.1%);
- no SCIP time limit and no external process timeout.

Success still requires a feasible incumbent and SCIP `optimal` or `gaplimit` within 0.1%, followed by independent serialized-output validation. A nonzero warm start is evidence of feasibility, not an optimality certificate.

## Deterministic nonzero full warm start

Before the country-scale restart, construct a conservative one-route integer solution:

1. enumerate every allowed `(source, destination, species)` route;
2. for each route compute the maximum positive integer amount satisfying source inventory, the 1 kg source-N guard, destination-N capacity, and destination-ammonia capacity;
3. score each candidate under the exact normalized 4:2:1 objective, including its source-composition deviation when all other movement variables are zero;
4. select the highest strictly positive candidate;
5. independently recompute every physical residual and objective component;
6. set initial values for **every** movement variable, source reference variable, and deviation variable so SCIP receives a complete feasible original-space solution rather than an incomplete partial assignment;
7. call SCIP with `warmStart=True`.

The preregistered seed for China is source index 38 → destination index 212, species index 3, amount 84,030,010. Independently computed human-scale objective is `0.04145449090808133`, with positive residuals for source N, destination N, and destination ammonia. All unspecified movement/reference/deviation variables are exactly zero.

## Acceptance check before trusting the long run

The live SCIP log must show a warm-start solution accepted/stored with a strictly positive objective no later than the initial transformed-solution stage. If it instead reports partial-solution completion failure, infeasibility, or retains only objective zero, stop that new attempt, preserve diagnostics, and fix the warm-start construction before another long run.

## Scheduling and artifacts

- Canonical restart root: `baselines/milp/results/weighted_4_2_1_cn_restart/`.
- Preserve the prior aborted attempt unchanged under `results/weighted_4_2_1/`.
- Run the China restart as a separate persistent user systemd service so the current Brazil weighted run is not interrupted.
- Timings must be labeled concurrent-load because Brazil weighted and Australia exact-primary jobs are active at launch.
- Persist preflight, seed JSON, liveness ledger, SCIP/stdout/stderr logs, summary/workbooks, and independent validation.
