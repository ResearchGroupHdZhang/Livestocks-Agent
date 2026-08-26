# Protocol Amendment 2 — Primary-Stage Scalability

Locked before running the final four-country batch.

The 60-second Australia hierarchical attempt produced a feasible stage-1 incumbent, but stage 2 spent its entire allowance in presolve and produced no incumbent. Continuing stage 3 would therefore violate lexicographic ordering. The unchanged three-stage formulation is not executable under a uniform short budget beyond EU.

The final cross-country experiment is narrowed to **stage 1 of the same MILP**: maximize source nitrogen resolved under the same source inventory, source N, destination N, destination ammonia, and integer movement constraints. This is the shared, scientifically meaningful primary objective and isolates scaling without pretending that stages 2–3 completed.

Locked settings:

- SCIP/PuLP, 60-second solver limit per country;
- sequential order Australia, China, United States, Brazil;
- record end-to-end wall time, model-build time, solver wall time, SCIP time/status/gap, peak RSS, model size, primary objective, and feasibility diagnostics;
- report all results as bounded-time primary-stage incumbents unless SCIP says `optimal`;
- retain both failed broader attempts as negative scalability evidence.
