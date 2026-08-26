# Protocol Amendment 3 — Unlimited Exact Run

Locked before restarting the four-country experiment.

Per user direction, remove all solver and external wall-clock limits. The previous 60-second primary-stage runs are retained only as pilot/scalability diagnostics under `results/pilot_60s/`; they are not final country results.

Final protocol:

- run the original archived three-stage lexicographic MILP unchanged for Australia, China, United States, and Brazil;
- stages 1 and 2 must reach SCIP `optimal` before continuing;
- stage 3 also has no time limit and must reach `optimal`; no time-limited incumbent is accepted as final;
- run countries sequentially in the fixed order Australia, China, United States, Brazil to avoid memory contention;
- measure every stage's SCIP time and Python wall time, plus per-country end-to-end wall time, peak RSS, and the total sequential batch time;
- preserve periodic liveness data (elapsed wall time, RSS, CPU, current stage/log size) while each country runs;
- write all code, logs, results, timing summaries, validation, and analysis only under `baselines/milp/`.

The run may take days or longer. Completion means all three stages finish and validation artifacts are produced; elapsed time alone is not treated as failure.
