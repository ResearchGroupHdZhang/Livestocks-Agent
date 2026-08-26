# Protocol Amendment 1 — Bounded-Time Cross-Country Run

Locked before collecting any completed four-country result.

The first Australia exact attempt showed that objective stage 1 had not proved optimality after 748 wall-clock seconds; it was deliberately stopped with no result retained. Applying the EU requirement that stages 1 and 2 must be exact would make the four-country extension impractical and potentially unbounded in duration because the other models contain 7.49M–27.23M movement variables.

The extension therefore changes **solver termination only**:

- each of the three stages receives a 60-second SCIP limit;
- a time-limited stage may continue only if SCIP has a feasible incumbent;
- the incumbent objective is locked before the next stage, so results are bounded-time hierarchical MILP solutions, not proven lexicographic optima;
- objectives, constraints, input mapping, integer variables, and verification remain unchanged;
- countries run sequentially in the locked order Australia, China, United States, Brazil;
- a four-hour external timeout per country protects against model-construction or serialization stalls;
- total timing includes loading, model construction, all stages, and output serialization.

The aborted exact attempt is retained separately as scalability evidence and excluded from the four completed-run timing total.
