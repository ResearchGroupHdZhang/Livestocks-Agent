# Protocol Amendment 12 — Stop China and Advance to USA

Locked after the user explicitly requested on 2026-09-07 that the current China weighted solve be stopped and the batch advance to the next country.

## China disposition

- Terminate only the active China (`cn`) unified weighted 4:2:1 solve.
- Preserve its preflight, solver/stdout/stderr logs, liveness records, and a machine-readable user-abort record.
- Classify China as `aborted_by_user`, `certified: false`.
- The terminated attempt is not a feasible nonzero result and must not enter cross-country outcome tables as a solved instance.
- Do not automatically retry China after service restart or host reboot. A future China restart requires new explicit user approval and a new locked protocol amendment.

At termination, China had run for approximately 589,165 seconds. Its latest solver evidence remained: trivial all-zero incumbent, scaled primal bound `0`, scaled dual bound `4.832770e9`, infinite native gap, and unresolved numerical trouble in root LP 2 followed by a pseudo-solution fallback. This is an aborted, uncertified scalability/numerical-failure attempt—not evidence that zero movement is optimal.

## Remaining batch

Continue immediately with the next locked countries:

1. United States (`usa`);
2. Brazil (`br`).

For both, preserve the weighted model and certificate policy from Amendment 11 unchanged:

- objective `maximize 4*N_hat - 2*L_hat + E_hat`;
- no SCIP time limit and no external timeout;
- success only at SCIP `optimal` or native relative MIP gap `<= 0.001` with a feasible incumbent;
- independent serialized-output validation required before completion.

Australia exact-primary remains a separate process and must not be stopped by this amendment. Timings remain concurrent-load measurements while it is active.
