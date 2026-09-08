# Protocol Amendment 14 — Paper-Method PPO versus MILP Comparison

Locked before implementing or running the new comparison evaluator on 2026-09-08.

## Question

On the same first-stage national input, how do the archived paper-method PPO layouts compare with the certified normalized weighted 4:2:1 MILP layouts under independently recomputed physical outcomes, feasibility checks, and the common MILP score?

## Scope

### Confirmatory matched comparisons

- EU:
  - input: `data/欧盟24/欧盟更新PB后第一步.xlsx`
  - PPO: `results/v9/eu/`
  - MILP: `baselines/milp/results/weighted_4_2_1/eu/`
- USA:
  - input: `data/美国1223/美国数据国家尺度第一步1223.xlsx`
  - PPO: `results/v9/usa/`
  - MILP: `baselines/milp/results/weighted_4_2_1/usa/`

The PPO version is selected because its final source and destination entity IDs/order and species columns exactly match the MILP input. Later same-named archive directories that do not match these entities are excluded rather than coerced.

### Deferred instances

- Australia has an input-matched archived PPO v8 result and a certified MILP result, but it is not the v9 archive used for the EU/USA confirmatory comparison. It may be reported separately as a version-qualified robustness comparison after the confirmatory analysis.
- China and Brazil archived PPO results exist, but the weighted MILP runs are not yet certified. They cannot enter the certified quality table until the MILP summary and independent validation are complete.

## Method Mapping

The paper PPO reward is action-local:

\[
R_t = 4R_{NH_3,t}+2R_{sens,t}+R_{PM2.5,t},
\]

where the first term is a normalized destination-ammonia-capacity term and the latter terms reward less-sensitive/lower-PM2.5 destinations. PPO selects a source–destination pair; the shared local linear allocator determines livestock quantities.

The current weighted MILP score is global:

\[
J = 4\hat N-2\hat L+\hat E,
\]

where \(\hat N\) is normalized source-N resolution, \(\hat L\) is the source-composition proxy loss, and \(\hat E\) combines destination sensitivity and PM2.5 as 2:1.

These are not the same native objective. Therefore:

1. Native PPO episodic reward may be described but is not used to rank PPO against MILP.
2. Every final layout is independently rescored under the same physical evaluator.
3. Common score \(J\) is reported explicitly as a post-hoc MILP-objective score for PPO, not as PPO's training reward.
4. Paper outcome indicators and feasibility are primary; the scalar score is diagnostic.

## Locked Metrics

For each method and country, recompute from final source/destination inventories and route rows:

1. source N resolved (kg and ratio);
2. source N remaining (kg) and rate of source counties reaching the 90% resolution target;
3. destination N capacity violation count and maximum excess;
4. destination ammonia-capacity violation count and maximum excess;
5. exact livestock conservation and route-to-aggregate consistency;
6. total moved animals and movement by species;
7. destination sensitivity reward, PM2.5 reward, and the normalized 2:1 environmental score;
8. source-level composition proxy \(\hat L\);
9. actual source remaining-share L1 change;
10. post-hoc common score \(J=4\hat N-2\hat L+\hat E\);
11. MILP solver status/gap/time; PPO archive provenance and, only if trustworthy logs exist, training/inference timing.

Strict raw violation counts and scale-aware tolerance counts will both be shown. A method is marked physically validated only if exact conservation/route checks pass and all scale-aware N/ammonia violation counts are zero.

## Predictions

- MILP should dominate or nearly dominate PPO on post-hoc \(J\) because MILP directly optimizes it to a 0.1% certificate.
- PPO may obtain higher raw environmental reward by moving more low-N animals; total movement and N resolved per animal will reveal this mechanism.
- PPO's archived solution may contain small source-N residual violations caused by sequential numerical stopping; these must be reported, not silently clipped.
- The comparison will support a scalability/solution-generation claim only if PPO produces physically validated layouts; it will not support a universal quality-superiority claim if MILP has higher \(J\).

## Validation and Failure Rules

- Input and output entity IDs/order and species columns must match exactly.
- Route quantities must reproduce outbound and inbound inventory changes exactly.
- All moves must be nonnegative integers and conserve each species exactly.
- Metrics are computed from the input coefficients, not trusted from saved result columns or PPO reward.
- If any identity, route, integrality, or scale-aware capacity check fails, label the artifact invalid for the main comparison and retain the diagnostic record.
- Do not modify archived PPO or MILP outputs.
