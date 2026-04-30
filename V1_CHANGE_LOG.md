# V1 Change Log

Minimal notes for implementation changes that affect paper claims, results, or methodology.

## PC+greedy threshold calibration

- Replaced fixed `tau = 0.5` with a z-calibrated mean-shift threshold:
  `tau_j = 1.95996 * sqrt(s_obs,j^2 / n_obs + s_int,j^2 / n_int)`.
- Preserved v0 orientation behavior: significant shift means `target -> neighbor`; otherwise
  default to `neighbor -> target`.
- Audit on v0 seeds: active PC+greedy directed F1 `42.7 -> 48.8`, SHD `4.792 -> 4.479`.
- Paper note: non-significant shift is a heuristic reverse decision, not proof of reverse
  causality.

## Random directed-F1 floor

- Random policy: sample a random topological order, then sample `m` uniformly from
  `{0, ..., M}`, where `M = d(d-1)/2`.
- Conditional expected directed F1:
  `E[F1 | m] = k*m / [M*(m+k)] = rho*s / (rho+s)`, where `rho = k/M` and `s = m/M`.
- Exact discrete floor:
  `E[F1] = (1/(M+1)) * sum_{m=0}^{M} k*m / [M*(m+k)]`.
- `rho/(1+2*rho)` is only the midpoint/Jensen envelope at `s=1/2`, not the exact floor.
- Prior art: Petersen 2025 for random-graph F1 expectation; Reisach et al. 2021 for
  structure-blind diagnostic baselines.

## V1 scale-calibrated ladder

- `ladder_levels()` now returns v1. The original v0 ladder is preserved as
  `ladder_levels_v0()`; the active ladder is also available as `ladder_levels_v1()`.
- V1 is a graph-scale/generalization ladder with calibrated random floor, not a full
  factorial identifiability x statistics design.
- Sample sizes and noise are fixed after L0 so graph scale is not confounded with decreasing
  `n_obs`/`n_int`. Statistical scarcity should be a separate ablation.
- 2026-04-30 update: all v1 levels now use `budget_slack=2`.

| L | d | k | rho | n_obs | n_int | noise | slack | E_random Dir-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4 | 3 | 0.500 | 50 | 25 | 0.5 | 2 | 0.215 |
| 1 | 6 | 6 | 0.400 | 50 | 25 | 1.0 | 2 | 0.196 |
| 2 | 8 | 9 | 0.321 | 50 | 25 | 1.0 | 2 | 0.173 |
| 3 | 10 | 12 | 0.267 | 50 | 25 | 1.0 | 2 | 0.155 |
| 4 | 12 | 14 | 0.212 | 50 | 25 | 1.0 | 2 | 0.133 |
| 5 | 14 | 16 | 0.176 | 50 | 25 | 1.0 | 2 | 0.117 |

## 2026-04-30 -- Uniform budget slack

- Set `budget_slack=2` for L1-L5 in `ladder_levels_v1()`; L0 already used slack 2.
- Budget formula is unchanged: `budget = len(optimal_intervention_set) + budget_slack`.
- `optimal_intervention_set` remains the brute-force minimum set computed by
  `compute_minimum_intervention_set()`, accounting for Meek propagation.
- Motivation: zero slack at L4-L5 made one poor intervention target unrecoverable and mixed
  causal reasoning quality with exact minimum-set selection.
- Paper/result impact: all v1 active results must be rerun; random floor is unchanged.

## Validation probes

- Acceptance probe: `50/50` accepted instances at every v1 level with `max_attempts=1000`.
- PC probe on 8 seeds: PC+greedy SHD rises `0.625 -> 9.375`; observational PC SHD rises
  `3.000 -> 13.000`.
- Interpretation: random floor is calibrated and PC error burden increases. Directed F1 need
  not be perfectly monotone because it is ratio-sensitive and seed-limited.

## Paper implications

- V1 result tables are stale until random, PC, PC+greedy, and LLM panels are rerun on v1.
- Report the exact random floor alongside directed F1 to separate skill from metric leakage.
- Preserve v0 numbers as an audit/calibration story, not as final v1 benchmark evidence.
