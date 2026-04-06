# Benchmark Results (BENCHMARK_CONFIG, 5 seeds)

## Baseline Agents (deterministic, no LLM)

Agent                Mean    Std  Per-seed
--------------------------------------------------------------------------------
Random              472.6 249.6  s42=116  s123=529  s456=811  s789=272  s1024=635
NaivePattern        435.4 249.3  s42=615  s123=67  s456=778  s789=441  s1024=276
CausalLearner      1324.0 219.6  s42=1400  s123=1590  s456=1450  s789=1230  s1024=950
CausalReasoner     1516.0 143.5  s42=1660  s123=1560  s456=1450  s789=1270  s1024=1640
Oracle             1716.0  53.1  s42=1750  s123=1620  s456=1740  s789=1770  s1024=1700

## Expected ordering
Random ≈ NaivePattern << CausalLearner < CausalReasoner <= Oracle

## LLM Agents (GPT 5.4)

Agent                Mean    Std  Per-seed
--------------------------------------------------------------------------------
LLM+Solver         1124.0 354.6  s42=550  s123=1140  s456=1160  s789=1130  s1024=1640
LLMAgent (3/5)      663.3 372.3  s42=240  s123=940  s456=810  s789=???  s1024=???
CodingAgent (1/5)  1069.0    —   s42=1069

### Detailed Metrics

Agent            Reward    Brier(S)  Brier(E)  ToolGap  InfGap   PlanGap
---------------------------------------------------------------------------------
LLM+Solver      1124.0    0.1870    0.3093      0.0    392.0      0.0
LLMAgent (3/5)   663.3    0.0870    0.2280      0.0      —        —
CodingAgent        —        —         —          —       —        —

### LLM+Solver per-seed detail

Seed   Reward  Brier(S)  Brier(E)  InfGap
42      550    0.2444    0.3978    1110.0
123    1140    0.1973    0.2781     420.0
456    1160    0.2010    0.2305     290.0
789    1130    0.1113    0.2857     140.0
1024   1640    0.1810    0.3546       0.0

### LLMAgent per-seed detail (partial — 3/5 seeds)

Seed   Reward  Brier(S)  Brier(E)
42      240    0.1084    0.2329
123     940    0.0769    0.2780
456     810    0.0757    0.1730
