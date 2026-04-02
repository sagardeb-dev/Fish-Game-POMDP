# Baseline Results (BENCHMARK_CONFIG, full tools, 5 seeds)

## Summary

Agent                Mean    Std  Per-seed
--------------------------------------------------------------------------------
Random              472.6 249.6  s42=116  s123=529  s456=811  s789=272  s1024=635
NaivePattern        472.6 331.5  s42=371  s123=22  s456=1013  s789=627  s1024=330
CausalLearner      1480.0 141.4  s42=1430  s123=1590  s456=1450  s789=1260  s1024=1670
CausalReasoner     1582.0  82.8  s42=1690  s123=1590  s456=1450  s789=1540  s1024=1640
Oracle             1716.0  53.1  s42=1750  s123=1620  s456=1740  s789=1770  s1024=1700

## Expected ordering
Random < NaivePattern < CausalLearner <= CausalReasoner <= Oracle

## LLM agents (to be filled in after running run_llm_benchmark.py)
Agent                Mean    Std  Per-seed
--------------------------------------------------------------------------------
LLMAgent            720.0 456.6  s42=650  s123=130  s456=1530  s789=540  s1024=750
LLM+Solver         1524.0 220.1  s42=1750  s123=1620  s456=1450  s789=1130  s1024=1670
