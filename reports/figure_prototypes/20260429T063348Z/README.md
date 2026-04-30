# ACDB Figure Prototypes

These artifacts are for narrative inspection only. They are not final paper figures.

They were generated from current traces and may include partial or v0 runs.

## Generated Figures

- `hidden_agent_asymmetry`
  - `reports\figure_prototypes\20260429T063348Z\01_hidden_agent_asymmetry.pdf`
  - `reports\figure_prototypes\20260429T063348Z\01_hidden_agent_asymmetry.png`
- `layered_scoring_contract`
  - `reports\figure_prototypes\20260429T063348Z\02_layered_scoring_contract.pdf`
  - `reports\figure_prototypes\20260429T063348Z\02_layered_scoring_contract.png`
- `dag_cpdag_intervention`
  - `reports\figure_prototypes\20260429T063348Z\03_dag_cpdag_intervention.pdf`
  - `reports\figure_prototypes\20260429T063348Z\03_dag_cpdag_intervention.png`
- `representative_graph_output`
  - `reports\figure_prototypes\20260429T063348Z\04_representative_graph_output.pdf`
  - `reports\figure_prototypes\20260429T063348Z\04_representative_graph_output.png`
- `random_floor_density`
  - `reports\figure_prototypes\20260429T063348Z\05_random_floor_density.pdf`
  - `reports\figure_prototypes\20260429T063348Z\05_random_floor_density.png`
- `v0_v1_ladder_regions`
  - `reports\figure_prototypes\20260429T063348Z\08_v0_v1_ladder_regions.pdf`
  - `reports\figure_prototypes\20260429T063348Z\08_v0_v1_ladder_regions.png`
- `precision_recall_shd`
  - `reports\figure_prototypes\20260429T063348Z\06_precision_recall_shd.pdf`
  - `reports\figure_prototypes\20260429T063348Z\06_precision_recall_shd.png`
- `active_gain`
  - `reports\figure_prototypes\20260429T063348Z\07_active_gain.pdf`
  - `reports\figure_prototypes\20260429T063348Z\07_active_gain.png`

## Representative Graph

```json
{
  "status": "data-derived",
  "level": 0,
  "seed": 1276453582,
  "source_trace": "deepseek_v4_flash_ladder_2seed",
  "true_edges": "X0->X3, X1->X0, X3->X2",
  "optimal_intervention_set": [
    1
  ],
  "pc_directed_edges": "X0->X3, X1->X0, X3->X2",
  "llm_directed_edges": "X0->X1, X0->X2, X1->X2, X2->X3"
}
```

## Source Files

- `traces\ladder\deepseek_v4_flash_ladder_2seed\events.jsonl`
- `traces\ladder\deepseek_v4_flash_ladder_2seed\results_long.csv`
- `traces\ladder\full_ladder_toolcall_run1\events.jsonl`
- `traces\ladder\full_ladder_toolcall_run1\results_long.csv`
- `traces\ladder\sonnet46_full_ladder_run1\events.jsonl`
- `traces\ladder\sonnet46_full_ladder_run1\results_long.csv`
- `traces\ladder\sonnet46_provider_smoke\events.jsonl`
- `traces\ladder\sonnet46_provider_smoke\results_long.csv`
