# tests

Test suite for the benchmark. Run via `python -m scripts.run_tests`.

## Layout

```
tests/
  unit/
    fishing_game/
      test_fishing_game.py    config, POMDP, simulator, evaluator, baselines, ablation, LLM+Solver
    world_gen/
      test_knobs.py           curriculum_knobs determinism, constraint enforcement, reward fixedness
      test_validator.py       structural validation, malformed config rejection, entropy thresholds
  integration/
    test_world_gen_generator.py     generated config instantiates FishingGameEnv, seed determinism
    test_world_gen_demo_smoke.py    demo_generate and demo_pipeline CLI smoke tests
```

## Running

```bash
python -m scripts.run_tests                              # all tests
python -m scripts.run_tests tests/unit/ -q               # unit only
python -m scripts.run_tests tests/integration/ -q        # integration only
python -m scripts.run_tests tests/unit/fishing_game/ -v  # fishing_game verbose
```
