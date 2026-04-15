# demos

CLI entrypoints for inspecting generated worlds and running baseline agents on them.

## Entrypoints

| Script | Purpose |
|---|---|
| `demo_generate.py` | Generate one world and print its knobs, distributions, transitions, and validator result |
| `demo_pipeline.py` | Generate a world, validate it, run selected non-LLM baselines, print per-agent metrics |

Both accept the same world-tuning args:

| Arg | Effect |
|---|---|
| `--level` | Curriculum difficulty (0.0 to 1.0) |
| `--seed` | World generation seed |
| `--d-prime` | Override observation informativeness |
| `--transition-alpha` | Override transition stochasticity |
| `--sensor-zones` | Override visible zones per day |
| `--trap-strength` | Override confound strength |

`demo_pipeline.py` additionally accepts:

| Arg | Effect |
|---|---|
| `--episodes` | Number of episodes per agent |
| `--agents` | Space-separated list: random, naive, learner, reasoner, oracle |
