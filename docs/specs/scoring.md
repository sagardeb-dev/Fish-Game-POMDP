# Scoring Spec

This benchmark uses one scoring contract for every agent and baseline.

```text
GraphSubmission -> score_submission(instance, submission) -> ScoreReport
```

No algorithm gets a custom scoring path. PC, LLM agents, oracle baselines, and
active learners all submit the same object.

## GraphSubmission

```text
num_nodes: int
directed_edges: frozenset[(src, dst)]
undirected_edges: frozenset[(a, b)]
interventions_used: int
```

Undirected edges are canonicalized as `(min_node, max_node)`.

Validation:

- node ids are in range
- no self-loops
- no directed 2-cycles
- no pair is both directed and undirected
- directed portion is acyclic
- `interventions_used >= 0`

Adjacency matrix convention:

```text
matrix[i][j] = 1, matrix[j][i] = 0  => i -> j
matrix[i][j] = 1, matrix[j][i] = 1  => i -- j
matrix[i][j] = 0, matrix[j][i] = 0  => no edge
```

## ScoreReport

```text
skeleton_precision
skeleton_recall
skeleton_f1

compelled_precision
compelled_recall
compelled_f1

directed_precision
directed_recall
directed_f1

dag_shd
interventions_used
optimal_interventions
efficiency
```

## Observational Layer

The observational ceiling is the true CPDAG. Observation alone should not be
expected to identify orientations inside the Markov equivalence class.

Skeleton score compares submitted adjacencies against the true CPDAG skeleton.
Direction is ignored. Missing and extra adjacencies are both penalized.

Compelled-orientation score compares only edges directed in the true CPDAG.
Orientations submitted for CPDAG-undirected edges are ignored here because those
edges are observationally ambiguous. They are handled by the DAG layer.

## DAG Layer

Directed DAG score compares submitted directed edges against the true DAG.
Submitted undirected edges are unresolved; they are not counted as correct
directed edges.

DAG SHD is an edge edit distance against the true DAG:

```text
Missing edge:     true DAG has A -> B, submission has nothing          => +1
Extra edge:       submission has A -> B or A -- B, true DAG has nothing => +1
Reversed edge:    true DAG has A -> B, submission has B -> A            => +1
Unresolved edge:  true DAG has A -> B, submission has A -- B            => +1
```

A reversed edge is one error, not missing plus extra.

## Efficiency Layer

Efficiency compares interventions used against the precomputed minimum
intervention set size.

```text
if optimal == 0 and used == 0: 1.0
if optimal == 0 and used > 0:  0.0
otherwise:                    optimal / max(used, optimal)
```

The budget itself is enforced by runtime. The efficiency score measures whether
the agent needed more interventions than the graph-theoretic minimum.
