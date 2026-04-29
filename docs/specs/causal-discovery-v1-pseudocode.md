# Causal Discovery V1 Pseudocode

This document captures the high-level benchmark pseudocode for the `feat/causal-discovery` branch.

It is a spec artifact, not an implementation artifact. The purpose is to make later code verifiable against the benchmark design before library choices or runtime details start coupling pieces together.

## High-Level Benchmark Flow

```text
BUILD_BENCHMARK_INSTANCE(config):
    INPUT:
        d                  # number of variables
        k                  # number of edges, default d + 1
        n_obs              # observational sample count
        n_int              # rows returned per intervention
        weight_range       # e.g. (-2,-0.5) U (0.5,2)
        noise_var          # default 1.0
        faithfulness_eps   # near-unfaithfulness rejection threshold
        budget_slack       # x in min_interventions + x

    REPEAT:
        dag_internal = SAMPLE_RANDOM_DAG(d, k)
        cpdag_internal = DAG_TO_CPDAG(dag_internal)

        IF REJECT_GRAPH(dag_internal, cpdag_internal):
            CONTINUE

        scm_internal = PARAMETERIZE_LINEAR_GAUSSIAN_SCM(
            dag_internal,
            weight_range,
            noise_var
        )

        IF REJECT_SCM(scm_internal, faithfulness_eps):
            CONTINUE

        intervention_set_opt = COMPUTE_MIN_INTERVENTION_SET(
            dag_internal,
            cpdag_internal
        )

        IF REJECT_INTERVENTION_PROFILE(intervention_set_opt, cpdag_internal):
            CONTINUE

        permutation = SAMPLE_RANDOM_LABEL_PERMUTATION(d)
        dag_public = APPLY_LABEL_PERMUTATION(dag_internal, permutation)
        cpdag_public = APPLY_LABEL_PERMUTATION(cpdag_internal, permutation)
        scm_public = APPLY_LABEL_PERMUTATION(scm_internal, permutation)
        intervention_set_public = APPLY_LABEL_PERMUTATION(
            intervention_set_opt,
            permutation
        )

        obs_data = SAMPLE_OBSERVATIONAL_DATA(scm_public, n_obs)

        budget = SIZE(intervention_set_public) + budget_slack

        RETURN BENCHMARK_INSTANCE(
            scm = scm_public,
            true_dag = dag_public,
            observational_ceiling = cpdag_public,
            optimal_intervention_set = intervention_set_public,
            observational_data = obs_data,
            intervention_budget = budget,
            metadata = {
                d, k, n_obs, n_int, noise_var, faithfulness_eps, budget_slack
            }
        )
```

## Random DAG Generation

```text
SAMPLE_RANDOM_DAG(d, k):
    topo_order = SAMPLE_RANDOM_PERMUTATION([1..d])
    valid_edges = ALL_PAIRS(i -> j where i precedes j in topo_order)
    chosen_edges = SAMPLE_UNIFORMLY(valid_edges, k)
    RETURN DAG(nodes=1..d, edges=chosen_edges)
```

## Graph Rejection Filter

```text
REJECT_GRAPH(dag, cpdag):
    undirected_edges = GET_UNDIRECTED_EDGES(cpdag)

    IF cpdag == dag:
        RETURN TRUE

    IF COUNT(undirected_edges) < 2:
        RETURN TRUE

    IF ALL_UNDIRECTED_EDGES_SHARE_ONE_NODE(undirected_edges):
        RETURN TRUE

    IF DAG_IS_DISCONNECTED(dag):
        RETURN TRUE

    RETURN FALSE
```

## Linear-Gaussian SCM Parameterization

```text
PARAMETERIZE_LINEAR_GAUSSIAN_SCM(dag, weight_range, noise_var):
    FOR each node X_j in topological order:
        parents = PARENTS(X_j)
        weights = SAMPLE_EDGE_WEIGHTS(parents, weight_range)
        noise_j ~ Normal(0, noise_var)
        equation_j:
            X_j = SUM_i(weights[i,j] * X_i for X_i in parents) + noise_j
    RETURN SCM({equation_j for all j})
```

## SCM Rejection Filter

```text
REJECT_SCM(scm, faithfulness_eps):
    cov = IMPLIED_COVARIANCE_MATRIX(scm)

    IF COVARIANCE_IS_NEAR_SINGULAR(cov):
        RETURN TRUE

    partial_corrs = COMPUTE_RELEVANT_PARTIAL_CORRELATIONS(cov)

    IF ANY_ABS(partial_corrs) < faithfulness_eps for dependencies
       that should be structurally present:
        RETURN TRUE

    RETURN FALSE
```

## Minimum Intervention Set

```text
COMPUTE_MIN_INTERVENTION_SET(dag, cpdag):
    # under perfect single-node hard interventions
    # goal: fully orient cpdag into true dag
    RETURN OPTIMAL_NODE_SET_FROM_GRAPH_THEORY(dag, cpdag)
```

## Intervention Profile Rejection

```text
REJECT_INTERVENTION_PROFILE(intervention_set_opt, cpdag):
    # optional quality filter, not core theory
    IF SIZE(intervention_set_opt) == 0:
        RETURN TRUE

    RETURN FALSE
```

## Runtime Session

```text
RUNTIME_SESSION(instance, agent):
    data_obs = instance.observational_data
    budget = instance.intervention_budget
    n_int = instance.metadata.n_int

    agent.observe(data_obs)

    WHILE budget > 0:
        request = agent.next_action()

        IF request.type == "intervene":
            var = request.var
            val = request.val
            data_int = SAMPLE_INTERVENTIONAL_DATA(
                instance.scm,
                do(var = val),
                n_int
            )
            agent.observe_intervention_result(var, val, data_int)
            budget = budget - 1

        ELSE IF request.type == "submit_graph":
            graph_submission = request.graph_submission
            BREAK

        ELSE:
            ERROR("invalid agent action")

    IF no graph submitted:
        graph_submission = agent.final_graph_guess_or_empty()

    RETURN AGENT_OUTPUT(
        submission = graph_submission.with_interventions_used(
            instance.intervention_budget - budget
        )
    )
```

## Scoring

```text
SCORE(instance, agent_output):
    submission = agent_output.submission

    skeleton_score =
        SCORE_SKELETON(submission, instance.observational_ceiling)

    compelled_orientation_score =
        SCORE_COMPELLED_ORIENTATIONS(submission, instance.observational_ceiling)

    directed_dag_score =
        SCORE_DIRECTED_DAG(submission, instance.true_dag)

    dag_shd =
        SHD(submission, instance.true_dag)

    efficiency =
        SCORE_INTERVENTION_EFFICIENCY(
            used = submission.interventions_used,
            optimal = SIZE(instance.optimal_intervention_set)
        )

    RETURN {
        skeleton_score,
        compelled_orientation_score,
        directed_dag_score,
        dag_shd,
        efficiency
    }
```

## Verification Checklist

- `BUILD_BENCHMARK_INSTANCE` returns:
  - true DAG
  - CPDAG
  - SCM
  - observational data
  - optimal intervention set
  - budget
- agent never sees:
  - internal topological order
  - true DAG
  - CPDAG
  - optimal intervention set
- interventions return sampled rows, not oracle graph facts
- observational score is computed against CPDAG
- final/full score is computed against DAG
- efficiency uses optimal intervention set, not arbitrary budget
