import csv
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from run_ladder import (
    _sanitize_submission_edges,
    allowed_actions_for_method,
    make_work_items,
    make_action_response_schema,
    make_action_tool,
    mean_shift_threshold,
    summarize_results,
)


def test_action_response_schema_requires_stable_action_fields():
    response_schema = make_action_response_schema(frozenset({"intervene", "submit_graph"}))
    schema = response_schema["schema"]
    assert response_schema["strict"] is True
    assert schema["additionalProperties"] is False
    assert schema["properties"]["action"]["enum"] == ["intervene", "submit_graph"]
    assert set(schema["required"]) == {
        "action",
        "var",
        "value",
        "directed_edges",
        "undirected_edges",
        "reasoning_summary",
    }


def test_action_tool_uses_method_specific_strict_schema_contract():
    raw_tool = make_action_tool(allowed_actions_for_method("llm_raw"))
    assert raw_tool["type"] == "function"
    function = raw_tool["function"]
    assert function["name"] == "causal_discovery_action"
    assert function["strict"] is True
    assert function["parameters"]["properties"]["action"]["enum"] == [
        "intervene",
        "submit_graph",
    ]


def test_action_tool_schema_fields_match_llm_policy():
    expected = {
        "llm_raw_obs": {
            "action",
            "directed_edges",
            "undirected_edges",
            "reasoning_summary",
        },
        "llm_raw": {
            "action",
            "var",
            "value",
            "directed_edges",
            "undirected_edges",
            "reasoning_summary",
        },
        "llm_stats_obs": {
            "action",
            "i",
            "j",
            "conditioning_on",
            "alpha",
            "directed_edges",
            "undirected_edges",
            "reasoning_summary",
        },
        "llm_stats": {
            "action",
            "var",
            "value",
            "i",
            "j",
            "conditioning_on",
            "alpha",
            "directed_edges",
            "undirected_edges",
            "reasoning_summary",
        },
    }
    for method, fields in expected.items():
        schema = make_action_response_schema(allowed_actions_for_method(method))["schema"]
        assert set(schema["properties"]) == fields
        assert set(schema["required"]) == fields


def test_raw_methods_do_not_expose_stats_actions():
    assert allowed_actions_for_method("llm_raw") == frozenset(
        {"intervene", "submit_graph"}
    )
    assert allowed_actions_for_method("llm_raw_obs") == frozenset({"submit_graph"})


def test_hybrid_methods_have_expected_action_spaces():
    assert allowed_actions_for_method("pc_cpdag_llm") == frozenset(
        {"intervene", "submit_graph"}
    )
    assert allowed_actions_for_method("llm_stats_cpdag_greedy") == frozenset(
        {"correlation", "partial_correlation", "independence_test", "submit_graph"}
    )


def test_reasoning_summary_is_required_for_non_submit_actions():
    schema = make_action_response_schema(frozenset({"correlation"}))["schema"]
    assert set(schema["required"]) == {"action", "i", "j", "reasoning_summary"}


def test_make_work_items_includes_hybrid_active_methods():
    items = make_work_items([3], {3: [123]}, ["openrouter/test/model"])
    methods = {item.method for item in items if item.panel == "active"}
    assert "pc_cpdag_llm" in methods
    assert "llm_stats_cpdag_greedy" in methods


def test_sanitize_submission_edges_keeps_directed_and_drops_conflicting_undirected():
    directed, undirected, dropped = _sanitize_submission_edges(
        directed_edges=[(1, 4), (0, 1)],
        undirected_edges=[(4, 1), (2, 3)],
    )
    assert dropped == 1
    assert directed == frozenset({(1, 4), (0, 1)})
    assert undirected == frozenset({(2, 3)})


def test_mean_shift_threshold_matches_l1_sanity_value():
    threshold = mean_shift_threshold(obs_var=1.0, n_obs=25, int_var=1.0, n_int=15)
    assert abs(threshold - 0.640121556847375) < 1e-12


def test_mean_shift_threshold_decreases_with_more_samples():
    small_sample = mean_shift_threshold(obs_var=1.0, n_obs=25, int_var=1.0, n_int=15)
    large_sample = mean_shift_threshold(obs_var=1.0, n_obs=50, int_var=1.0, n_int=30)
    assert large_sample < small_sample


def test_mean_shift_threshold_increases_with_variance():
    low_variance = mean_shift_threshold(obs_var=1.0, n_obs=25, int_var=1.0, n_int=15)
    high_variance = mean_shift_threshold(obs_var=2.0, n_obs=25, int_var=2.0, n_int=15)
    assert high_variance > low_variance


def test_mean_shift_threshold_is_symmetric_between_samples():
    left = mean_shift_threshold(obs_var=1.2, n_obs=25, int_var=0.8, n_int=15)
    right = mean_shift_threshold(obs_var=0.8, n_obs=15, int_var=1.2, n_int=25)
    assert left == right


def test_summarize_results_tracks_failed_rows_and_behavior_means():
    rows = [
        {
            "panel": "active",
            "level": "0",
            "method": "llm_stats",
            "model": "gpt-5.4",
            "status": "success",
            "skeleton_f1": "0.8",
            "directed_f1": "0.2",
            "dag_shd": "4.0",
            "efficiency": "1.0",
            "submit_directed": "2",
            "submit_undirected": "3",
            "interventions_used": "0",
            "total_actions": "3",
            "stats_actions": "2",
            "intervene_actions": "0",
            "submit_actions": "1",
            "sanitized_overlap_count": "1",
        },
        {
            "panel": "active",
            "level": "0",
            "method": "llm_stats",
            "model": "gpt-5.4",
            "status": "failed",
            "skeleton_f1": "",
            "directed_f1": "",
            "dag_shd": "",
            "efficiency": "",
            "submit_directed": "",
            "submit_undirected": "",
            "interventions_used": "",
            "total_actions": "",
            "stats_actions": "",
            "intervene_actions": "",
            "submit_actions": "",
            "sanitized_overlap_count": "",
        },
    ]
    out_dir = ROOT / "traces" / "ladder"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"_summary_test_{uuid.uuid4().hex}.csv"
    try:
        summarize_results(rows, out_csv)
        with out_csv.open("r", encoding="utf-8", newline="") as f:
            parsed = list(csv.DictReader(f))
    finally:
        pass
    assert len(parsed) == 1
    row = parsed[0]
    assert row["n_total"] == "2"
    assert row["n_success"] == "1"
    assert row["n_failed"] == "1"
    assert float(row["submit_directed_mean"]) == 2.0
    assert float(row["stats_actions_mean"]) == 2.0
