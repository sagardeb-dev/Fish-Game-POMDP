import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from run_corr_obs_probe import (  # noqa: E402
    build_corr_obs_session_prompt,
    corr_std_summary,
    format_corr_matrix,
    format_std_row,
    load_manifest_seed_map,
    submission_from_model_payload,
)


def test_corr_std_summary_uses_sample_std_and_rounded_symmetric_corr():
    data = np.array(
        [
            [1.0, 1.0, 2.0],
            [2.0, 3.0, 3.0],
            [3.0, 5.0, 5.0],
            [4.0, 7.0, 9.0],
        ]
    )

    corr, std = corr_std_summary(data)

    assert np.allclose(corr, corr.T)
    assert np.array_equal(corr, np.round(corr, 3))
    assert np.array_equal(std, np.round(np.std(data, axis=0, ddof=1), 3))
    assert std.tolist() == [1.291, 2.582, 3.096]


def test_corr_obs_formatting_is_labelled_and_deterministic():
    names = ("X0", "X1")
    std = np.array([1.23456, 0.98765])
    corr = np.array([[1.0, -0.12345], [-0.12345, 1.0]])

    assert format_std_row(names, std) == "X0=1.235, X1=0.988"
    assert format_corr_matrix(names, corr) == (
        "           X0      X1\n"
        "   X0   1.000  -0.123\n"
        "   X1  -0.123   1.000"
    )


def test_corr_obs_session_prompt_contains_no_raw_rows():
    names = ("X0", "X1")
    prompt = build_corr_obs_session_prompt(
        variable_names=names,
        n_obs=25,
        std=np.array([1.0, 2.0]),
        corr=np.array([[1.0, 0.5], [0.5, 1.0]]),
    )
    payload = json.loads(prompt)

    assert payload["variables"] == ["X0", "X1"]
    assert payload["n_obs"] == 25
    assert "observational_data" not in payload
    assert payload["sample_std_ddof1_rounded_3dp"] == [1.0, 2.0]
    assert payload["correlation_matrix_rounded_3dp"] == [[1.0, 0.5], [0.5, 1.0]]


def test_manifest_seed_map_uses_first_n_selected_level_seeds(tmp_path):
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "seed_map": {
                    "1": [101, 102, 103],
                    "3": [301, 302, 303],
                    "5": [501, 502, 503],
                }
            }
        ),
        encoding="utf-8",
    )

    assert load_manifest_seed_map(manifest, [1, 5], 2) == {
        1: [101, 102],
        5: [501, 502],
    }


def test_submission_from_model_payload_sanitizes_to_valid_graph_submission():
    payload = {
        "action": "submit_graph",
        "directed_edges": [
            [0, 1],
            [1, 0],
            [1, 2],
            [2, 0],
            [3, 3],
            [4, 0],
            ["bad", 2],
        ],
        "undirected_edges": [[1, 0], [2, 3], [2, 2], [0, 4], ["x"]],
    }

    submission, diagnostics = submission_from_model_payload(payload, d=4)

    assert submission.interventions_used == 0
    assert submission.directed_edges == frozenset({(0, 1), (1, 2)})
    assert submission.undirected_edges == frozenset({(2, 3)})
    assert diagnostics == {
        "sanitized_overlap_count": 2,
        "sanitized_invalid_count": 6,
        "sanitized_cycle_count": 1,
    }
