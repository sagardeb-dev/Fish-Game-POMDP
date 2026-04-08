import subprocess
import sys


def test_demo_generate_smoke():
    result = subprocess.run(
        [sys.executable, "-m", "world_gen.demo_generate", "--level", "0.5", "--seed", "42"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "WORLD GENERATION DEMO" in result.stdout
    assert "Reward Schedule:" in result.stdout


def test_demo_pipeline_smoke():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "world_gen.demo_pipeline",
            "--level",
            "0.5",
            "--seed",
            "42",
            "--d-prime",
            "1.8",
            "--sensor-zones",
            "2",
            "--episodes",
            "1",
            "--agents",
            "random",
            "reasoner",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "WORLD GENERATION PIPELINE" in result.stdout
    assert "Overrides:" in result.stdout
    assert "Generated World:" in result.stdout
    assert "SUMMARY" in result.stdout
