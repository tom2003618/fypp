import os
import runpy
import subprocess
import sys
from pathlib import Path


PROJECT_MARKERS = (
    ".git",
    "canada_federal_vote_share_2000_2025.csv",
    "econ_2021_2025.py",
)

BASE_PACKAGES = (
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("matplotlib", "matplotlib"),
    ("openpyxl", "openpyxl"),
)

SCRIPT_PREREQUISITES = {
    "factor_based_2025_prediction_and_projection.py": (
        ("outputs/national_vote_share_clean.csv", "election_workflow.py"),
        ("outputs_demographics/econ_AB_BC_ON_QC_2021_2025.csv", "economic_summary_workflow.py"),
        ("outputs_demographics/x_nlp_party_summary_2021_2025.csv", "x_nlp_data_recovery.py"),
    ),
    "education_attainment_2025_prediction.py": (
        ("outputs/national_vote_share_clean.csv", "election_workflow.py"),
    ),
    "voter_generation_preference_model.py": (
        ("outputs/national_vote_share_clean.csv", "election_workflow.py"),
    ),
    "x_nlp_vote_prediction.py": (
        ("outputs/national_vote_share_clean.csv", "election_workflow.py"),
        ("outputs_demographics/x_nlp_party_summary_2021_2025.csv", "x_nlp_data_recovery.py"),
    ),
    "multi_factor_ensemble.py": (
        ("outputs/national_vote_share_clean.csv", "election_workflow.py"),
        ("outputs_demographics/factor_based_2025_predictions.csv", "factor_based_2025_prediction_and_projection.py"),
        ("outputs_demographics/education_attainment_predictions_2025.csv", "education_attainment_2025_prediction.py"),
        ("outputs_demographics/voter_generation_party_preference_model_2011_2025.csv", "voter_generation_preference_model.py"),
        ("outputs/prediction_2025_vs_actual.csv", "prediction_2025_vs_actual.py"),
    ),
}


def ensure_base_packages():
    missing = []
    for module_name, pip_name in BASE_PACKAGES:
        try:
            __import__(module_name)
        except ModuleNotFoundError:
            missing.append(pip_name)

    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", *missing])


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    helper_dir = Path(__file__).resolve().parent
    candidates = [start, *start.parents, helper_dir, *helper_dir.parents]
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if any((candidate / marker).exists() for marker in PROJECT_MARKERS):
            return candidate
    return start


def _resolve_script_path(project_root: Path, script_name: str) -> Path:
    candidates = [
        project_root / script_name,
        project_root.parent / script_name,
    ]
    script = next((path for path in candidates if path.exists()), None)
    if script is None:
        raise FileNotFoundError(f"Cannot find script: {script_name}")
    return script


def _run_script_path(project_root: Path, script_path: Path):
    prev_cwd = Path.cwd()
    os.chdir(project_root)
    try:
        return runpy.run_path(str(script_path), run_name="__main__")
    finally:
        os.chdir(prev_cwd)


def _ensure_prerequisites(project_root: Path, script_name: str, active_scripts: set[str]):
    for rel_path, producer_script in SCRIPT_PREREQUISITES.get(script_name, ()):
        if (project_root / rel_path).exists():
            continue
        if producer_script in active_scripts:
            continue
        active_scripts.add(producer_script)
        try:
            _ensure_prerequisites(project_root, producer_script, active_scripts)
            _run_script_path(project_root, _resolve_script_path(project_root, producer_script))
        finally:
            active_scripts.discard(producer_script)


def run_project_script(script_name: str, install_base_packages: bool = True):
    if install_base_packages:
        ensure_base_packages()

    project_root = find_project_root()
    _ensure_prerequisites(project_root, script_name, {script_name})
    return _run_script_path(project_root, _resolve_script_path(project_root, script_name))
