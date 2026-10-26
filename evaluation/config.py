from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = ROOT / "outputs"
GOLD_FILES = {
    "evaluation": DATA_DIR / "evaluation.jsonl",
    "validation": DATA_DIR / "validation.jsonl",
}
PREDICTION_DIRS = {
    "evaluation": RESULTS_DIR / "evaluation",
    "validation": RESULTS_DIR / "validation",
}
POSITIVE_LABEL = 1
