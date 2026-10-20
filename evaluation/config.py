"""项目统一配置。路径均以项目根目录为基准。"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
GOLD_FILE = DATA_DIR / "evaluation.jsonl"

# AFQMC 二分类：0/1；默认将 1 视为正类
POSITIVE_LABEL = 1
REQUIRED_PREDICTION_FIELDS = ("id", "label", "prediction")
OPTIONAL_TEXT_FIELDS = ("sentence1", "sentence2")
