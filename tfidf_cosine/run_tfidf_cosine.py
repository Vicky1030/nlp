"""Package-root wrapper for the portable one-command TF-IDF pipeline."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    args = sys.argv[1:]
    if "--data-dir" not in args or "--output-root" not in args:
        raise SystemExit("--data-dir and --output-root are required")
    data_dir = Path(args[args.index("--data-dir") + 1]).resolve()
    output = Path(args[args.index("--output-root") + 1]).resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing output directory: {output}")
    hashes = json.loads((base / "reports" / "processed_hashes.json").read_text(encoding="utf-8"))
    for name, spec in hashes.items():
        path = data_dir / f"{name}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing external unified input: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
            raise ValueError(f"{name}: SHA-256 mismatch; do not re-split the unified data")
    entry = base / "src" / "run_tfidf_cosine.py"
    command = [sys.executable, str(entry), *args]
    # The unified child owns directory creation and writes stage logs/manifest.
    result = subprocess.run(command, cwd=base, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    raise SystemExit(result.returncode)
