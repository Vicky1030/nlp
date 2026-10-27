"""Backward-compatible alias; the documented unique entry is run_all.py."""
from run_all import main


if __name__ == "__main__":
    raise SystemExit(main())
