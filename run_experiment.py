"""Convenience wrapper for baseline experiment runs."""

from __future__ import annotations

import sys

from src.main import main


if __name__ == "__main__":
    if "--task" not in sys.argv:
        sys.argv.extend(["--task", "experiment"])
    main()
