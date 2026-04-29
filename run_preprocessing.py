"""
Top-level CLI to run the full preprocessing pipeline.

Usage
-----
    python run_preprocessing.py

This script is kept at the project root (rather than inside ``src/``)
so that the preprocessor is imported as ``src.data.preprocessor``,
not as ``__main__``.  That matters because pickled ``IDMapper``
instances reference the class's module path; if the module is
``__main__`` at write time, later sessions cannot unpickle them.
"""

from __future__ import annotations

import logging

from src.data.preprocessor import print_summary, run_preprocessing


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    summary = run_preprocessing()
    print_summary(summary)


if __name__ == "__main__":
    main()
