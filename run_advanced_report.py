"""Build one validation comparison table from all recommender experiments.

This script only reads already-computed validation metrics.  It deliberately
does not touch the test split, so it is safe to rerun while developing models.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("artifacts/results")


def _read_json(name: str):
    with (RESULTS_DIR / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    rows: list[dict] = []

    rows.extend(
        [
            _read_json("popularity_val.json"),
            _read_json("content_based_val.json"),
            _read_json("collaborative_val.json"),
            _read_json("hybrid_val.json"),
            *_read_json("retrieve_rank_val.json"),
            _read_json("ease_val.json"),
            _read_json("three_channel_rrf_val.json"),
            *_read_json("torch_cf_val.json"),
        ]
    )

    keep = [
        "model",
        "precision@10",
        "recall@10",
        "ndcg@10",
        "hit_rate@10",
        "coverage@10",
        "novelty@10",
        "diversity@10",
        "precision@20",
        "recall@20",
        "ndcg@20",
        "hit_rate@20",
        "coverage@20",
        "novelty@20",
        "diversity@20",
    ]
    table = pd.DataFrame(rows)[keep].sort_values("ndcg@10", ascending=False)
    table.insert(0, "rank_by_ndcg@10", range(1, len(table) + 1))

    als_ndcg = float(
        table.loc[
            table["model"].str.startswith("CollaborativeFiltering"), "ndcg@10"
        ].iloc[0]
    )
    hybrid_ndcg = float(
        table.loc[table["model"].str.startswith("Hybrid("), "ndcg@10"].iloc[0]
    )
    table["ndcg@10_lift_vs_als_pct"] = (table["ndcg@10"] / als_ndcg - 1.0) * 100.0
    table["ndcg@10_lift_vs_linear_hybrid_pct"] = (
        table["ndcg@10"] / hybrid_ndcg - 1.0
    ) * 100.0

    csv_path = RESULTS_DIR / "advanced_model_comparison_val.csv"
    json_path = RESULTS_DIR / "advanced_model_comparison_val.json"
    md_path = RESULTS_DIR / "advanced_model_comparison_val.md"
    table.to_csv(csv_path, index=False)
    table.to_json(json_path, orient="records", indent=2)

    display_cols = [
        "rank_by_ndcg@10",
        "model",
        "precision@10",
        "recall@10",
        "ndcg@10",
        "precision@20",
        "recall@20",
        "ndcg@20",
        "coverage@20",
        "novelty@20",
    ]
    display = table[display_cols]
    markdown_lines = [
        "# Advanced recommender validation comparison",
        "",
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join("---" for _ in display.columns) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        cells = [f"{value:.6f}" if isinstance(value, float) else str(value) for value in row]
        markdown_lines.append("| " + " | ".join(cells) + " |")
    md_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    print(table[display_cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nWrote {csv_path}, {json_path}, and {md_path}")


if __name__ == "__main__":
    main()
