#!/usr/bin/env python3
"""
Generate static charts for a FLUXSynID face benchmark result folder.

Example:
    python plot_flux_benchmark_results.py results\\face_model_benchmark\\flux_3000_t0800884_fast
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation.flux_benchmark_plots import (  # noqa: E402
    BenchmarkPlotError,
    generate_benchmark_chart_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate buyer-friendly charts for FLUXSynID benchmark results."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Benchmark output folder containing comparison.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Chart output folder. Defaults to <run_dir>/charts.",
    )
    parser.add_argument(
        "--top-models",
        nargs="+",
        default=None,
        help="Models to include in the overlay chart. Defaults to best two by FAR/FRR.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Report title shown in chart headings and executive_summary.md.",
    )
    parser.add_argument(
        "--format",
        choices=("png", "svg"),
        default="png",
        help="Chart image format. Defaults to png.",
    )
    parser.add_argument(
        "--language",
        choices=("en", "tr"),
        default="en",
        help="Language for chart labels and executive summary. Defaults to en.",
    )
    args = parser.parse_args()

    try:
        artifacts = generate_benchmark_chart_report(
            args.run_dir,
            output_dir=args.output_dir,
            title=args.title,
            top_models=args.top_models,
            image_format=args.format,
            language=args.language,
        )
    except BenchmarkPlotError as exc:
        print(f"Benchmark chart error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI should fail with a clear message.
        print(f"Benchmark chart generation failed: {exc}", file=sys.stderr)
        return 1

    print("\nFLUXSynID benchmark chart report generated.")
    print(f"  Output: {artifacts.output_dir}")
    print(f"  Executive summary: {artifacts.executive_summary}")
    print(f"  Buyer table: {artifacts.buyer_summary_table}")
    print(f"  Turkish buyer table: {artifacts.buyer_summary_table_tr}")
    for name, path in artifacts.charts.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
