"""
rp-pipeline -- run the full research-project pipeline in one command.

Default (no flags):  realign > build-realigned-source > feature-matrix > train
Add --plots, --shap, --ablation, --tune for optional stages, or --all for everything.
Use --skip-to STAGE to resume from a specific stage.
"""

import argparse
import sys
import time

STAGES = [
    "realign",
    "build-realigned-source",
    "feature-matrix",
    "train",
    "plots",
    "shap",
]


def _banner(name: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  STAGE: {name}")
    print(f"{'=' * 72}\n")


def _run_stage(name: str, func, **kwargs) -> None:
    _banner(name)
    t0 = time.perf_counter()
    func(**kwargs)
    elapsed = time.perf_counter() - t0
    print(f"\n  [{name}] done in {elapsed:.1f}s")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--skip-to",
        choices=STAGES,
        default=None,
        help="Resume from this stage (skip earlier stages)",
    )
    p.add_argument("--tune", action="store_true", help="Enable Optuna tuning in train stage")
    p.add_argument("--plots", action="store_true", help="Run publication plots after training")
    p.add_argument("--shap", action="store_true", help="Run SHAP report after training")
    p.add_argument("--ablation", action="store_true", help="Run ablation study in train stage")
    p.add_argument("--all", action="store_true", help="Enable every optional stage and flag")
    args = p.parse_args()

    if args.all:
        args.tune = args.plots = args.shap = args.ablation = True

    # Determine which stages to run
    start = STAGES.index(args.skip_to) if args.skip_to else 0

    # Core stages (realign through train) are always active when in range.
    # Optional post-training stages are active if flagged OR if --skip-to targets them.
    active = set(STAGES[start:4])
    if args.plots or start == STAGES.index("plots"):
        active.add("plots")
    if args.shap or start >= STAGES.index("shap"):
        active.add("shap")

    plan = [s for s in STAGES[start:] if s in active]

    print("Pipeline plan:", " > ".join(plan))
    print()

    # --- data pipeline ------------------------------------------------
    if "realign" in plan:
        from research_project.data_pipeline.realign import main as realign_main

        _run_stage("realign", realign_main)

    if "build-realigned-source" in plan:
        from research_project.data_pipeline.build_realigned_source import main as brs_main

        _run_stage("build-realigned-source", brs_main)

    if "feature-matrix" in plan:
        from research_project.data_pipeline.feature_matrix import main as fm_main

        _run_stage("feature-matrix", fm_main)

    # --- training -----------------------------------------------------
    if "train" in plan:
        # Build sys.argv for train's own argparse
        train_argv = ["rp-train"]
        if args.tune:
            train_argv.append("--tune")
        if args.shap:
            train_argv.append("--shap")
        if args.ablation:
            train_argv.append("--ablation")

        saved = sys.argv
        sys.argv = train_argv
        try:
            from research_project.training.train import main as train_main

            _run_stage("train", train_main)
        finally:
            sys.argv = saved

    # --- optional post-training stages --------------------------------
    if "plots" in plan:
        saved = sys.argv
        sys.argv = ["rp-plots", "all"]
        try:
            from research_project.training.plots import main as plots_main

            _run_stage("plots", plots_main)
        finally:
            sys.argv = saved

    if "shap" in plan:
        from research_project.analysis.shap_analysis import main as shap_main

        _run_stage("shap", shap_main)

    print("\n" + "=" * 72)
    print("  Pipeline complete.")
    print("=" * 72)
