"""
rp-pipeline -- run the full research-project pipeline in one command.

Default (no flags):  realign > build-realigned-source > feature-matrix > train
Add --plots, --shap, --ablation, --tune for optional stages, or --all for everything.
Use --skip-to STAGE to resume from a specific stage.

Orchestration notes (avoid duplicate work):
- When ``--tune`` runs ``train``, that stage already regenerates publication figures
  (iAUC distribution + OOF scatter); the separate ``plots`` stage is skipped in that case.
- When ``--shap`` is set, the full SHAP report runs as the ``shap`` stage only;
  ``train`` is not given ``--shap`` (which would duplicate TreeExplainer work and
  overlap with figures 01–02 under ``plots/shap/``). Use ``rp-train --shap`` alone
  for the lightweight flat ``shap_summary.png`` + ``shap_dependence_top4.png`` only.
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


def resolve_pipeline_plan(
    skip_to: str | None,
    tune: bool,
    plots: bool,
    shap: bool,
) -> list[str]:
    """
    Ordered stages to run after de-duplication rules.

    Callers must expand ``--all`` on the argparse namespace *before* calling this
    (set tune, plots, shap, ablation all True). Ablation only affects ``train`` flags,
    not stage order.
    """
    start = STAGES.index(skip_to) if skip_to else 0

    active: set[str] = set(STAGES[start:4])
    if plots or start == STAGES.index("plots"):
        active.add("plots")
    if shap or start >= STAGES.index("shap"):
        active.add("shap")

    plan = [s for s in STAGES[start:] if s in active]
    # train already runs publication plots when --tune (see train.main)
    if tune and "train" in plan:
        plan = [s for s in plan if s != "plots"]
    return plan


def train_argv_for_pipeline(tune: bool, ablation: bool) -> list[str]:
    """``sys.argv`` fragment for ``research_project.training.train:main`` (pipeline only)."""
    argv = ["rp-train"]
    if tune:
        argv.append("--tune")
    if ablation:
        argv.append("--ablation")
    return argv


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
    p.add_argument("--shap", action="store_true", help="Run full SHAP report after training")
    p.add_argument("--ablation", action="store_true", help="Run ablation study in train stage")
    p.add_argument("--all", action="store_true", help="Enable every optional stage and flag")
    args = p.parse_args()

    if args.all:
        args.tune = args.plots = args.shap = args.ablation = True

    plan = resolve_pipeline_plan(
        args.skip_to,
        args.tune,
        args.plots,
        args.shap,
    )

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
        train_argv = train_argv_for_pipeline(args.tune, args.ablation)
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
