"""Pipeline stage order and train argv (no duplicate SHAP / plots)."""

import unittest

from research_project.pipeline import (
    STAGES,
    resolve_pipeline_plan,
    train_argv_for_pipeline,
)


class TestResolvePipelinePlan(unittest.TestCase):
    def test_all_drops_plots_when_train_tuned(self) -> None:
        plan = resolve_pipeline_plan(None, tune=True, plots=True, shap=True)
        self.assertIn("train", plan)
        self.assertIn("shap", plan)
        self.assertNotIn("plots", plan)

    def test_plots_kept_without_tune(self) -> None:
        plan = resolve_pipeline_plan(None, tune=False, plots=True, shap=False)
        self.assertIn("plots", plan)
        self.assertIn("train", plan)

    def test_default_no_optional_stages(self) -> None:
        plan = resolve_pipeline_plan(None, tune=False, plots=False, shap=False)
        self.assertEqual(plan, STAGES[:4])


class TestTrainArgv(unittest.TestCase):
    def test_never_includes_shap(self) -> None:
        self.assertEqual(train_argv_for_pipeline(False, False), ["rp-train"])
        self.assertEqual(train_argv_for_pipeline(True, False), ["rp-train", "--tune"])
        self.assertEqual(
            train_argv_for_pipeline(True, True),
            ["rp-train", "--tune", "--ablation"],
        )
        for argv in (
            train_argv_for_pipeline(False, False),
            train_argv_for_pipeline(True, True),
        ):
            self.assertNotIn("--shap", argv)


if __name__ == "__main__":
    unittest.main()
