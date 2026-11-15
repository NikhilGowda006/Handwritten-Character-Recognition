"""Tests for the Streamlit dashboard and model-performance page."""

import json
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class StreamlitAppTests(unittest.TestCase):
    def run_app(self) -> AppTest:
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run()
        self.assertEqual(len(app.exception), 0)
        return app

    def test_dashboard_starts_with_measured_metrics(self) -> None:
        app = self.run_app()
        metrics_path = PROJECT_ROOT / "results" / "evaluation.json"
        measured = json.loads(metrics_path.read_text(encoding="utf-8"))
        displayed_metrics = {metric.label: metric.value for metric in app.metric}

        self.assertIn("HANDWRITTEN CHARACTER RECOGNITION", app.title[0].value)
        self.assertEqual(
            displayed_metrics["Test accuracy"],
            f"{measured['test_accuracy']:.2%}",
        )
        self.assertEqual(
            displayed_metrics["Test samples"],
            f"{measured['test_sample_count']:,}",
        )

    def test_performance_page_displays_metrics_and_result_assets(self) -> None:
        app = self.run_app()
        app.radio(key="navigation").set_value("Model Performance").run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.subheader[0].value, "Model performance")
        self.assertIn("Test accuracy", [metric.label for metric in app.metric])
        self.assertIn("Best validation accuracy", [metric.label for metric in app.metric])

        for filename in ("accuracy.png", "loss.png", "confusion_matrix.png"):
            asset = PROJECT_ROOT / "results" / filename
            self.assertTrue(asset.is_file(), f"Missing performance plot: {filename}")
            self.assertGreater(asset.stat().st_size, 0)

    def test_draw_mode_eraser_and_clear_all_do_not_predict(self) -> None:
        app = self.run_app()
        app.radio(key="navigation").set_value("Draw Character").run()

        self.assertEqual(app.session_state["drawing_mode"], "draw")
        app.button(key="drawing_mode_eraser").click().run()
        self.assertEqual(app.session_state["drawing_mode"], "eraser")
        self.assertEqual(app.session_state["eraser_size"], 14)

        app.slider(key="eraser_size").set_value(20).run()
        self.assertEqual(app.session_state["eraser_size"], 20)
        app.button(key="drawing_mode_draw").click().run()
        self.assertEqual(app.session_state["drawing_mode"], "draw")
        self.assertNotIn("last_prediction", app.session_state)

        app.button(key="drawing_mode_eraser").click().run()
        app.button(key="clear_canvas").click().run()
        self.assertEqual(app.session_state["canvas_generation"], 1)
        self.assertEqual(app.session_state["drawing_mode"], "draw")
        self.assertNotIn("last_prediction", app.session_state)
        self.assertNotIn("last_input_image", app.session_state)


if __name__ == "__main__":
    unittest.main()
