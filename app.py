"""Streamlit interface for handwritten character recognition."""

import json
from pathlib import Path

import streamlit as st
from streamlit_drawable_canvas import st_canvas

from model.cnn_model import CharacterCNN
from utils.labels import EMNIST_BALANCED_LABELS
from utils.prediction import load_model, predict_image
from utils.preprocessing import ImageInput, load_image


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"

PAGES = (
    "Home Dashboard",
    "Draw Character",
    "Upload Image",
    "Prediction Results",
    "Model Performance",
    "About Project",
)


def load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="Loading trained CNN...")
def get_model() -> CharacterCNN:
    return load_model()


def run_prediction(
    source: ImageInput,
    original_image: ImageInput,
    source_name: str,
) -> None:
    try:
        with st.spinner("Preprocessing image and running CNN inference..."):
            result = predict_image(source, get_model())
    except (OSError, RuntimeError, ValueError) as error:
        st.error(f"Could not predict this {source_name}: {error}")
        return

    st.session_state["last_prediction"] = result
    st.session_state["last_input_image"] = original_image
    st.session_state["navigate_to_results"] = True
    st.rerun()


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --accent-purple: #9b7bff;
            --accent-blue: #55b7ff;
            --muted-text: #a7aec4;
            --panel: rgba(25, 29, 48, 0.88);
        }
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(ellipse at 12% 0%, rgba(111, 66, 193, .22), transparent 34%),
                radial-gradient(ellipse at 88% 12%, rgba(42, 127, 214, .18), transparent 32%),
                #0c0e18;
            color: #f5f4fb;
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #151426 0%, #101522 100%);
            border-right: 1px solid rgba(164, 143, 255, .16);
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            border-radius: 10px;
            padding: 5px 8px;
        }
        .block-container { max-width: 1280px; padding-top: 2rem; }
        h1, h2, h3 { letter-spacing: -0.025em; }
        .app-kicker {
            color: var(--accent-blue);
            font-size: .78rem;
            font-weight: 700;
            letter-spacing: .16em;
            text-transform: uppercase;
        }
        .hero-panel, .info-panel {
            background: linear-gradient(135deg, rgba(42, 35, 75, .88), var(--panel));
            border: 1px solid rgba(164, 143, 255, .2);
            border-radius: 18px;
            padding: 1.4rem 1.6rem;
            margin: .4rem 0 1.2rem;
        }
        .hero-panel {
            background:
                radial-gradient(circle at 88% 16%, rgba(85, 183, 255, .17), transparent 28%),
                linear-gradient(120deg, rgba(72, 47, 133, .78), rgba(23, 33, 59, .94));
        }
        .hero-panel p, .info-panel p, .muted-copy { color: var(--muted-text); }
        .hero-panel p { font-size: 1.05rem; line-height: 1.65; }
        div.stButton > button {
            border: 1px solid rgba(172, 151, 255, .45);
            border-radius: 10px;
            background: linear-gradient(100deg, #7654d9, #427fd4);
            color: white;
            font-weight: 650;
            min-height: 2.7rem;
        }
        div.stButton > button:hover {
            border-color: #b8a5ff;
            color: white;
            filter: brightness(1.08);
        }
        .st-key-drawing-mode-controls div.stButton > button {
            border-radius: 12px;
            min-height: 2.8rem;
        }
        .st-key-drawing-mode-controls div.stButton > button[kind="primary"] {
            border-color: #c4b5ff;
            box-shadow: 0 0 0 1px rgba(85, 183, 255, .32);
        }
        .st-key-drawing-mode-controls div.stButton > button[kind="secondary"] {
            background: rgba(25, 29, 48, .82);
            border-color: rgba(164, 143, 255, .28);
            color: #d5d6e5;
        }
        [data-testid="stMetric"] {
            background: rgba(25, 29, 48, .82);
            border: 1px solid rgba(164, 143, 255, .17);
            border-radius: 14px;
            padding: 1rem;
        }
        [data-testid="stProgressBar"] > div > div {
            background-image: linear-gradient(90deg, #8b65f6, #52b7ff);
        }
        iframe[title="streamlit_drawable_canvas.st_canvas"] {
            width: 320px !important;
            max-width: 100%;
            height: 324px !important;
            border: 1px solid rgba(164, 143, 255, .2);
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        '<div class="app-kicker">AI-powered character classification</div>',
        unsafe_allow_html=True,
    )
    st.title("HANDWRITTEN CHARACTER RECOGNITION")
    st.caption("AI-Powered Character Classification Using Deep Learning")


def render_home() -> None:
    st.subheader("Project dashboard")
    st.markdown(
        """
        <div class="hero-panel">
            <h3>Turn handwritten characters into predictions</h3>
            <p>
                Draw a character or upload an image and explore how a trained
                convolutional neural network processes it. The model was trained
                on EMNIST Balanced and supports 47 digit and letter classes.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics = load_json(RESULTS_DIR / "evaluation.json")
    history = load_json(RESULTS_DIR / "training_history.json")
    if metrics:
        test_accuracy = metrics["test_accuracy"]
        sample_count = metrics["test_sample_count"]
        train_accuracy = history["train_accuracy"][-1] if history else None
        columns = st.columns(3)
        columns[0].metric("Test accuracy", f"{test_accuracy:.2%}")
        if train_accuracy is not None:
            columns[1].metric("Training accuracy", f"{train_accuracy:.2%}")
        else:
            columns[1].metric("Training accuracy", "Not available")
        columns[2].metric("Test samples", f"{sample_count:,}")
        st.caption("Metrics shown are loaded from this project's measured evaluation output.")
    else:
        st.info(
            "Evaluation results are not available yet. Train and evaluate the model "
            "to display measured metrics."
        )

    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            <div class="info-panel">
                <h3>Project objective</h3>
                <p>Recognize one handwritten digit or letter from a drawing or image,
                and show the model's most likely classes with their confidence.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="info-panel">
                <h3>Model and dataset</h3>
                <p>Two convolutional feature-extraction blocks feed a fully connected
                classifier. Training uses EMNIST Balanced, with a separate held-out
                test split for evaluation.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("**Supported classes**")
    st.code(" ".join(EMNIST_BALANCED_LABELS), language=None)


def render_draw_page() -> None:
    st.subheader("Draw a character")
    st.markdown(
        '<p class="muted-copy">Draw one character in the center of the canvas. '
        "Use a clear, single stroke pattern for the best input.</p>",
        unsafe_allow_html=True,
    )
    if "drawing_mode" not in st.session_state:
        st.session_state.drawing_mode = "draw"
    if "canvas_generation" not in st.session_state:
        st.session_state.canvas_generation = 0

    with st.container(key="drawing_mode_controls"):
        draw_column, eraser_column, clear_column = st.columns(3)
        with draw_column:
            if st.button(
                "Draw",
                type=(
                    "primary"
                    if st.session_state.drawing_mode == "draw"
                    else "secondary"
                ),
                key="drawing_mode_draw",
            ):
                st.session_state.drawing_mode = "draw"
        with eraser_column:
            if st.button(
                "Eraser",
                type=(
                    "primary"
                    if st.session_state.drawing_mode == "eraser"
                    else "secondary"
                ),
                key="drawing_mode_eraser",
            ):
                st.session_state.drawing_mode = "eraser"
        with clear_column:
            if st.button("Clear All", key="clear_canvas"):
                st.session_state.canvas_generation += 1
                st.session_state.drawing_mode = "draw"
                st.session_state.pop("last_prediction", None)
                st.session_state.pop("last_input_image", None)
                st.rerun()

    is_eraser = st.session_state.drawing_mode == "eraser"
    if is_eraser:
        stroke_width = st.slider(
            "Eraser size",
            min_value=4,
            max_value=28,
            value=14,
            key="eraser_size",
        )
        stroke_color = "#10121B"
    else:
        stroke_width = st.slider(
            "Brush size",
            min_value=4,
            max_value=28,
            value=14,
            key="brush_size",
        )
        stroke_color = "#FFFFFF"

    canvas_column, guide_column = st.columns([1.05, 1])
    with canvas_column:
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color="#10121B",
            height=320,
            width=320,
            drawing_mode="freedraw",
            update_streamlit=True,
            display_toolbar=False,
            key=f"character_canvas_{st.session_state.canvas_generation}",
        )
    with guide_column:
        st.markdown(
            """
            <div class="info-panel">
                <h3>Drawing tips</h3>
                <p>Draw one character with a bright stroke on the dark canvas.
                Keep it near the center and avoid extra marks.</p>
                <p>Switch to Eraser to remove unwanted strokes and adjust its
                size. Use Clear All to reset the canvas.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    _, predict_column = st.columns([1, 2])
    with predict_column:
        if st.button("Predict character", type="primary", key="predict_canvas"):
            if canvas_result.image_data is None:
                st.warning("Draw a character before requesting a prediction.")
            else:
                run_prediction(
                    canvas_result.image_data,
                    canvas_result.image_data,
                    "drawing",
                )


def render_upload_page() -> None:
    st.subheader("Upload a character image")
    st.markdown(
        '<p class="muted-copy">Choose a PNG or JPG image containing one handwritten '
        "character. Light and dark backgrounds are supported.</p>",
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
        key="character_upload",
    )
    if uploaded_file is None:
        return

    image_bytes = uploaded_file.getvalue()
    try:
        image = load_image(image_bytes)
    except ValueError as error:
        st.error(str(error))
        return

    st.image(image, caption="Uploaded image", width=320)
    if st.button("Predict character", type="primary", key="predict_upload"):
        run_prediction(image_bytes, image, "uploaded image")


def render_prediction_page() -> None:
    st.subheader("Prediction results")
    result = st.session_state.get("last_prediction")
    if result is None:
        st.info("Draw or upload a character to see its prediction results here.")
        return

    top_prediction = result.top_predictions[0]
    original_image = st.session_state.get("last_input_image")
    image_column, detail_column = st.columns([1, 1.35])
    with image_column:
        if original_image is not None:
            st.image(original_image, caption="Original input", width="stretch")
        st.image(result.processed_image, caption="Processed 28x28 input", width=196)
    with detail_column:
        st.metric(
            "Predicted character",
            top_prediction.character,
            f"{top_prediction.confidence:.2%} confidence",
        )
        st.markdown("**Top 3 predictions**")
        for rank, prediction in enumerate(result.top_predictions, start=1):
            st.write(f"{rank}. **{prediction.character}** — {prediction.confidence:.2%}")
            st.progress(prediction.confidence)


def render_performance_page() -> None:
    st.subheader("Model performance")
    metrics = load_json(RESULTS_DIR / "evaluation.json")
    history = load_json(RESULTS_DIR / "training_history.json")
    if metrics is None:
        st.warning("Evaluation metrics are missing. Run `python -m training.evaluate` after training.")
        return

    metric_columns = st.columns(3)
    metric_columns[0].metric("Test accuracy", f"{metrics['test_accuracy']:.2%}")
    metric_columns[1].metric("Test samples", f"{metrics['test_sample_count']:,}")
    if history:
        metric_columns[2].metric(
            "Best validation accuracy",
            f"{history['best_validation_accuracy']:.2%}",
        )

    accuracy_path = RESULTS_DIR / "accuracy.png"
    loss_path = RESULTS_DIR / "loss.png"
    left, right = st.columns(2)
    if accuracy_path.is_file():
        left.image(str(accuracy_path), caption="Training and validation accuracy")
    if loss_path.is_file():
        right.image(str(loss_path), caption="Training and validation loss")

    confusion_path = RESULTS_DIR / "confusion_matrix.png"
    if confusion_path.is_file():
        st.image(str(confusion_path), caption="Confusion matrix for all 47 test classes")

    report = metrics.get("classification_report", {})
    report_rows = [
        {"Class": name, **values}
        for name, values in report.items()
        if isinstance(values, dict)
    ]
    if report_rows:
        with st.expander("Classification report", expanded=False):
            st.dataframe(report_rows, width="stretch", hide_index=True)


def render_about_page() -> None:
    st.subheader("About this project")
    st.markdown(
        """
        ### What is a CNN?
        A convolutional neural network learns visual patterns from small regions
        of an image. Early layers can respond to edges and curves; later layers
        combine those patterns to distinguish character classes.

        ### Why use a CNN?
        Handwritten characters vary in size, stroke thickness, and shape. A CNN
        learns spatial features directly from pixels and is well suited to image
        classification.

        ### Dataset and workflow
        EMNIST Balanced contains 47 digit and letter classes. The training split
        is divided into training and validation subsets; the separate test split
        is reserved for final evaluation. Images are converted to grayscale,
        oriented, normalized, and resized to 28x28.

        ### CNN architecture
        The network uses two convolution → ReLU → max-pooling blocks, followed
        by a flattened feature vector, a 128-unit fully connected layer, dropout,
        and a 47-class output layer. Cross-entropy loss and Adam are used during
        training.

        ### Technologies
        Python 3.11, PyTorch, Torchvision, OpenCV, NumPy, Pillow, Streamlit,
        Streamlit Drawable Canvas, Matplotlib, and scikit-learn.
        """
    )


def main() -> None:
    st.set_page_config(
        page_title="Handwritten Character Recognition",
        page_icon="H",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_styles()
    if st.session_state.pop("navigate_to_results", False):
        st.session_state["navigation"] = "Prediction Results"
    with st.sidebar:
        st.markdown("### Character AI")
        st.caption("EMNIST Balanced · 47 classes")
        selected_page = st.radio(
            "Navigation",
            PAGES,
            label_visibility="collapsed",
            key="navigation",
        )

    render_header()
    page_renderers = {
        "Home Dashboard": render_home,
        "Draw Character": render_draw_page,
        "Upload Image": render_upload_page,
        "Prediction Results": render_prediction_page,
        "Model Performance": render_performance_page,
        "About Project": render_about_page,
    }
    page_renderers[selected_page]()


if __name__ == "__main__":
    main()
