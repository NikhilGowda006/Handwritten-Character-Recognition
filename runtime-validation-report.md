# Runtime Validation Report

**Updated:** 2026-11-14 — original thin-stroke upload reproduced and validated  
**Target:** `C:\Users\HP\OneDrive\Desktop\Handwrite Recognization`

## Summary

| Step | Status | Exit Code | Details |
|---|---|---:|---|
| Startup | PASS | 0 | Streamlit started; `/_stcore/health` returned HTTP 200 (`ok`) within the 30-second readiness window. |
| Automated tests | PASS | 0 | 15 tests across five files; all passed after stroke-width augmentation. |
| Browser E2E | PASS | 0 | Draw/upload/error flows passed in the prior UI validation; after retraining, AppTest rechecked current dashboard/performance metrics and the live health endpoint returned HTTP 200. |
| Dependency check | PASS | 0 | `pip check` reported no broken requirements. |
| Python syntax | PASS | 0 | `compileall` completed successfully for application and test modules. |

**Overall:** PASS

## Validation Environment

- Python: 3.11.9
- Node.js: available, v26.2.0
- Docker: unavailable; `docker` command not found. This application has no database or containerized infrastructure dependency, so no infrastructure fallback was needed.
- Browser tier: integrated Playwright browser runner, available and used against the running Streamlit app. No local browser installation was required.
- Test command: `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`

## Prediction-Accuracy Diagnosis

- The EMNIST Balanced test split contains 18,800 samples (400 per class); its
  47 class names and integer target indices exactly match
  `EMNIST_BALANCED_LABELS`.
- Training applies `EMNISTOrientation` (transpose) to upright characters.
  Actual EMNIST test images passed through the same orientation and then
  `predict_image` produced the expected top prediction for representative
  `7`, `T`, `L`, `I`, and `1` samples.
- Before the fix, training and direct test evaluation used orientation,
  `ToTensor`, and normalization only. Inference additionally thresholded,
  cropped, scaled, denoised, and centered images. These are observably
  different model inputs.
- On the full held-out test set, the original checkpoint scored 87.18% with
  the old training transform, but 78.88% when inputs used the application's
  inference preprocessing. Under that app path, digit `7` recall was 62.25%
  and 100 of 400 sevens were predicted as `T`.
- Training, validation, and test now call the same `preprocess_image`
  function used by prediction. A regression test compares these tensors
  exactly on actual EMNIST samples for `7`, `T`, `L`, `I`, and `1`.
- The user supplied the original 1080x1225 RGBA upload. The current pipeline
  reproduced the screenshot exactly (`F` 32.21%, `f` 19.72%, `P` 12.65%).
  Its processed glyph had only 86 nonzero pixels (peak intensity 208); held-out
  EMNIST `B` samples had a minimum of 119 and median of 215 nonzero pixels.
- Unconditional 3x3 dilation changed this individual upload to `B` 84.68%,
  but reduced full held-out accuracy to 75.51%, so inference-time dilation was
  rejected.
- Training-only stroke-width augmentation (random 3x3 thickening on 15% and
  thinning on 15% of training examples) kept validation and inference
  preprocessing unchanged. The retained checkpoint now predicts the original
  upload as `B` 99.15% (`8` and `D` are next), with 87.05% best validation
  accuracy and 86.46% held-out accuracy. The latter is 0.29 percentage points
  below the prior preprocessing-aligned checkpoint (86.75%); macro precision,
  recall, and F1 are 86.54%, 86.46%, and 86.27%.
- Full before/after metrics are in
  [`results/model_debug_comparison.json`](./results/model_debug_comparison.json).

## Test Evidence

Automated suite: **15 passed, 0 failed, 0 skipped**.

- [tests/test_model.py](./tests/test_model.py): CNN output shape, trained checkpoint loading, missing-checkpoint error, and local EMNIST Balanced loading.
- [tests/test_preprocessing.py](./tests/test_preprocessing.py): light-background inversion, RGBA canvas input, blank-image rejection, and corrupt upload rejection.
- [tests/test_prediction.py](./tests/test_prediction.py): real checkpoint inference for canvas-array and uploaded-PNG inputs; top-three ordering, labels, and confidences; upright EMNIST test samples for `7`, `T`, `L`, `I`, and `1`.
- [tests/test_model.py](./tests/test_model.py): exact training/inference tensor equality and cached-tensor normalization on actual EMNIST samples for those same classes.
- [tests/test_streamlit_app.py](./tests/test_streamlit_app.py): app startup, measured dashboard metrics, performance metrics, and plot assets.
- [tests/test_training.py](./tests/test_training.py): deterministic thinning, thickening, and unchanged-image augmentation behavior.

Browser flows:

1. Streamlit AppTest displayed the latest measured test accuracy, training accuracy, and test sample count from the updated evaluation data.
2. Drawing an A-like canvas sample completed inference and displayed three ranked classes.
3. Uploading a PNG completed inference and displayed three ranked classes.
4. Blank and corrupt images displayed explicit validation errors.
5. The performance page displayed its accuracy and loss plots, confusion matrix, measured metrics, and classification-report expander.

No authentication or database persistence flows apply: the application has no authentication or database, and predictions are held in Streamlit session state.

## Issue Fixed

Training, validation, and testing previously bypassed the deployed preprocessing
pipeline. The shared transform is now used throughout. A later real upload
exposed a separate thin-stroke generalization issue; training-only stroke-width
augmentation improved that sample without changing prediction preprocessing,
labels, architecture, or UI. Regression tests cover preprocessing parity and
the augmentation operations.
