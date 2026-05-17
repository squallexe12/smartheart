# SmartHeart

> Heart disease risk classifier built from scratch in NumPy, with a premium static web demo that runs entirely in the browser.

![Accuracy](https://img.shields.io/badge/test_accuracy-88.1%25-brightgreen) ![AUC](https://img.shields.io/badge/AUC-0.95-brightgreen) ![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

## What this is

A logistic regression model that predicts the probability of heart disease from 13 clinical features. The defining constraint: the learning algorithm — sigmoid, binary cross-entropy loss, gradient descent — is implemented by hand using only NumPy. No scikit-learn, PyTorch, or TensorFlow in the model code.

A trained model is exported as `web/public/model.json`. The static web app mirrors the preprocessing and inference logic in vanilla JavaScript, so it runs entirely in the user's browser with no backend.

## Repository layout

```
smartheart/                 # NumPy-only library
  model.py                  # LogisticRegressionScratch + sigmoid + bce_loss
  preprocessing.py          # standardization, split, one-hot
  metrics.py                # accuracy, precision, recall, F1, ROC, AUC
scripts/
  download_data.py          # Fetch UCI Cleveland dataset
  train.py                  # Train and export artifacts
tests/                      # pytest suite (36 tests)
web/                        # Static frontend (HTML + Tailwind CDN + vanilla JS)
  public/model.json         # Browser-consumable model artifact
data/raw/heart.csv          # UCI Cleveland dataset
docs/superpowers/           # Design spec and implementation plan
```

## The math

Hypothesis: $\hat{y} = \sigma(Xw + b)$ where $\sigma(z) = 1 / (1 + e^{-z})$.

Loss (binary cross-entropy):
$\mathcal{L} = -\frac{1}{m}\sum [y \log \hat{y} + (1-y)\log(1-\hat{y})]$

Gradients:
$\partial\mathcal{L}/\partial w = \frac{1}{m} X^T(\hat{y}-y)$, $\partial\mathcal{L}/\partial b = \frac{1}{m}\sum(\hat{y}-y)$

Update: $w \leftarrow w - \alpha \nabla_w$, $b \leftarrow b - \alpha \nabla_b$.

## Quick start

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

python scripts/download_data.py       # downloads UCI dataset
python scripts/train.py               # trains model, exports artifacts
pytest -v                             # runs the test suite (36 tests)

# Serve the web demo locally:
python -m http.server 8000 --directory web
# Open http://localhost:8000
```

> On Windows / some shells, `scripts/train.py` may need to be run with `PYTHONPATH=. python scripts/train.py` from the project root so the `smartheart` package resolves.

## Results

Trained on the UCI Heart Disease (Cleveland) dataset, 297 rows after cleaning, 80/20 stratified split, seed 42.

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.881  |
| Precision | 0.857  |
| Recall    | 0.889  |
| F1        | 0.873  |
| AUC       | 0.955  |

Re-run `python scripts/train.py` to reproduce.

## Architecture

Two cleanly separated layers, communicating via a JSON model artifact:

1. **`smartheart/` (Python, NumPy-only)** — the model, preprocessing, and metrics, all implemented by hand. `scripts/train.py` trains on UCI Cleveland and exports both a Python pickle (`models/scratch_model.pkl`) and a JSON artifact (`web/public/model.json`).
2. **`web/` (static HTML + JS)** — loads `model.json` and mirrors the preprocessing + sigmoid in vanilla JavaScript. Inference happens entirely in the browser. A parity test (`tests/test_artifact.py`) confirms JS-equivalent inference matches the Python pickle to floating-point tolerance.

## Limitations

- 297 samples after cleaning — small dataset, limited generalization claims.
- Vanilla gradient descent, no regularization, no momentum.
- Cleveland-only subset of UCI (the cleanest of the 4 hospital subsets).
- **Not a medical device.** This is an academic exercise. Real diagnosis requires validated clinical-grade models, regulatory approval, and clinician interpretation.

## License

MIT.

## Author

Kaan — UCAS 2026/2027 applicant, Computer Science & AI.
GitHub: [@squallexe12](https://github.com/squallexe12)
