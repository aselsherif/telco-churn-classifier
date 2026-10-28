# Telco Customer Churn Predictor

A Streamlit app that estimates whether a telecom customer is likely to churn, using an AdaBoost classifier trained on the Telco Customer Churn dataset.

## Features

- Manual prediction form for all 19 customer features.
- Five quick-prediction buttons loaded from real test-set customers.
- Prediction result shown in a modal with churn probability.
- Quick predictions also show the actual outcome for comparison.
- Dark theme with a sticky Predict button.

## Project structure

```text
.
├── app.py                         # Streamlit entrypoint
├── config.py                      # Paths, quick cases, default form values
├── inference.py                   # Validation, preprocessing, prediction
├── ui.py                          # Sidebar, form, quick actions, result modal
├── requirements.txt
├── .streamlit/
│   └── config.toml                # Dark theme
├── model/
│   ├── deployment_artifact.joblib # Model + preprocessing objects used by the app
│   └── churn_classifier.joblib    # Model-only export
└── preprocessing/                 # Training notebook and datasets (local only)
```

The app loads only `model/deployment_artifact.joblib`. It does not need the raw CSV or the notebook at runtime.

## Requirements

- Python 3.12 or newer
- `pip`

## Installation

```bash
git clone <your-repository-url>
cd telco-churn-classifier
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run locally

From the project root:

```bash
streamlit run app.py
```

Open the URL printed in the terminal, usually:

```text
http://localhost:8501
```

For development with auto-reload:

```bash
streamlit run app.py --server.runOnSave true
```

## How to use the app

### Quick prediction

1. Click one of the five quick-action buttons.
2. The form is filled with that customer's details.
3. The prediction modal opens and shows:
   - Whether the customer is likely to churn.
   - The estimated churn probability.
   - The actual outcome and whether the prediction matches.

### Manual prediction

1. Fill in the customer profile, services, and account fields.
2. Click **Predict**.
3. Review the result in the modal.
4. Close the modal and adjust the inputs to run another prediction.

The result is a model estimate, not a guaranteed outcome.

## Deploy on Streamlit Community Cloud

1. Push the project to GitHub.
2. Make sure these files are committed:
   - `app.py`
   - `config.py`
   - `inference.py`
   - `ui.py`
   - `requirements.txt`
   - `.streamlit/config.toml`
   - `model/deployment_artifact.joblib`
3. Go to https://share.streamlit.io and sign in with GitHub.
4. Click **New app** and select the repository and branch.
5. Set the main file path to `app.py`.
6. Use Python 3.12 if available.
7. Deploy and check the build logs.

Do not add local module names such as `inference` to `requirements.txt`. They are project files, not PyPI packages.

## Regenerate the deployment artifact

The training workflow lives in `preprocessing/project_1.ipynb`.

1. Open the notebook and run the cells in order.
2. Run the final **Export deployment artifact** cell.
3. Confirm that `model/deployment_artifact.joblib` was created.
4. Restart the Streamlit app.

The artifact contains the fitted model, ordinal mappings, one-hot encoder, scalers, feature order, input columns, threshold, and validation metadata.

## Troubleshooting

**`ModuleNotFoundError: No module named 'inference'`**

`inference.py` must exist in the project root. Do not install an `inference` package from PyPI.

**Deployment fails while installing requirements**

Use the pinned versions in `requirements.txt` and avoid adding notebook-only packages. Reboot the app after updating dependencies.

**Model artifact is missing**

Run the notebook export cell to create `model/deployment_artifact.joblib`.

## Notes

- The app expects the same preprocessing used during training. Do not change the artifact manually.
- The five quick cases are hardcoded samples from the test set.
- This project is for educational and demonstration purposes.
