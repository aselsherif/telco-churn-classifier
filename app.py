import streamlit as st

from config import ARTIFACTS_PATH, DEFAULT_VALUES, MODEL_PATH, QUICK_CASES
from inference import load_model_files, predict_churn, to_model_customer
from ui import (
    init_state,
    render_form,
    render_quick_actions,
    render_result,
    render_sidebar,
)


@st.cache_resource
def load_files():
    return load_model_files(MODEL_PATH, ARTIFACTS_PATH)


st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📞",
    layout="wide",
)

if not MODEL_PATH.exists() or not ARTIFACTS_PATH.exists():
    st.error(
        "Model files are missing. Add `model/churn_classifier.joblib` and "
        "`model/preprocessing_artifacts.joblib` before running the app."
    )
    st.stop()

model, artifacts = load_files()
init_state(DEFAULT_VALUES)

st.title("Telco Customer Churn Predictor")
st.caption("Estimate whether a customer is likely to leave the service.")
render_sidebar(model)
render_quick_actions(QUICK_CASES, model, artifacts)

submitted, values = render_form()
if submitted:
    customer = to_model_customer(values)
    prediction, probability = predict_churn(customer, model, artifacts)
    st.session_state["prediction_result"] = {
        "prediction": prediction,
        "probability": probability,
        "customer": customer,
        "source": "Manual input",
        "actual": None,
    }

render_result()
