import streamlit as st

from config import ARTIFACT_PATH, DEFAULT_VALUES, QUICK_CASES
from inference import load_artifact, predict_churn, to_model_customer
from ui import (
    init_state,
    inject_css,
    render_form,
    render_quick_actions,
    render_result_dialog,
    render_sidebar,
)


@st.cache_resource
def load_files():
    return load_artifact(ARTIFACT_PATH)


st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📞",
    layout="wide",
)

inject_css()

if not ARTIFACT_PATH.exists():
    st.error(
        "Deployment artifact is missing. Run the notebook export cell to create "
        "`model/deployment_artifact.joblib`."
    )
    st.stop()

artifact = load_files()
init_state(DEFAULT_VALUES)

st.title("Telco Customer Churn Predictor")
st.caption("Estimate whether a customer is likely to leave the service.")
render_sidebar(artifact["model"])
render_quick_actions(QUICK_CASES, artifact)

submitted, values = render_form()
if submitted:
    customer = to_model_customer(values)
    prediction, probability = predict_churn(customer, artifact)
    st.session_state["prediction_result"] = {
        "prediction": prediction,
        "probability": probability,
        "customer": customer,
        "source": "Manual input",
        "actual": None,
    }
    st.session_state["show_result_modal"] = True

if st.session_state.pop("show_result_modal", False):
    render_result_dialog()
