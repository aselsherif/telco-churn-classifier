import pandas as pd
import streamlit as st

from inference import predict_churn, to_model_customer

FORM_KEYS = (
    "gender",
    "senior",
    "partner",
    "dependents",
    "tenure",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
    "monthly_charges",
    "total_charges",
)


def init_state(defaults):
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("prediction_result", None)


def inject_css():
    st.markdown(
        """
        <style>
        div[data-testid="stFormSubmitButton"] {
            position: sticky;
            bottom: 0.75rem;
            z-index: 999;
            padding: 0.75rem 0;
            background-color: var(--background-color, #0E1117);
        }
        div[data-testid="stFormSubmitButton"] button {
            background-color: #16A34A !important;
            border-color: #16A34A !important;
            color: #FFFFFF !important;
            box-shadow: 0 6px 16px rgba(22, 163, 74, 0.25);
        }
        div[data-testid="stFormSubmitButton"] button:hover {
            background-color: #15803D !important;
            border-color: #15803D !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(model):
    with st.sidebar:
        st.header("Model")
        st.write("AdaBoost classifier trained on the Telco Customer Churn dataset.")
        st.caption(f"n_estimators = {model.n_estimators}")
        st.caption(f"learning_rate = {model.learning_rate}")
        st.caption("Prediction threshold = 0.50")


def render_quick_actions(quick_cases, artifact):
    st.subheader("Quick predict")
    st.caption(
        "Load a real test customer and predict instantly. "
        "Actual Yes = churned, No = stayed."
    )
    quick_cols = st.columns(5)
    for index, (label, values, actual) in enumerate(quick_cases):
        if quick_cols[index].button(
            label, width="stretch", key=f"quick_case_{index}"
        ):
            for key, value in values.items():
                st.session_state[key] = value

            customer = to_model_customer(values)
            prediction, probability = predict_churn(customer, artifact)
            st.session_state["prediction_result"] = {
                "prediction": prediction,
                "probability": probability,
                "customer": customer,
                "source": label,
                "actual": actual,
            }
            st.session_state["show_result_modal"] = True
            st.rerun()

        actual_text = "Yes" if actual == 1 else "No"
        badge_bg = "#FEE2E2" if actual == 1 else "#DCFCE7"
        badge_fg = "#B91C1C" if actual == 1 else "#15803D"
        quick_cols[index].markdown(
            "<div style='text-align:center;margin-top:-0.35rem;'>"
            f"<span style='background:{badge_bg};color:{badge_fg};"
            "padding:2px 8px;border-radius:999px;font-size:0.72rem;"
            f"font-weight:600;'>Actual: {actual_text}</span>"
            "</div>",
            unsafe_allow_html=True,
        )


def render_form():
    with st.form("churn_form"):
        st.subheader("Customer profile")
        profile_cols = st.columns(4)
        profile_cols[0].selectbox("Gender", ["Female", "Male"], key="gender")
        profile_cols[1].selectbox("Senior citizen", ["No", "Yes"], key="senior")
        profile_cols[2].selectbox("Partner", ["No", "Yes"], key="partner")
        profile_cols[3].selectbox("Dependents", ["No", "Yes"], key="dependents")

        st.subheader("Services")
        service_cols = st.columns(3)
        service_cols[0].selectbox(
            "Phone service", ["Yes", "No"], key="phone_service"
        )
        service_cols[1].selectbox(
            "Multiple lines",
            ["No", "Yes", "No phone service"],
            key="multiple_lines",
        )
        service_cols[2].selectbox(
            "Internet service",
            ["DSL", "Fiber optic", "No"],
            key="internet_service",
        )
        service_cols[0].selectbox(
            "Online security",
            ["No", "Yes", "No internet service"],
            key="online_security",
        )
        service_cols[1].selectbox(
            "Online backup",
            ["No", "Yes", "No internet service"],
            key="online_backup",
        )
        service_cols[2].selectbox(
            "Device protection",
            ["No", "Yes", "No internet service"],
            key="device_protection",
        )
        service_cols[0].selectbox(
            "Tech support",
            ["No", "Yes", "No internet service"],
            key="tech_support",
        )
        service_cols[1].selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"],
            key="streaming_tv",
        )
        service_cols[2].selectbox(
            "Streaming movies",
            ["No", "Yes", "No internet service"],
            key="streaming_movies",
        )

        st.subheader("Account")
        account_cols = st.columns(3)
        account_cols[0].slider("Tenure (months)", 0, 72, key="tenure")
        account_cols[1].selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
            key="contract",
        )
        account_cols[2].selectbox(
            "Paperless billing", ["Yes", "No"], key="paperless_billing"
        )
        account_cols[0].selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            key="payment_method",
        )
        account_cols[1].number_input(
            "Monthly charges ($)",
            min_value=0.0,
            value=None,
            step=0.05,
            key="monthly_charges",
        )
        account_cols[2].number_input(
            "Total charges ($)",
            min_value=0.0,
            value=None,
            step=0.05,
            key="total_charges",
        )

        actions_buttons = st.columns(4)

        submitted = actions_buttons[3].form_submit_button(
            "Predict", type="primary", width="stretch"
        )

    if submitted:
        return True, {key: st.session_state[key] for key in FORM_KEYS}
    return False, None


@st.dialog("Prediction result", width="medium")
def render_result_dialog():
    result = st.session_state.get("prediction_result")
    if not result:
        st.info("No prediction available.")
        return

    if result["prediction"] == 1:
        st.error("This customer is at risk of leaving ⚠️")
    else:
        st.success("This customer is likely to stay ✅")

    st.metric("Estimated churn probability", f"{result['probability']:.1%}")
    st.progress(min(max(result["probability"], 0.0), 1.0))

    if result["source"] != "Manual input":
        st.caption(f"Quick case: {result['source']}")

    if result.get("actual") is not None:
        actual_label = "Churned" if result["actual"] == 1 else "Stayed"
        comparison = (
            "matches ✅"
            if result["prediction"] == result["actual"]
            else "differs ❌"
        )
        st.caption(
            f"Actual outcome: **{actual_label}** · Prediction {comparison}"
        )

    with st.expander("Review input data"):
        st.dataframe(
            pd.DataFrame([result["customer"]]),
            width="stretch",
            hide_index=True,
        )
