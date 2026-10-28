import joblib
import pandas as pd


def load_model_files(model_path, artifacts_path):
    model = joblib.load(model_path)
    artifacts = joblib.load(artifacts_path)
    return model, artifacts


def prepare_input(customer, artifacts):
    row = pd.DataFrame([customer])

    for col, mapping in artifacts["ordinal_mappings"].items():
        row[col] = row[col].map(mapping)

    encoded = artifacts["onehot_encoder"].transform(row[artifacts["nominal_cols"]])
    if not isinstance(encoded, pd.DataFrame):
        encoded = pd.DataFrame(
            encoded,
            columns=artifacts["onehot_encoder"].get_feature_names_out(artifacts["nominal_cols"]),
        )

    row = pd.concat(
        [
            row.drop(columns=artifacts["nominal_cols"]).reset_index(drop=True),
            encoded.reset_index(drop=True),
        ],
        axis=1,
    )

    row[artifacts["standard_cols"]] = artifacts["standard_scaler"].transform(
        row[artifacts["standard_cols"]]
    )
    row[artifacts["robust_cols"]] = artifacts["robust_scaler"].transform(
        row[artifacts["robust_cols"]]
    )

    return row[artifacts["feature_order"]]


def to_model_customer(values):
    return {
        "gender": values["gender"],
        "SeniorCitizen": 1 if values["senior"] == "Yes" else 0,
        "Partner": values["partner"],
        "Dependents": values["dependents"],
        "tenure": values["tenure"],
        "PhoneService": values["phone_service"],
        "MultipleLines": values["multiple_lines"],
        "InternetService": values["internet_service"],
        "OnlineSecurity": values["online_security"],
        "OnlineBackup": values["online_backup"],
        "DeviceProtection": values["device_protection"],
        "TechSupport": values["tech_support"],
        "StreamingTV": values["streaming_tv"],
        "StreamingMovies": values["streaming_movies"],
        "Contract": values["contract"],
        "PaperlessBilling": values["paperless_billing"],
        "PaymentMethod": values["payment_method"],
        "MonthlyCharges": values["monthly_charges"],
        "TotalCharges": values["total_charges"],
    }


def predict_churn(customer, model, artifacts):
    features = prepare_input(customer, artifacts)
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])
    return prediction, probability
