import joblib
import pandas as pd


def load_artifact(path):
    return joblib.load(path)


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


def prepare_input(customer, artifact):
    row = pd.DataFrame([customer])

    for col, mapping in artifact["ordinal_mappings"].items():
        row[col] = row[col].map(mapping)

    encoded = artifact["onehot_encoder"].transform(row[artifact["nominal_cols"]])
    if not isinstance(encoded, pd.DataFrame):
        encoded = pd.DataFrame(
            encoded,
            columns=artifact["onehot_encoder"].get_feature_names_out(
                artifact["nominal_cols"]
            ),
        )

    row = pd.concat(
        [
            row.drop(columns=artifact["nominal_cols"]).reset_index(drop=True),
            encoded.reset_index(drop=True),
        ],
        axis=1,
    )

    row[artifact["standard_cols"]] = artifact["standard_scaler"].transform(
        row[artifact["standard_cols"]]
    )
    row[artifact["robust_cols"]] = artifact["robust_scaler"].transform(
        row[artifact["robust_cols"]]
    )

    return row[artifact["feature_order"]]


def validate_customer(customer, artifact):
    required = set(artifact["input_columns"])
    missing = required - set(customer)
    if missing:
        raise ValueError(f"Missing fields: {sorted(missing)}")

    unexpected = set(customer) - required
    if unexpected:
        raise ValueError(f"Unexpected fields: {sorted(unexpected)}")

    if isinstance(customer["SeniorCitizen"], bool) or customer[
        "SeniorCitizen"
    ] not in (0, 1):
        raise ValueError("SeniorCitizen must be 0 or 1")

    minimum, maximum = artifact["validation"]["tenure"]
    tenure = customer["tenure"]
    if isinstance(tenure, bool) or not isinstance(tenure, (int, float)):
        raise ValueError("tenure must be numeric")
    if tenure < minimum:
        raise ValueError(f"tenure must be >= {minimum}")
    if tenure > maximum:
        raise ValueError(f"tenure must be <= {maximum}")

    for col in ("MonthlyCharges", "TotalCharges"):
        value = customer[col]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{col} must be numeric")
        if value < 0:
            raise ValueError(f"{col} must be >= 0")

    allowed_values = {
        col: set(mapping) for col, mapping in artifact["ordinal_mappings"].items()
    }
    for col, categories in zip(
        artifact["nominal_cols"],
        artifact["onehot_encoder"].categories_,
    ):
        allowed_values[col] = set(categories)

    for col, allowed in allowed_values.items():
        if customer[col] not in allowed:
            raise ValueError(f"Invalid value for {col}: {customer[col]!r}")


def predict_churn(customer, artifact):
    validate_customer(customer, artifact)
    features = prepare_input(customer, artifact)
    model = artifact["model"]
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])
    return prediction, probability


def predict_probability(customer, artifact):
    _, probability = predict_churn(customer, artifact)
    return probability
