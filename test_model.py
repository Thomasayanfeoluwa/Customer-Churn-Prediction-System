import pandas as pd
import pickle
import tensorflow as tf

# Load saved preprocessing objects and model
model = tf.keras.models.load_model("model.h5")

with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


def predict_customer(customer):
    input_df = pd.DataFrame([customer])

    # Encode gender
    input_df["Gender"] = label_encoder_gender.transform(
        input_df["Gender"]
    )

    # Encode geography
    geo_encoded = onehot_encoder_geo.transform(
        input_df[["Geography"]]
    ).toarray()

    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
    )

    # Combine features
    input_df = pd.concat(
        [
            input_df.drop("Geography", axis=1),
            geo_encoded_df
        ],
        axis=1
    )

    expected_columns = [
        "CreditScore",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Geography_France",
        "Geography_Germany",
        "Geography_Spain"
    ]

    input_df = input_df[expected_columns]

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    probability = float(model.predict(input_scaled, verbose=0)[0][0])

    return probability


# Deliberately high-risk customer
bad_customer = {
    "CreditScore": 350,
    "Gender": "Female",
    "Age": 70,
    "Tenure": 0,
    "Balance": 200000,
    "NumOfProducts": 4,
    "HasCrCard": 0,
    "IsActiveMember": 0,
    "EstimatedSalary": 30000,
    "Geography": "Germany"
}

probability = predict_customer(bad_customer)

print("\nHIGH-RISK TEST CUSTOMER")
print("-----------------------")
print(f"Churn probability: {probability:.4%}")
print(f"Stay probability:  {(1 - probability):.4%}")