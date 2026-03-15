import streamlit as st
import numpy as np
import pickle
from pathlib import Path


# ------------------------------------------------
# PAGE TITLE
# ------------------------------------------------
st.title("Iris Species Predictor")
st.write(
    "Enter flower measurements below to predict the Iris species using a trained machine learning model."
)


# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------
MODEL_PATH = Path("models/dt_model.pkl")

@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file not found. Place `dt_model.pkl` inside the **models** folder.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


model = load_model()


# ------------------------------------------------
# INPUT SECTION
# ------------------------------------------------
st.subheader("Enter Measurements")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1,
    )

    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
    )

with col2:
    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=0.1,
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=0.2,
        step=0.1,
    )


# ------------------------------------------------
# PREDICTION BUTTON
# ------------------------------------------------
predict = st.button("Predict Species")


# ------------------------------------------------
# PREDICTION LOGIC
# ------------------------------------------------
if predict and model is not None:

    input_data = np.array(
        [[sepal_length, sepal_width, petal_length, petal_width]]
    )

    prediction = model.predict(input_data)

    species = ["Setosa", "Versicolor", "Virginica"]

    predicted_species = species[int(prediction[0])]

    st.success(f"Predicted Iris Species: **{predicted_species}**")


    # ------------------------------------------------
    # SHOW PROBABILITY (IF MODEL SUPPORTS)
    # ------------------------------------------------
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(input_data)[0]

        st.subheader("Prediction Confidence")

        prob_data = {
            "Species": species,
            "Probability": probabilities,
        }

        st.bar_chart(prob_data, x="Species", y="Probability")


# ------------------------------------------------
# INFO SECTION
# ------------------------------------------------
with st.expander("About the Model"):

    st.markdown(
        """
        This prediction model is trained on the **Iris dataset**, one of the most
        famous datasets in machine learning.

        Features used for prediction:

        - Sepal Length
        - Sepal Width
        - Petal Length
        - Petal Width

        Model Type:
        - Decision Tree Classifier
        """
    )