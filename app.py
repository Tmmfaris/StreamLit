import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path


# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Data Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------
# LOAD CSS STYLE
# ------------------------------------------------
def load_css():
    css_file = Path("assets/style.css")
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()


# ------------------------------------------------
# SAMPLE DATA
# ------------------------------------------------
@st.cache_data
def sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Name": ["Alice", "Bob", "Charlie"],
            "Age": [25, 30, 35],
            "City": ["New York", "Los Angeles", "Chicago"],
            "Sales": [150, 200, 250],
            "Expenses": [100, 150, 200],
        }
    )


# ------------------------------------------------
# LOAD ML MODEL
# ------------------------------------------------
def load_model(path: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.warning(f"Model file not found: {path}")
    except Exception as e:
        st.warning(f"Unable to load model: {e}")
    return None


# ------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------
st.sidebar.title("Analytics Dashboard")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Data Explorer", "Species Prediction", "About"],
    index=0,
)


# ------------------------------------------------
# OVERVIEW PAGE
# ------------------------------------------------
if page == "Overview":

    st.title("Welcome")

    st.write(
        "This dashboard provides a clean interface for exploring data, visualizations, and a predictive model demo."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Sales", "$1.2M", "+12%")
    col2.metric("Active Users", "3,200", "+8%")
    col3.metric("Satisfaction", "92%", "+2%")

    with st.expander("About this app"):

        st.markdown(
            """
            - Uses **Streamlit modern layout**
            - Sidebar navigation
            - Interactive forms
            - Data visualization
            - Machine learning prediction demo
            """
        )

    with st.form("user_profile", clear_on_submit=False):

        st.subheader("Tell us about yourself")

        name = st.text_input("Name", st.session_state.get("name", ""))

        age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            value=st.session_state.get("age", 25),
        )

        gender = st.radio("Gender", ["Male", "Female", "Other"], index=0)

        hobbies = st.multiselect(
            "Hobbies",
            ["Reading", "Traveling", "Gaming", "Cooking", "Sports"],
            default=st.session_state.get("hobbies", []),
        )

        submitted = st.form_submit_button("Save")

        if submitted:
            st.session_state.name = name
            st.session_state.age = age
            st.session_state.gender = gender
            st.session_state.hobbies = hobbies

            st.success("Your preferences were saved.")

    if st.session_state.get("name"):
        st.info(f"Hello {st.session_state.name}! Thanks for visiting.")


# ------------------------------------------------
# DATA EXPLORER
# ------------------------------------------------
elif page == "Data Explorer":

    st.title("Sample Data")

    df = sample_data()

    st.data_editor(df, use_container_width=True)

    st.markdown("### Charts")

    col1, col2 = st.columns(2)

    col1.line_chart(df["Sales"])
    col2.bar_chart(df["Expenses"])


# ------------------------------------------------
# IRIS PREDICTION PAGE
# ------------------------------------------------
elif page == "Species Prediction":

    st.title("Iris Species Predictor")

    model = load_model(Path("models/dt_model.pkl"))

    if model is None:

        st.error("Model not found. Add `dt_model.pkl` to the models folder.")

    else:

        with st.form("iris_form"):

            st.subheader("Enter measurements")

            col1, col2 = st.columns(2)

            sepal_length = col1.number_input(
                "Sepal length (cm)", 0.0, 10.0, 5.0
            )

            sepal_width = col2.number_input(
                "Sepal width (cm)", 0.0, 10.0, 3.0
            )

            petal_length = col1.number_input(
                "Petal length (cm)", 0.0, 10.0, 1.0
            )

            petal_width = col2.number_input(
                "Petal width (cm)", 0.0, 10.0, 0.2
            )

            submitted = st.form_submit_button("Predict")

        if submitted:

            X = np.array(
                [[sepal_length, sepal_width, petal_length, petal_width]]
            )

            prediction = model.predict(X)

            species = ["Setosa", "Versicolor", "Virginica"]

            st.success(
                f"Predicted species: **{species[int(prediction[0])] }**"
            )


# ------------------------------------------------
# ABOUT PAGE
# ------------------------------------------------
elif page == "About":

    st.title("About")

    st.write(
        "This modern Streamlit application demonstrates a clean dashboard structure, sidebar navigation, and machine learning inference."
    )

    st.markdown(
        """
        **Features**

        - Interactive analytics dashboard
        - Data exploration tools
        - Visualization charts
        - Machine learning prediction
        - Modern UI styling
        """
    )

    st.write("---")

    st.caption(
        "Tip: You can add more pages by extending the sidebar navigation."
    )