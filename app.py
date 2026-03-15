import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ------------------------------------------------

# PAGE CONFIGURATION

# ------------------------------------------------

st.set_page_config(
page_title="Data Analytics Dashboard",
page_icon="📊",
layout="wide",
initial_sidebar_state="expanded",
)

# ------------------------------------------------

# HIDE STREAMLIT DEFAULT PAGE NAVIGATION

# ------------------------------------------------

st.markdown(
""" <style>
[data-testid="stSidebarNav"] {display:none;} </style>
""",
unsafe_allow_html=True
)

# ------------------------------------------------

# LOAD CUSTOM CSS

# ------------------------------------------------

def load_css():
css_path = Path("assets/style.css")
if css_path.exists():
with open(css_path) as f:
st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ------------------------------------------------

# SAMPLE DATA

# ------------------------------------------------

@st.cache_data
def sample_data():
data = {
"Name": ["Alice", "Bob", "Charlie"],
"Age": [25, 30, 35],
"City": ["New York", "Los Angeles", "Chicago"],
"Sales": [150, 200, 250],
"Expenses": [100, 150, 200],
}

```
df = pd.DataFrame(data)
df["Profit"] = df["Sales"] - df["Expenses"]

return df
```

# ------------------------------------------------

# LOAD MACHINE LEARNING MODEL

# ------------------------------------------------

@st.cache_resource
def load_model(path):
try:
with open(path, "rb") as f:
return pickle.load(f)
except FileNotFoundError:
st.warning("Model file not found.")
except Exception as e:
st.warning(f"Error loading model: {e}")

```
return None
```

# ------------------------------------------------

# SIDEBAR NAVIGATION

# ------------------------------------------------

st.sidebar.title("Analytics Dashboard")

page = st.sidebar.radio(
"Navigate",
[
"Overview",
"Data Explorer",
"Species Prediction",
"About"
]
)

# =================================================

# OVERVIEW PAGE

# =================================================

if page == "Overview":

```
st.title("Welcome")

st.write(
    "This dashboard provides a clean interface for exploring data, "
    "visualizations, and a predictive model demo."
)

# Metrics
col1, col2, col3 = st.columns(3)

col1.metric("Sales", "$1.2M", "+12%")
col2.metric("Active Users", "3,200", "+8%")
col3.metric("Customer Satisfaction", "92%", "+2%")

# App description
with st.expander("About this Dashboard"):

    st.markdown(
        """
```

* Built using **Streamlit**
* Interactive **data visualization**
* Machine learning **prediction demo**
* Simple **analytics dashboard layout**
  """
  )

  # User form

  with st.form("user_profile"):

  ```
    st.subheader("Tell us about yourself")

    name = st.text_input("Name", st.session_state.get("name", ""))

    age = st.number_input(
        "Age",
        min_value=0,
        max_value=120,
        value=st.session_state.get("age", 25)
    )

    gender = st.radio("Gender", ["Male", "Female", "Other"])

    hobbies = st.multiselect(
        "Hobbies",
        ["Reading", "Traveling", "Gaming", "Cooking", "Sports"],
        default=st.session_state.get("hobbies", [])
    )

    submitted = st.form_submit_button("Save")

    if submitted:
        st.session_state.name = name
        st.session_state.age = age
        st.session_state.gender = gender
        st.session_state.hobbies = hobbies

        st.success("Preferences saved successfully.")
  ```

  if st.session_state.get("name"):
  st.info(f"Hello **{st.session_state.name}**, welcome back!")

# =================================================

# DATA EXPLORER PAGE

# =================================================

elif page == "Data Explorer":

```
st.title("Data Explorer")

df = sample_data()

st.subheader("Dataset Preview")

st.data_editor(
    df,
    use_container_width=True
)

st.subheader("Data Visualization")

col1, col2 = st.columns(2)

with col1:
    st.markdown("Sales")
    st.line_chart(df["Sales"])

with col2:
    st.markdown("Expenses")
    st.bar_chart(df["Expenses"])
```

# =================================================

# SPECIES PREDICTION PAGE

# =================================================

elif page == "Species Prediction":

```
st.title("Iris Species Prediction")

model = load_model(Path("models/dt_model.pkl"))

if model is None:

    st.error(
        "Model not found. Please place **dt_model.pkl** inside the `models` folder."
    )

else:

    with st.form("prediction_form"):

        st.subheader("Enter Flower Measurements")

        col1, col2 = st.columns(2)

        with col1:
            sepal_length = st.number_input(
                "Sepal Length (cm)", 0.0, 10.0, 5.0
            )
            petal_length = st.number_input(
                "Petal Length (cm)", 0.0, 10.0, 1.0
            )

        with col2:
            sepal_width = st.number_input(
                "Sepal Width (cm)", 0.0, 10.0, 3.0
            )
            petal_width = st.number_input(
                "Petal Width (cm)", 0.0, 10.0, 0.2
            )

        submitted = st.form_submit_button("Predict Species")

    if submitted:

        X = np.array(
            [[sepal_length, sepal_width, petal_length, petal_width]]
        )

        prediction = model.predict(X)

        species = [
            "Setosa",
            "Versicolor",
            "Virginica"
        ]

        predicted = species[int(prediction[0])]

        st.success(f"Predicted Species: **{predicted}**")
```

# =================================================

# ABOUT PAGE

# =================================================

elif page == "About":

```
st.title("About This Project")

st.write(
    """
```

This **Analytics Dashboard** demonstrates how data visualization
and machine learning models can be deployed using **Streamlit**.
"""
)

```
st.subheader("Dashboard Features")

st.markdown(
    """
```

* Interactive analytics dashboard
* Data exploration tools
* Visualization charts
* Machine learning prediction
* Modern UI styling
  """
  )

  st.subheader("Technology Stack")

  col1, col2 = st.columns(2)

  with col1:
  st.markdown(
  """
  **Framework**
* Streamlit

**Language**

* Python
  """
  )

  with col2:
  st.markdown(
  """
  **Libraries**
* Pandas
* NumPy
* Scikit-learn
  """
  )

  st.subheader("Machine Learning Model")

  st.markdown(
  """
  This project uses the **Iris dataset** to predict flower species.

Features used:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

Predicted classes:

* Setosa
* Versicolor
* Virginica
  """
  )

  st.subheader("Author")

  st.markdown(
  """
  **Muhammed Faris T M**

Data Science • Analytics • Machine Learning
"""
)

```
st.subheader("Connect")

st.markdown(
    """
```

LinkedIn
http://www.linkedin.com/in/muhammed-faris-tm-ab1233196
"""
)

```
st.write("---")
st.caption("Built with Streamlit • Analytics Dashboard")
```
